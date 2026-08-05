"""Execution, aggregation, and comparison of benchmark runs."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from cognitivetree.benchmark.suite import BenchmarkTask, default_suite
from cognitivetree.config import SearchConfig
from cognitivetree.observability.metrics import RunMetrics
from cognitivetree.search import SearchOutcome


@dataclass(frozen=True, slots=True)
class TaskResult:
    """Outcome of one task under one seed."""

    task: str
    seed: int | None
    outcome: str
    solved: bool
    iterations: int
    nodes: int
    wall_seconds: float
    tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "seed": self.seed,
            "outcome": self.outcome,
            "solved": self.solved,
            "iterations": self.iterations,
            "nodes": self.nodes,
            "wall_seconds": round(self.wall_seconds, 6),
            "tokens": self.tokens,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Aggregate of every task-and-seed combination in one benchmark run."""

    label: str
    results: tuple[TaskResult, ...]

    @property
    def runs(self) -> int:
        return len(self.results)

    @property
    def solved(self) -> int:
        return sum(1 for r in self.results if r.solved)

    @property
    def solve_rate(self) -> float:
        return self.solved / self.runs if self.results else 0.0

    @property
    def mean_iterations(self) -> float:
        return _mean(r.iterations for r in self.results)

    @property
    def mean_nodes(self) -> float:
        return _mean(r.nodes for r in self.results)

    @property
    def total_wall_seconds(self) -> float:
        return sum(r.wall_seconds for r in self.results)

    @property
    def total_tokens(self) -> int | None:
        counted = [r.tokens for r in self.results if r.tokens is not None]
        return sum(counted) if counted else None

    @property
    def outcome_counts(self) -> dict[str, int]:
        return dict(Counter(r.outcome for r in self.results))

    def solved_mean_iterations(self) -> float:
        """Mean iterations across solved runs only.

        Unsolved runs terminate at whatever budget stopped them, so folding
        them into the mean measures the budget rather than the search.
        """
        return _mean(r.iterations for r in self.results if r.solved)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "runs": self.runs,
            "solved": self.solved,
            "solve_rate": round(self.solve_rate, 6),
            "mean_iterations": round(self.mean_iterations, 4),
            "mean_nodes": round(self.mean_nodes, 4),
            "total_wall_seconds": round(self.total_wall_seconds, 6),
            "total_tokens": self.total_tokens,
            "outcome_counts": self.outcome_counts,
            "results": [r.to_dict() for r in self.results],
        }

    def format_report(self) -> str:
        """Renders a per-task table followed by the aggregate summary."""
        lines = [f"benchmark: {self.label}", ""]
        lines.append(f"  {'task':<28} {'seed':>5} {'outcome':<17} {'iters':>6} {'nodes':>6}")
        for result in self.results:
            mark = "+" if result.solved else "-"
            seed = "-" if result.seed is None else str(result.seed)
            lines.append(
                f"{mark} {result.task:<28} {seed:>5} {result.outcome:<17} "
                f"{result.iterations:>6} {result.nodes:>6}"
            )
        lines.extend(
            [
                "",
                f"  solve rate       : {self.solved}/{self.runs} = {self.solve_rate:.0%}",
                f"  mean iterations  : {self.mean_iterations:.1f} "
                f"(solved only: {self.solved_mean_iterations():.1f})",
                f"  mean nodes       : {self.mean_nodes:.1f}",
                f"  wall time        : {self.total_wall_seconds * 1000:.0f} ms",
            ]
        )
        if self.total_tokens is not None:
            lines.append(f"  tokens           : {self.total_tokens}")
        outcomes = ", ".join(f"{k}={v}" for k, v in sorted(self.outcome_counts.items()))
        lines.append(f"  outcomes         : {outcomes}")
        return "\n".join(lines)


def run_benchmark(
    config: SearchConfig,
    tasks: Sequence[BenchmarkTask] | None = None,
    label: str = "run",
    repeats: int = 1,
    archive_dir: str | Path | None = None,
) -> BenchmarkReport:
    """Runs every task under ``config`` and aggregates the outcomes.

    ``repeats`` re-runs each task with the seed offset by the repeat index, so
    a configuration whose advantage is a lucky tie-break shows up as variance
    rather than as a win. A base seed of ``None`` leaves every repeat
    genuinely random.

    ``archive_dir`` writes each run to a JSON archive, so a surprising result
    can be reopened and inspected long after the benchmark finished.
    """
    if repeats < 1:
        raise ValueError("repeats must be a positive integer")

    suite = tuple(tasks) if tasks is not None else default_suite()
    destination = Path(archive_dir) if archive_dir is not None else None
    results: list[TaskResult] = []

    for task in suite:
        base = task.resolve_config(config)
        for repeat in range(repeats):
            effective = _seeded(base, repeat)
            setup = task.build(effective)

            # perf_counter, not monotonic: individual runs can finish well
            # inside the ~15 ms resolution of the coarser clock, which would
            # round most of the suite to zero.
            started = time.perf_counter()
            result = setup.controller.run(task.statement)
            elapsed = time.perf_counter() - started

            tokens = setup.usage().total_tokens if setup.usage is not None else None
            solved = task.is_solved_by(
                result.outcome is SearchOutcome.SUCCEEDED, result.solution
            )
            results.append(
                TaskResult(
                    task=task.name,
                    seed=effective.seed,
                    outcome=result.outcome.value,
                    solved=solved,
                    iterations=result.iterations,
                    nodes=result.node_count,
                    wall_seconds=elapsed,
                    tokens=tokens,
                )
            )

            if destination is not None:
                from cognitivetree.persistence import save_run

                metrics = RunMetrics.from_result(result)
                save_run(
                    result,
                    destination / f"{label}-{task.name}-seed{effective.seed}.json",
                    metrics=metrics.to_dict(),
                )

    return BenchmarkReport(label=label, results=tuple(results))


def scaling_curve(
    budgets: Sequence[int],
    config: SearchConfig | None = None,
    tasks: Sequence[BenchmarkTask] | None = None,
    repeats: int = 1,
) -> tuple[BenchmarkReport, ...]:
    """Measures solve rate across a range of iteration budgets.

    This is the test-time compute question in its most direct form: how much
    does the answer improve when the search is allowed to think longer.
    """
    base = config or SearchConfig()
    return tuple(
        run_benchmark(
            replace(base, max_iterations=budget),
            tasks=tasks,
            label=f"budget={budget}",
            repeats=repeats,
        )
        for budget in budgets
    )


def compare_reports(baseline: BenchmarkReport, candidate: BenchmarkReport) -> str:
    """Renders a head-to-head comparison of two benchmark reports."""
    rows = [
        ("solve rate", f"{baseline.solve_rate:.0%}", f"{candidate.solve_rate:.0%}",
         _delta(candidate.solve_rate - baseline.solve_rate, ".0%", higher_is_better=True)),
        ("mean iterations", f"{baseline.mean_iterations:.1f}",
         f"{candidate.mean_iterations:.1f}",
         _delta(candidate.mean_iterations - baseline.mean_iterations, ".1f",
                higher_is_better=False)),
        ("mean nodes", f"{baseline.mean_nodes:.1f}", f"{candidate.mean_nodes:.1f}",
         _delta(candidate.mean_nodes - baseline.mean_nodes, ".1f",
                higher_is_better=False)),
        ("wall time (ms)", f"{baseline.total_wall_seconds * 1000:.0f}",
         f"{candidate.total_wall_seconds * 1000:.0f}",
         _delta((candidate.total_wall_seconds - baseline.total_wall_seconds) * 1000,
                ".0f", higher_is_better=False)),
    ]
    if baseline.total_tokens is not None and candidate.total_tokens is not None:
        rows.append(
            ("tokens", str(baseline.total_tokens), str(candidate.total_tokens),
             _delta(candidate.total_tokens - baseline.total_tokens, ".0f",
                    higher_is_better=False))
        )

    width = max(len(baseline.label), len(candidate.label), 10)
    lines = [
        f"{'metric':<18} {baseline.label:>{width}} {candidate.label:>{width}}   delta",
        "-" * (20 + 2 * width + 12),
    ]
    lines.extend(
        f"{name:<18} {left:>{width}} {right:>{width}}   {delta}"
        for name, left, right, delta in rows
    )
    return "\n".join(lines)


def _seeded(config: SearchConfig, repeat: int) -> SearchConfig:
    """Offsets the base seed by the repeat index, preserving randomness."""
    if config.seed is None:
        return config
    return replace(config, seed=config.seed + repeat)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def _delta(value: float, spec: str, higher_is_better: bool) -> str:
    """Formats a signed delta, marking whether it is an improvement."""
    if abs(value) < 1e-9:
        return "same"
    improved = value > 0 if higher_is_better else value < 0
    return f"{value:+{spec}} {'better' if improved else 'worse'}"

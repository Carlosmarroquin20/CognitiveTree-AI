"""Structural and temporal metrics derived from a completed run."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from cognitivetree.feedback.revision import REVISION_ATTEMPTS_KEY
from cognitivetree.node import NodeStatus
from cognitivetree.search import SearchOutcome, SearchResult
from cognitivetree.state import SearchPhase


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Aggregate token accounting for the LLM calls of a single run."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __sub__(self, earlier: TokenUsage) -> TokenUsage:
        """Returns the consumption between ``earlier`` and this tally."""
        return TokenUsage(
            calls=self.calls - earlier.calls,
            prompt_tokens=self.prompt_tokens - earlier.prompt_tokens,
            completion_tokens=self.completion_tokens - earlier.completion_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class RunMetrics:
    """Quantitative summary of one reasoning run.

    Every field is reconstructed from the run's phase history and final tree,
    so metrics are a pure projection of the result — computing them never
    perturbs the search and can be repeated on an archived result.
    """

    outcome: str
    iterations: int
    nodes: int
    wall_time_seconds: float
    solution_depth: int | None
    best_path_length: int
    structural_backtracks: int
    revision_backtracks: int
    revisions_granted: int
    status_counts: dict[str, int]
    phase_counts: dict[str, int]
    phase_seconds: dict[str, float]
    token_usage: TokenUsage | None = None

    @classmethod
    def from_result(
        cls, result: SearchResult, token_usage: TokenUsage | None = None
    ) -> RunMetrics:
        """Projects ``result`` onto a metrics summary."""
        history = result.phase_history
        phase_counts = Counter(t.target.value for t in history)
        phase_seconds = _phase_durations(result)
        wall = (
            history[-1].timestamp - history[0].timestamp if len(history) >= 2 else 0.0
        )

        backtrack_notes = [
            t.note for t in history if t.target is SearchPhase.BACKTRACKING
        ]
        revision_backtracks = sum(1 for n in backtrack_notes if "revision" in n.lower())

        status_counts = {status.value: 0 for status in NodeStatus}
        revisions_granted = 0
        for node in result.tree.nodes():
            status_counts[node.status.value] += 1
            revisions_granted += int(node.metadata.get(REVISION_ATTEMPTS_KEY, 0))

        solved = result.outcome is SearchOutcome.SUCCEEDED and bool(result.best_path)
        return cls(
            outcome=result.outcome.value,
            iterations=result.iterations,
            nodes=result.node_count,
            wall_time_seconds=wall,
            solution_depth=result.best_path[-1].depth if solved else None,
            best_path_length=len(result.best_path),
            structural_backtracks=len(backtrack_notes) - revision_backtracks,
            revision_backtracks=revision_backtracks,
            revisions_granted=revisions_granted,
            status_counts=status_counts,
            phase_counts=dict(phase_counts),
            phase_seconds=phase_seconds,
            token_usage=token_usage,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the metrics for transport and archival."""
        return {
            "outcome": self.outcome,
            "iterations": self.iterations,
            "nodes": self.nodes,
            "wall_time_seconds": round(self.wall_time_seconds, 6),
            "solution_depth": self.solution_depth,
            "best_path_length": self.best_path_length,
            "structural_backtracks": self.structural_backtracks,
            "revision_backtracks": self.revision_backtracks,
            "revisions_granted": self.revisions_granted,
            "status_counts": dict(self.status_counts),
            "phase_counts": dict(self.phase_counts),
            "phase_seconds": {k: round(v, 6) for k, v in self.phase_seconds.items()},
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
        }

    def format_report(self) -> str:
        """Renders a compact human-readable report for terminal inspection."""
        lines = [
            "run metrics",
            f"  outcome            : {self.outcome}",
            f"  iterations         : {self.iterations}",
            f"  nodes              : {self.nodes}",
            f"  wall time          : {self.wall_time_seconds * 1000:.1f} ms",
            f"  solution depth     : {self.solution_depth}",
            f"  best path length   : {self.best_path_length}",
            f"  backtracks         : {self.structural_backtracks} structural, "
            f"{self.revision_backtracks} revision",
            f"  revisions granted  : {self.revisions_granted}",
            "  node status        : "
            + ", ".join(f"{k}={v}" for k, v in self.status_counts.items() if v),
            "  phase time (ms)    : "
            + ", ".join(
                f"{k}={v * 1000:.1f}"
                for k, v in sorted(
                    self.phase_seconds.items(), key=lambda kv: kv[1], reverse=True
                )
                if v > 0
            ),
        ]
        if self.token_usage is not None:
            usage = self.token_usage
            lines.append(
                f"  llm tokens         : {usage.total_tokens} "
                f"({usage.prompt_tokens} prompt + {usage.completion_tokens} completion) "
                f"across {usage.calls} calls"
            )
        return "\n".join(lines)


def _phase_durations(result: SearchResult) -> dict[str, float]:
    """Attributes wall time to each phase from consecutive transition stamps.

    A transition's target phase occupies the interval until the next
    transition; the terminal phase, having no successor, contributes zero.
    """
    history = result.phase_history
    durations: dict[str, float] = {}
    for index, transition in enumerate(history):
        following = index + 1
        end = history[following].timestamp if following < len(history) else transition.timestamp
        durations[transition.target.value] = durations.get(
            transition.target.value, 0.0
        ) + (end - transition.timestamp)
    return durations

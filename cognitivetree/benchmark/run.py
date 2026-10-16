"""Command-line entry point for the benchmark suite.

Examples:
    # Compute-scaling curve: solve rate as the iteration budget grows
    python -m cognitivetree.benchmark.run --budgets 10 40 120

    # One configuration in detail, averaged over three seeds
    python -m cognitivetree.benchmark.run --budgets 40 --repeats 3 --detail

    # Head-to-head: does a wider exploration weight help at a fixed budget?
    python -m cognitivetree.benchmark.run --budgets 40 --compare-exploration 3.0

    # LLM coding tasks with a scripted client: tokens, and what caching saves
    python -m cognitivetree.benchmark.run --suite llm-offline --budgets 10 \\
        --repeats 2 --cache-completions

    # The same tasks against a real model (key, if any, in COGNITIVETREE_API_KEY)
    python -m cognitivetree.benchmark.run --suite llm --budgets 4 10 \\
        --base-url http://localhost:11434/v1 --model llama3.3
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from cognitivetree.benchmark.llm_tasks import llm_suite, offline_llm_suite
from cognitivetree.benchmark.runner import (
    compare_reports,
    run_benchmark,
    scaling_curve,
)
from cognitivetree.benchmark.suite import BenchmarkTask, default_suite
from cognitivetree.config import SearchConfig
from cognitivetree.llm.caching import CompletionCache
from cognitivetree.llm.openai_compatible import API_KEY_ENV_VAR, OpenAICompatibleClient

SUITES = ("synthetic", "llm-offline", "llm")


def build_parser() -> argparse.ArgumentParser:
    """Declares the CLI surface."""
    parser = argparse.ArgumentParser(
        prog="cognitivetree-benchmark",
        description="Runs the CognitiveTree-AI benchmark suite.",
    )
    parser.add_argument(
        "--suite",
        choices=SUITES,
        default="synthetic",
        help=(
            "synthetic: hidden-sequence search, no model; llm-offline: coding "
            "tasks through the LLM stack with a scripted client; llm: the same "
            "tasks against a live OpenAI-compatible endpoint (default: synthetic)"
        ),
    )
    parser.add_argument("--base-url", help="endpoint root for --suite llm")
    parser.add_argument("--model", help="served model identifier for --suite llm")
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="generator temperature for the LLM suites (default: 0 offline, 0.7 live)",
    )
    parser.add_argument(
        "--cache-completions",
        action="store_true",
        help=(
            "share one completion cache across every run of the LLM suites; "
            "only temperature-0 requests are cached"
        ),
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[10, 40, 120],
        help="iteration budgets to measure (default: 10 40 120)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="runs per task, with the seed offset each time (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=7, help="base seed (default: 7)")
    parser.add_argument(
        "--exploration",
        type=float,
        default=SearchConfig().exploration_weight,
        help="UCT exploration weight for the baseline",
    )
    parser.add_argument(
        "--compare-exploration",
        type=float,
        default=None,
        help="second exploration weight to compare against the baseline",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent candidate evaluations per batch (default: 1)",
    )
    parser.add_argument(
        "--detail", action="store_true", help="print the per-task table"
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="write every run to this directory as a JSON archive",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write the reports to a JSON file"
    )
    return parser


def select_tasks(
    args: argparse.Namespace,
) -> tuple[Sequence[BenchmarkTask], CompletionCache | None]:
    """Builds the task suite the CLI selected, with its completion cache."""
    llm_only = args.temperature is not None or args.cache_completions
    if args.suite == "synthetic":
        if llm_only:
            # The synthetic suite calls no model, so these would silently do
            # nothing; refusing them beats pretending they apply.
            raise SystemExit("--temperature/--cache-completions need an LLM suite")
        return default_suite(), None
    if args.temperature is not None and not 0.0 <= args.temperature <= 2.0:
        raise SystemExit("--temperature must lie within [0.0, 2.0]")

    cache = CompletionCache() if args.cache_completions else None
    if args.suite == "llm-offline":
        temperature = 0.0 if args.temperature is None else args.temperature
        return offline_llm_suite(cache=cache, temperature=temperature), cache

    missing = [
        flag
        for flag, value in (("--base-url", args.base_url), ("--model", args.model))
        if not value
    ]
    if missing:
        raise SystemExit(f"--suite llm requires {', '.join(missing)}")
    base_url, model = args.base_url, args.model
    api_key = os.environ.get(API_KEY_ENV_VAR) or None

    def client() -> OpenAICompatibleClient:
        return OpenAICompatibleClient(base_url=base_url, model=model, api_key=api_key)

    temperature = 0.7 if args.temperature is None else args.temperature
    return llm_suite(client, temperature=temperature, cache=cache), cache


def main(argv: list[str] | None = None) -> None:
    """Runs the requested benchmarks and prints their reports."""
    args = build_parser().parse_args(argv)
    if args.repeats < 1:
        raise SystemExit("--repeats must be a positive integer")
    if args.workers < 1:
        raise SystemExit("--workers must be a positive integer")
    if any(budget < 1 for budget in args.budgets):
        raise SystemExit("--budgets must all be positive integers")

    base = SearchConfig(
        seed=args.seed,
        exploration_weight=args.exploration,
        evaluation_workers=args.workers,
    )

    tasks, cache = select_tasks(args)
    reports = scaling_curve(args.budgets, config=base, tasks=tasks, repeats=args.repeats)
    if args.detail:
        for report in reports:
            print(report.format_report())
            print()

    with_tokens = any(report.total_tokens is not None for report in reports)
    print(f"compute scaling curve ({args.suite} suite)")
    header = f"  {'budget':>8} {'solve rate':>11} {'mean iters':>11} {'wall (ms)':>10}"
    print(header + (f" {'tokens':>8}" if with_tokens else ""))
    for report in reports:
        budget = report.label.split("=", 1)[1]
        row = (
            f"  {budget:>8} {report.solve_rate:>10.0%} "
            f"{report.mean_iterations:>11.1f} "
            f"{report.total_wall_seconds * 1000:>10.1f}"
        )
        print(row + (f" {report.total_tokens or 0:>8}" if with_tokens else ""))
    if cache is not None:
        stats = cache.stats
        print(
            f"  completion cache: {stats.hits} hits, {stats.misses} misses "
            f"({stats.hit_rate:.0%} hit rate)"
        )

    comparisons = []
    if args.compare_exploration is not None:
        budget = args.budgets[-1]
        baseline = run_benchmark(
            replace(base, max_iterations=budget),
            tasks=tasks,
            label=f"expl={args.exploration:g}",
            repeats=args.repeats,
        )
        candidate = run_benchmark(
            replace(
                base,
                max_iterations=budget,
                exploration_weight=args.compare_exploration,
            ),
            tasks=tasks,
            label=f"expl={args.compare_exploration:g}",
            repeats=args.repeats,
        )
        print()
        print(f"head-to-head at budget {budget}")
        print(compare_reports(baseline, candidate))
        comparisons = [baseline, candidate]

    if args.archive_dir is not None:
        for budget in args.budgets:
            run_benchmark(
                replace(base, max_iterations=budget),
                tasks=tasks,
                label=f"budget={budget}",
                repeats=args.repeats,
                archive_dir=args.archive_dir,
            )
        print()
        print(f"archives written to {args.archive_dir}")

    if args.json is not None:
        payload = [r.to_dict() for r in (*reports, *comparisons)]
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"reports written to {args.json}")


if __name__ == "__main__":
    main()

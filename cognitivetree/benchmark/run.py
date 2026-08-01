"""Command-line entry point for the benchmark suite.

Examples:
    # Compute-scaling curve: solve rate as the iteration budget grows
    python -m cognitivetree.benchmark.run --budgets 10 40 120

    # One configuration in detail, averaged over three seeds
    python -m cognitivetree.benchmark.run --budgets 40 --repeats 3 --detail

    # Head-to-head: does a wider exploration weight help at a fixed budget?
    python -m cognitivetree.benchmark.run --budgets 40 --compare-exploration 3.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cognitivetree.benchmark.runner import (
    compare_reports,
    run_benchmark,
    scaling_curve,
)
from cognitivetree.config import SearchConfig


def build_parser() -> argparse.ArgumentParser:
    """Declares the CLI surface."""
    parser = argparse.ArgumentParser(
        prog="cognitivetree-benchmark",
        description="Runs the CognitiveTree-AI benchmark suite.",
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

    reports = scaling_curve(args.budgets, config=base, repeats=args.repeats)
    if args.detail:
        for report in reports:
            print(report.format_report())
            print()

    print("compute scaling curve")
    print(f"  {'budget':>8} {'solve rate':>11} {'mean iters':>11} {'wall (ms)':>10}")
    for report in reports:
        budget = report.label.split("=", 1)[1]
        print(
            f"  {budget:>8} {report.solve_rate:>10.0%} "
            f"{report.mean_iterations:>11.1f} "
            f"{report.total_wall_seconds * 1000:>10.1f}"
        )

    comparisons = []
    if args.compare_exploration is not None:
        budget = args.budgets[-1]
        baseline = run_benchmark(
            replace(base, max_iterations=budget),
            label=f"expl={args.exploration:g}",
            repeats=args.repeats,
        )
        candidate = run_benchmark(
            replace(
                base,
                max_iterations=budget,
                exploration_weight=args.compare_exploration,
            ),
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

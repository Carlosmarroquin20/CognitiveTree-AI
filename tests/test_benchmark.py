"""Validates the benchmark suite, runner, aggregation, and comparison."""

import json
from pathlib import Path

import pytest

from cognitivetree.benchmark import (
    BenchmarkReport,
    BenchmarkTask,
    TaskResult,
    TaskSetup,
    TerminalOnlyEvaluator,
    compare_reports,
    default_suite,
    run_benchmark,
    scaling_curve,
)
from cognitivetree.benchmark.run import build_parser, main
from cognitivetree.benchmark.suite import VOCABULARY, sequence_task
from cognitivetree.config import SearchConfig
from cognitivetree.node import ThoughtNode
from cognitivetree.observability.metrics import TokenUsage
from cognitivetree.policies import Evaluation
from cognitivetree.search import TreeSearchController

BASE = SearchConfig(seed=7)


class TestTerminalOnlyEvaluator:
    """The evaluator that makes the suite a real search problem."""

    def test_partial_candidates_are_indistinguishable(self) -> None:
        evaluator = TerminalOnlyEvaluator(target=("north", "east", "south"))
        right = evaluator.evaluate(ThoughtNode(content="north"))
        wrong = evaluator.evaluate(ThoughtNode(content="west"))
        assert right.score == wrong.score == 0.5
        assert not right.is_terminal and not wrong.is_terminal

    def test_exact_match_is_an_accepted_terminal(self) -> None:
        evaluator = TerminalOnlyEvaluator(target=("north", "east"))
        verdict = evaluator.evaluate(ThoughtNode(content="north east"))
        assert verdict.score == 1.0
        assert verdict.is_terminal

    def test_full_length_mismatch_is_a_rejected_terminal(self) -> None:
        evaluator = TerminalOnlyEvaluator(target=("north", "east"))
        verdict = evaluator.evaluate(ThoughtNode(content="south west"))
        assert verdict.score == 0.0
        assert verdict.is_terminal


class TestSuite:
    """Shape and intrinsic configuration of the bundled tasks."""

    def test_default_suite_is_graded_by_difficulty(self) -> None:
        suite = default_suite()
        depths = [t.config_overrides["max_depth"] for t in suite]
        assert len(suite) >= 4
        assert min(depths) == 2
        assert max(depths) >= 4

    def test_task_overrides_depth_and_branching(self) -> None:
        task = sequence_task(("north", "east", "south"))
        resolved = task.resolve_config(SearchConfig(max_depth=99, branching_factor=1))
        assert resolved.max_depth == 3
        assert resolved.branching_factor == len(VOCABULARY)

    def test_overrides_leave_other_fields_untouched(self) -> None:
        task = sequence_task(("north", "east"))
        resolved = task.resolve_config(SearchConfig(max_iterations=55, seed=3))
        assert resolved.max_iterations == 55
        assert resolved.seed == 3

    def test_task_names_are_unique(self) -> None:
        names = [t.name for t in default_suite()]
        assert len(names) == len(set(names))

    def test_expected_solution_gates_the_solve(self) -> None:
        task = sequence_task(("north", "east"))
        assert task.is_solved_by(True, "north east")
        assert not task.is_solved_by(True, "north west")
        assert not task.is_solved_by(False, "north east")

    def test_missing_expectation_trusts_the_outcome(self) -> None:
        task = BenchmarkTask(name="t", statement="s", build=lambda c: None)
        assert task.is_solved_by(True, "anything")
        assert not task.is_solved_by(False, None)


class TestRunner:
    """Execution and aggregation behavior."""

    def test_report_covers_every_task(self) -> None:
        report = run_benchmark(SearchConfig(max_iterations=20, seed=7))
        assert report.runs == len(default_suite())
        assert {r.task for r in report.results} == {t.name for t in default_suite()}

    def test_repeats_offset_the_seed(self) -> None:
        report = run_benchmark(
            SearchConfig(max_iterations=10, seed=7), repeats=3, tasks=default_suite()[:2]
        )
        assert report.runs == 6
        assert sorted({r.seed for r in report.results}) == [7, 8, 9]

    def test_repeats_leave_an_unseeded_config_random(self) -> None:
        report = run_benchmark(
            SearchConfig(max_iterations=10, seed=None), repeats=2, tasks=default_suite()[:1]
        )
        assert {r.seed for r in report.results} == {None}

    def test_generous_budget_solves_the_whole_suite(self) -> None:
        report = run_benchmark(SearchConfig(max_iterations=400, seed=7))
        assert report.solve_rate == 1.0
        assert report.outcome_counts == {"succeeded": report.runs}

    def test_tight_budget_leaves_hard_tasks_unsolved(self) -> None:
        report = run_benchmark(SearchConfig(max_iterations=5, seed=7))
        assert 0.0 < report.solve_rate < 1.0
        assert "exhausted" in report.outcome_counts

    def test_solved_mean_excludes_budget_capped_runs(self) -> None:
        report = run_benchmark(SearchConfig(max_iterations=40, seed=7))
        # Unsolved runs burn the whole budget, so folding them in would
        # measure the budget rather than the search.
        assert report.solved_mean_iterations() < report.mean_iterations

    def test_wall_time_is_measured_with_usable_resolution(self) -> None:
        report = run_benchmark(SearchConfig(max_iterations=40, seed=7))
        assert report.total_wall_seconds > 0.0

    def test_rejects_non_positive_repeats(self) -> None:
        with pytest.raises(ValueError, match="repeats"):
            run_benchmark(BASE, repeats=0)

    def test_archives_every_run_when_asked(self, tmp_path: Path) -> None:
        from cognitivetree.persistence import load_run

        run_benchmark(
            SearchConfig(max_iterations=10, seed=7),
            tasks=default_suite()[:2],
            label="archived",
            archive_dir=tmp_path,
        )
        written = sorted(tmp_path.glob("*.json"))
        assert len(written) == 2
        assert load_run(written[0]).result.iterations > 0


class TestTokenReporting:
    """Consumption is reported when a task exposes it."""

    def test_usage_probe_feeds_the_report(self) -> None:
        class OneShotGenerator:
            def generate(self, node: ThoughtNode, k: int) -> list[str]:
                return ["done"] if node.is_root else []

        class AcceptingEvaluator:
            def evaluate(self, node: ThoughtNode) -> Evaluation:
                return Evaluation(score=1.0, is_terminal=True)

        def build(config: SearchConfig) -> TaskSetup:
            return TaskSetup(
                controller=TreeSearchController(
                    config=config,
                    generator=OneShotGenerator(),
                    evaluator=AcceptingEvaluator(),
                ),
                usage=lambda: TokenUsage(calls=2, prompt_tokens=30, completion_tokens=12),
            )

        task = BenchmarkTask(name="token-task", statement="s", build=build)
        report = run_benchmark(BASE, tasks=[task])

        assert report.results[0].tokens == 42
        assert report.total_tokens == 42
        assert "tokens" in report.format_report()

    def test_absent_usage_leaves_tokens_unset(self) -> None:
        report = run_benchmark(SearchConfig(max_iterations=10, seed=7))
        assert all(r.tokens is None for r in report.results)
        assert report.total_tokens is None


class TestScalingCurve:
    """The headline test-time compute measurement."""

    def test_solve_rate_is_monotonic_in_budget(self) -> None:
        reports = scaling_curve([5, 40, 400], config=BASE)
        rates = [r.solve_rate for r in reports]
        assert rates == sorted(rates)
        assert rates[0] < rates[-1]
        assert rates[-1] == 1.0

    def test_labels_identify_the_budget(self) -> None:
        reports = scaling_curve([10, 20], config=BASE)
        assert [r.label for r in reports] == ["budget=10", "budget=20"]


class TestReportRendering:
    """Serialization and human-readable output."""

    def report(self) -> BenchmarkReport:
        return BenchmarkReport(
            label="demo",
            results=(
                TaskResult("a", 7, "succeeded", True, 4, 17, 0.01),
                TaskResult("b", 7, "exhausted", False, 40, 161, 0.02),
            ),
        )

    def test_aggregates(self) -> None:
        report = self.report()
        assert report.runs == 2
        assert report.solved == 1
        assert report.solve_rate == 0.5
        assert report.mean_iterations == 22.0
        assert report.solved_mean_iterations() == 4.0
        assert report.outcome_counts == {"succeeded": 1, "exhausted": 1}

    def test_empty_report_does_not_divide_by_zero(self) -> None:
        empty = BenchmarkReport(label="none", results=())
        assert empty.solve_rate == 0.0
        assert empty.mean_iterations == 0.0
        assert empty.solved_mean_iterations() == 0.0

    def test_to_dict_is_json_serializable(self) -> None:
        payload = self.report().to_dict()
        assert json.loads(json.dumps(payload))["solve_rate"] == 0.5
        assert len(payload["results"]) == 2

    def test_format_report_marks_solved_and_unsolved(self) -> None:
        text = self.report().format_report()
        assert "+ a" in text
        assert "- b" in text
        assert "1/2 = 50%" in text


class TestComparison:
    """Head-to-head rendering between two reports."""

    def build(self, label: str, solved: bool, iterations: int) -> BenchmarkReport:
        return BenchmarkReport(
            label=label,
            results=(TaskResult("a", 7, "succeeded", solved, iterations, 10, 0.01),),
        )

    def test_improvement_is_labelled_better(self) -> None:
        text = compare_reports(
            self.build("base", False, 40), self.build("cand", True, 20)
        )
        assert "better" in text
        assert "base" in text and "cand" in text

    def test_regression_is_labelled_worse(self) -> None:
        text = compare_reports(
            self.build("base", True, 20), self.build("cand", False, 40)
        )
        assert "worse" in text

    def test_identical_reports_read_as_same(self) -> None:
        text = compare_reports(
            self.build("base", True, 20), self.build("cand", True, 20)
        )
        assert text.count("same") >= 3


class TestCli:
    """Argument handling of the benchmark entry point."""

    def test_defaults(self) -> None:
        args = build_parser().parse_args([])
        assert args.budgets == [10, 40, 120]
        assert args.repeats == 1
        assert args.workers == 1

    @pytest.mark.parametrize(
        "argv",
        [["--repeats", "0"], ["--workers", "0"], ["--budgets", "0"]],
        ids=["repeats", "workers", "budgets"],
    )
    def test_non_positive_arguments_are_rejected(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit, match="positive"):
            main(argv)

    def test_run_prints_the_curve(self, capsys) -> None:
        main(["--budgets", "5", "400"])
        out = capsys.readouterr().out
        assert "compute scaling curve" in out
        assert "100%" in out

    def test_comparison_mode(self, capsys) -> None:
        main(["--budgets", "40", "--compare-exploration", "3.0"])
        out = capsys.readouterr().out
        assert "head-to-head" in out
        assert "expl=1.414" in out and "expl=3" in out

    def test_detail_mode_prints_the_per_task_table(self, capsys) -> None:
        main(["--budgets", "10", "--detail"])
        out = capsys.readouterr().out
        assert "benchmark: budget=10" in out
        assert "solve rate" in out

    def test_archive_dir_writes_one_file_per_run(
        self, tmp_path: Path, capsys
    ) -> None:
        main(["--budgets", "10", "--archive-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert "archives written to" in out
        assert len(list(tmp_path.glob("*.json"))) == len(default_suite())

    def test_json_export(self, tmp_path: Path, capsys) -> None:
        destination = tmp_path / "nested" / "reports.json"
        main(["--budgets", "10", "--json", str(destination)])
        capsys.readouterr()
        payload = json.loads(destination.read_text(encoding="utf-8"))
        assert payload[0]["label"] == "budget=10"
        assert "results" in payload[0]

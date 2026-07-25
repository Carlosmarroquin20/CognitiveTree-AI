"""Validates run-metrics projection from completed search results."""

import pytest

from cognitivetree.config import SearchConfig
from cognitivetree.demo import SequencePuzzleEvaluator, SequencePuzzleGenerator
from cognitivetree.llm.demo import TASK, build_offline_controller, clamp_responder
from cognitivetree.llm.scripted import ScriptedLlmClient
from cognitivetree.observability import AccountingLlmClient, RunMetrics, TokenUsage
from cognitivetree.search import TreeSearchController


def offline_result():
    client = AccountingLlmClient(ScriptedLlmClient(clamp_responder))
    result = build_offline_controller(client=client).run(TASK)
    return result, client


def test_metrics_capture_revision_backtracking_run() -> None:
    result, _ = offline_result()
    metrics = RunMetrics.from_result(result)

    assert metrics.outcome == "succeeded"
    assert metrics.iterations == 2
    assert metrics.solution_depth == 1
    assert metrics.best_path_length == 2
    assert metrics.revision_backtracks == 1
    assert metrics.structural_backtracks == 0
    assert metrics.revisions_granted == 1
    assert metrics.status_counts["pruned"] == 3
    assert metrics.status_counts["terminal"] == 1


def test_phase_time_sums_to_wall_time() -> None:
    result, _ = offline_result()
    metrics = RunMetrics.from_result(result)
    assert metrics.phase_seconds
    assert sum(metrics.phase_seconds.values()) == pytest.approx(
        metrics.wall_time_seconds, abs=1e-9
    )
    assert metrics.wall_time_seconds >= 0.0


def test_token_usage_is_carried_and_serialized() -> None:
    result, client = offline_result()
    metrics = RunMetrics.from_result(result, token_usage=client.usage)

    assert metrics.token_usage is not None
    assert metrics.token_usage.calls >= 2
    payload = metrics.to_dict()
    assert payload["token_usage"]["total_tokens"] == metrics.token_usage.total_tokens
    assert "llm tokens" in metrics.format_report()


def test_to_dict_is_json_shaped() -> None:
    result, _ = offline_result()
    payload = RunMetrics.from_result(result).to_dict()
    assert payload["outcome"] == "succeeded"
    assert isinstance(payload["phase_counts"], dict)
    assert isinstance(payload["status_counts"], dict)
    assert payload["token_usage"] is None


def test_exhausted_run_reports_no_solution_depth() -> None:
    target = ("north", "east", "south")
    vocabulary = ("north", "south", "east", "west")
    controller = TreeSearchController(
        config=SearchConfig(
            max_iterations=64, max_depth=2, branching_factor=len(vocabulary), seed=7
        ),
        generator=SequencePuzzleGenerator(vocabulary=vocabulary),
        evaluator=SequencePuzzleEvaluator(target=target),
    )
    metrics = RunMetrics.from_result(controller.run("recover the sequence"))

    assert metrics.outcome == "exhausted"
    assert metrics.solution_depth is None
    assert metrics.status_counts["terminal"] == 0
    assert metrics.token_usage is None


def test_report_lists_core_fields() -> None:
    result, _ = offline_result()
    report = RunMetrics.from_result(result).format_report()
    assert "outcome" in report
    assert "revisions granted" in report
    assert "phase time" in report


def test_empty_token_usage_still_reports() -> None:
    result, _ = offline_result()
    metrics = RunMetrics.from_result(result, token_usage=TokenUsage())
    assert metrics.token_usage is not None
    assert "0" in metrics.format_report()

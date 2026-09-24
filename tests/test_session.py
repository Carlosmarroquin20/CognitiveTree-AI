"""Validates session lifecycle and the streamed envelope contract."""

import pytest
from conftest import EndlessRun

from cognitivetree.search import SearchOutcome
from cognitivetree.session import ReasoningSession, build_reference_session
from cognitivetree.state import SearchPhase


def test_reference_session_runs_to_success() -> None:
    result = build_reference_session().run()
    assert result.outcome is SearchOutcome.SUCCEEDED
    assert result.solution is not None


def test_stream_envelope_ordering_and_contract() -> None:
    envelopes = list(build_reference_session().stream())

    assert envelopes[0]["type"] == "phase"
    assert envelopes[0]["phase"] == "selection"
    assert envelopes[-1]["type"] == "result"

    kinds = {envelope["type"] for envelope in envelopes}
    assert kinds == {"phase", "snapshot", "metrics", "result"}

    result = envelopes[-1]
    assert result["outcome"] == "succeeded"
    assert result["solution"]
    assert result["iterations"] == 2
    assert result["best_path"]

    snapshots = [e for e in envelopes if e["type"] == "snapshot"]
    assert snapshots, "at least one snapshot expected"
    final_tree = snapshots[-1]["tree"]
    assert final_tree["size"] == result["node_count"]
    assert final_tree["root"]["children"]

    # The metrics envelope precedes the closing result and summarizes the run.
    metrics = [e for e in envelopes if e["type"] == "metrics"]
    assert len(metrics) == 1
    assert envelopes[-2]["type"] == "metrics"
    summary = metrics[0]["metrics"]
    assert summary["outcome"] == "succeeded"
    assert summary["revision_backtracks"] == 1
    assert summary["nodes"] == result["node_count"]

    phases = [e["phase"] for e in envelopes if e["type"] == "phase"]
    assert "backtracking" in phases
    assert phases[-1] == "succeeded"


def test_stream_and_run_are_independent_executions() -> None:
    session = build_reference_session()
    first = session.run()
    envelopes = list(session.stream())
    assert first.outcome is SearchOutcome.SUCCEEDED
    assert envelopes[-1]["outcome"] == "succeeded"


def test_blank_task_is_rejected() -> None:
    with pytest.raises(ValueError):
        ReasoningSession(task="  ", controller_factory=lambda sink: None)


def test_abandoned_stream_cancels_its_run(endless_run: EndlessRun) -> None:
    run = endless_run
    stream = ReasoningSession("endless", run.factory).stream()
    for _ in range(5):
        next(stream)
    stream.close()

    assert run.settled.wait(timeout=10)
    assert run.terminal is SearchPhase.CANCELLED
    assert run.generated < 1_000


def test_fully_consumed_stream_is_never_cancelled() -> None:
    envelopes = list(build_reference_session().stream())
    assert envelopes[-1]["outcome"] == "succeeded"
    assert "cancelled" not in [e.get("phase") for e in envelopes]

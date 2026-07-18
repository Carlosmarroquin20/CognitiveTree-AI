"""Validates session lifecycle and the streamed envelope contract."""

import pytest

from cognitivetree.search import SearchOutcome
from cognitivetree.session import ReasoningSession, build_reference_session


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
    assert kinds == {"phase", "snapshot", "result"}

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

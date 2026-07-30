"""Validates that replaying an archive reproduces the live envelope stream."""

from pathlib import Path

import pytest

from cognitivetree.llm.demo import TASK, build_offline_controller, build_offline_session
from cognitivetree.observability import RunMetrics
from cognitivetree.persistence import ReplaySession, load_run, save_run
from cognitivetree.state import SearchPhase


@pytest.fixture
def archived_run(tmp_path: Path):
    result = build_offline_controller().run(TASK)
    metrics = RunMetrics.from_result(result)
    path = save_run(result, tmp_path / "run.json", metrics=metrics.to_dict())
    return load_run(path)


def test_replay_exposes_the_session_surface(archived_run) -> None:
    session = ReplaySession(archived_run)
    assert session.task == TASK
    assert session.archive is archived_run


def test_replay_envelope_sequence_matches_a_live_run(archived_run) -> None:
    live = [e["type"] for e in build_offline_session().stream()]
    replayed = [e["type"] for e in ReplaySession(archived_run).stream()]
    assert replayed == live


def test_replay_phase_log_matches_the_archived_history(archived_run) -> None:
    envelopes = list(ReplaySession(archived_run).stream())
    phases = [e["phase"] for e in envelopes if e["type"] == "phase"]
    assert phases == [t.target.value for t in archived_run.result.phase_history]

    details = [e["detail"] for e in envelopes if e["type"] == "phase"]
    assert details == [t.note for t in archived_run.result.phase_history]


def test_replay_closes_with_the_archived_outcome(archived_run) -> None:
    envelopes = list(ReplaySession(archived_run).stream())
    closing = envelopes[-1]
    assert closing["type"] == "result"
    assert closing["outcome"] == archived_run.result.outcome.value
    assert closing["solution"] == archived_run.result.solution
    assert closing["best_path"] == [n.id for n in archived_run.result.best_path]


def test_replay_emits_the_stored_metrics(archived_run) -> None:
    envelopes = list(ReplaySession(archived_run).stream())
    metrics = [e for e in envelopes if e["type"] == "metrics"]
    assert len(metrics) == 1
    assert metrics[0]["metrics"] == archived_run.metrics


def test_replay_recomputes_metrics_when_absent(tmp_path: Path) -> None:
    result = build_offline_controller().run(TASK)
    archive = load_run(save_run(result, tmp_path / "bare.json"))
    assert archive.metrics is None

    envelopes = list(ReplaySession(archive).stream())
    summary = next(e for e in envelopes if e["type"] == "metrics")["metrics"]
    assert summary["outcome"] == "succeeded"
    assert summary["revision_backtracks"] == 1


def test_replay_snapshots_carry_the_full_tree(archived_run) -> None:
    envelopes = list(ReplaySession(archived_run).stream())
    snapshots = [e for e in envelopes if e["type"] == "snapshot"]
    assert snapshots
    for snapshot in snapshots:
        assert snapshot["tree"]["size"] == archived_run.result.node_count


def test_replay_iteration_counter_tracks_expansions(archived_run) -> None:
    envelopes = [e for e in ReplaySession(archived_run).stream() if e["type"] == "phase"]
    expansions = [e for e in envelopes if e["phase"] == SearchPhase.EXPANSION.value]
    assert [e["iteration"] for e in expansions] == list(range(1, len(expansions) + 1))
    assert envelopes[-1]["iteration"] == archived_run.result.iterations


def test_replay_is_repeatable(archived_run) -> None:
    session = ReplaySession(archived_run)
    first = [(e["type"], e.get("phase")) for e in session.stream()]
    second = [(e["type"], e.get("phase")) for e in session.stream()]
    assert first == second


def test_replay_speed_must_be_positive(archived_run) -> None:
    with pytest.raises(ValueError, match="speed must be positive"):
        ReplaySession(archived_run, speed=0)


def test_paced_replay_takes_measurable_time(archived_run) -> None:
    import time

    history = archived_run.result.phase_history
    span = history[-1].timestamp - history[0].timestamp
    speed = max(span / 0.05, 1.0)  # target roughly 50 ms of total pacing

    started = time.monotonic()
    list(ReplaySession(archived_run, speed=speed).stream())
    elapsed = time.monotonic() - started

    assert elapsed > 0.0
    assert elapsed < span + 5.0


def test_from_path_loads_and_wraps(tmp_path: Path) -> None:
    result = build_offline_controller().run(TASK)
    path = save_run(result, tmp_path / "run.json")
    session = ReplaySession.from_path(path)
    assert session.task == TASK
    assert list(session.stream())[-1]["outcome"] == "succeeded"

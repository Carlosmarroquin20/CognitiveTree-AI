"""Validates session lifecycle and the streamed envelope contract."""

import logging
import time
from pathlib import Path

import pytest
from conftest import EndlessRun

from cognitivetree.config import SearchConfig
from cognitivetree.persistence import ReplaySession, load_run
from cognitivetree.policies import Evaluation
from cognitivetree.search import SearchOutcome, TreeSearchController
from cognitivetree.session import EventSink, ReasoningSession, build_reference_session
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


class TestRunArchiving:
    """Sessions with an archive directory record every finished run."""

    def test_run_is_archived_and_reloadable(self, tmp_path: Path) -> None:
        result = build_reference_session(archive_dir=tmp_path).run()

        (path,) = tmp_path.glob("*.json")
        assert "-succeeded-" in path.name
        archive = load_run(path)
        assert archive.result.outcome is result.outcome
        assert archive.result.solution == result.solution
        assert archive.metrics is not None and archive.metrics["outcome"] == "succeeded"

    def test_streamed_run_is_archived_and_replayable(self, tmp_path: Path) -> None:
        live = list(build_reference_session(archive_dir=tmp_path).stream())

        (path,) = tmp_path.glob("*.json")
        replayed = list(ReplaySession.from_path(path).stream())
        assert replayed[-1]["outcome"] == live[-1]["outcome"] == "succeeded"

    def test_each_run_gets_its_own_archive(self, tmp_path: Path) -> None:
        session = build_reference_session(archive_dir=tmp_path)
        session.run()
        session.run()
        assert len(list(tmp_path.glob("*.json"))) == 2

    def test_cancelled_run_is_archived(
        self, tmp_path: Path, endless_run: EndlessRun
    ) -> None:
        stream = ReasoningSession("endless", endless_run.factory, archive_dir=tmp_path).stream()
        next(stream)
        stream.close()

        # The worker archives after the terminal transition, so allow it to land.
        deadline = time.monotonic() + 10
        while not list(tmp_path.glob("*.json")) and time.monotonic() < deadline:
            time.sleep(0.05)
        (path,) = tmp_path.glob("*.json")
        assert load_run(path).result.outcome is SearchOutcome.CANCELLED

    def test_unwritable_directory_is_logged_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("occupied", encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            result = build_reference_session(archive_dir=blocker).run()

        assert result.outcome is SearchOutcome.SUCCEEDED
        assert "run archive could not be written" in caplog.text

    def test_archiving_is_off_by_default(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        build_reference_session().run()
        assert list(tmp_path.rglob("*.json")) == []


def _finished_events(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if getattr(r, "event", None) == "run_finished"]


class BrokenBackend:
    """Fails every expansion, as an unreachable model backend would."""

    def generate(self, node: object, k: int) -> list[str]:
        raise RuntimeError("backend unreachable")

    def evaluate(self, node: object) -> Evaluation:
        return Evaluation(score=0.5)


class TestRunFinishedEvent:
    """Every finished run is reported as one structured log record."""

    def test_successful_run_logs_its_summary(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger="cognitivetree.session"):
            result = build_reference_session().run()

        (record,) = _finished_events(caplog)
        assert record.levelno == logging.INFO
        assert record.getMessage() == f"run succeeded after {result.iterations} iterations"
        assert record.outcome == "succeeded"
        assert record.nodes == result.node_count
        assert record.tokens is None and record.archive is None and record.error is None

    def test_archived_run_logs_its_archive_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="cognitivetree.session"):
            build_reference_session(archive_dir=tmp_path).run()

        (record,) = _finished_events(caplog)
        (path,) = tmp_path.glob("*.json")
        assert record.archive == str(path)

    def test_failed_run_logs_a_warning_with_the_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def factory(sink: EventSink | None) -> TreeSearchController:
            backend = BrokenBackend()
            return TreeSearchController(
                config=SearchConfig(seed=1),
                generator=backend,
                evaluator=backend,
                on_event=sink,
            )

        with caplog.at_level(logging.INFO, logger="cognitivetree.session"):
            ReasoningSession("broken", factory).run()

        (record,) = _finished_events(caplog)
        assert record.levelno == logging.WARNING
        assert record.outcome == "failed"
        assert "backend unreachable" in record.error

    def test_streamed_llm_run_logs_its_tokens(self, caplog: pytest.LogCaptureFixture) -> None:
        from cognitivetree.llm.demo import build_offline_session

        with caplog.at_level(logging.INFO, logger="cognitivetree.session"):
            list(build_offline_session().stream())

        (record,) = _finished_events(caplog)
        assert record.tokens > 0
        assert record.llm_calls == 2

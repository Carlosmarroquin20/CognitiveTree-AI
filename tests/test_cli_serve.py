"""Validates the streaming interface's CLI parsing and session wiring.

Sessions are inspected without running them: calling a session's controller
factory constructs the wired ``TreeSearchController`` (and, for the reference
and llm-demo backends, its in-process executor/client) without executing a
search, which is fast and side-effect-free, and lets these tests assert on
the exact ``SearchConfig`` that reached the controller.
"""

from pathlib import Path

import pytest

from cognitivetree.ui.serve import build_parser, session_factory_from_args


def parse(argv: list[str]):
    return build_parser().parse_args(argv)


def test_defaults_select_the_reference_backend_unbounded() -> None:
    args = parse([])
    assert args.backend == "reference"
    assert args.max_seconds is None
    assert args.host == "127.0.0.1"
    assert args.port == 8732


def test_backend_choices_are_closed() -> None:
    with pytest.raises(SystemExit):
        parse(["--backend", "not-a-real-backend"])


def test_llm_backend_requires_base_url_model_and_task() -> None:
    args = parse(["--backend", "llm"])
    with pytest.raises(SystemExit, match="base-url"):
        session_factory_from_args(args)


def test_llm_backend_builds_once_required_arguments_are_present() -> None:
    args = parse(
        [
            "--backend", "llm",
            "--base-url", "http://localhost:11434/v1",
            "--model", "llama3.3",
            "--task", "do the thing",
        ]
    )
    session = session_factory_from_args(args)()
    assert session.task == "do the thing"


def test_max_seconds_reaches_the_llm_session_config() -> None:
    args = parse(
        [
            "--backend", "llm",
            "--base-url", "http://localhost:11434/v1",
            "--model", "llama3.3",
            "--task", "do the thing",
            "--max-seconds", "9.0",
        ]
    )
    session = session_factory_from_args(args)()
    controller = session._factory(None)
    assert controller._config.max_wall_seconds == 9.0


@pytest.mark.parametrize(
    "backend_args",
    [["--backend", "reference"], ["--backend", "llm-demo"]],
    ids=["reference", "llm-demo"],
)
class TestModelFreeBackends:
    """Exercises the two backends that require no external endpoint."""

    def test_max_seconds_defaults_to_unbounded(self, backend_args: list[str]) -> None:
        session = session_factory_from_args(parse(backend_args))()
        controller = session._factory(None)
        assert controller._config.max_wall_seconds is None

    def test_max_seconds_threads_into_the_controller_config(
        self, backend_args: list[str]
    ) -> None:
        args = parse([*backend_args, "--max-seconds", "3.5"])
        session = session_factory_from_args(args)()
        controller = session._factory(None)
        assert controller._config.max_wall_seconds == 3.5

    def test_non_positive_max_seconds_is_rejected_up_front(
        self, backend_args: list[str]
    ) -> None:
        args = parse([*backend_args, "--max-seconds", "0"])
        with pytest.raises(SystemExit, match="max-seconds"):
            session_factory_from_args(args)


class TestReplayBackend:
    """Wiring of the archive-replay backend."""

    def archive_path(self, tmp_path: Path) -> Path:
        from cognitivetree.llm.demo import TASK, build_offline_controller
        from cognitivetree.persistence import save_run

        return save_run(build_offline_controller().run(TASK), tmp_path / "run.json")

    def test_replay_requires_an_archive(self) -> None:
        with pytest.raises(SystemExit, match="requires --archive"):
            session_factory_from_args(parse(["--backend", "replay"]))

    def test_missing_archive_fails_at_startup(self, tmp_path: Path) -> None:
        args = parse(
            ["--backend", "replay", "--archive", str(tmp_path / "absent.json")]
        )
        with pytest.raises(SystemExit, match="cannot read archive"):
            session_factory_from_args(args)

    def test_corrupt_archive_fails_at_startup(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text('{"format": "not-ours"}', encoding="utf-8")
        args = parse(["--backend", "replay", "--archive", str(broken)])
        with pytest.raises(SystemExit, match="cannot read archive"):
            session_factory_from_args(args)

    def test_replay_session_streams_the_archived_run(self, tmp_path: Path) -> None:
        args = parse(
            ["--backend", "replay", "--archive", str(self.archive_path(tmp_path))]
        )
        session = session_factory_from_args(args)()
        envelopes = list(session.stream())
        assert envelopes[-1]["type"] == "result"
        assert envelopes[-1]["outcome"] == "succeeded"

    def test_replay_speed_reaches_the_session(self, tmp_path: Path) -> None:
        args = parse(
            [
                "--backend", "replay",
                "--archive", str(self.archive_path(tmp_path)),
                "--replay-speed", "4.0",
            ]
        )
        session = session_factory_from_args(args)()
        assert session._speed == 4.0

    def test_non_positive_replay_speed_is_rejected(self, tmp_path: Path) -> None:
        args = parse(
            [
                "--backend", "replay",
                "--archive", str(self.archive_path(tmp_path)),
                "--replay-speed", "0",
            ]
        )
        with pytest.raises(SystemExit, match="replay-speed"):
            session_factory_from_args(args)

    def test_each_connection_gets_an_independent_session(self, tmp_path: Path) -> None:
        args = parse(
            ["--backend", "replay", "--archive", str(self.archive_path(tmp_path))]
        )
        factory = session_factory_from_args(args)
        assert factory() is not factory()

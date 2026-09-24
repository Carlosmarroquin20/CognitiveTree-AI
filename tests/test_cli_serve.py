"""Validates the streaming interface's CLI parsing and session wiring.

Sessions are inspected without running them: calling a session's controller
factory constructs the wired ``TreeSearchController`` (and, for the reference
and llm-demo backends, its in-process executor/client) without executing a
search, which is fast and side-effect-free, and lets these tests assert on
the exact ``SearchConfig`` that reached the controller.
"""

from pathlib import Path

import pytest

from cognitivetree.session import LlmSessionSpec
from cognitivetree.ui import serve
from cognitivetree.ui.serve import (
    API_KEY_ENV_VAR,
    build_parser,
    is_loopback_host,
    session_factory_from_args,
)


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


class TestTokenBudgetFlags:
    """Wiring and validation of the consumption-ceiling flags."""

    def test_defaults_are_unbounded(self) -> None:
        args = parse([])
        assert args.max_tokens is None
        assert args.max_llm_calls is None

    def test_ceilings_reach_the_llm_session_spec(self) -> None:
        args = parse(
            [
                "--backend", "llm",
                "--base-url", "http://localhost:11434/v1",
                "--model", "llama3.3",
                "--task", "do the thing",
                "--max-tokens", "5000",
                "--max-llm-calls", "12",
            ]
        )
        session = session_factory_from_args(args)()
        controller = session._factory(None)
        assert controller._stop_condition is not None
        assert controller._stop_condition._max_total_tokens == 5000
        assert controller._stop_condition._max_calls == 12

    def test_ceiling_reaches_the_offline_backend(self) -> None:
        args = parse(["--backend", "llm-demo", "--max-tokens", "50"])
        session = session_factory_from_args(args)()
        controller = session._factory(None)
        assert controller._stop_condition is not None
        assert controller._stop_condition._max_total_tokens == 50

    def test_no_stop_condition_without_a_ceiling(self) -> None:
        args = parse(["--backend", "llm-demo"])
        session = session_factory_from_args(args)()
        assert session._factory(None)._stop_condition is None

    @pytest.mark.parametrize("flag", ["--max-tokens", "--max-llm-calls"])
    def test_non_positive_ceilings_are_rejected(self, flag: str) -> None:
        args = parse(["--backend", "llm-demo", flag, "0"])
        with pytest.raises(SystemExit, match="positive integer"):
            session_factory_from_args(args)

    def test_ceilings_are_refused_on_the_model_free_reference_backend(self) -> None:
        args = parse(["--backend", "reference", "--max-tokens", "100"])
        with pytest.raises(SystemExit, match="need an LLM backend"):
            session_factory_from_args(args)


class TestEvaluationWorkersFlag:
    """Wiring and validation of the concurrency flag."""

    def test_default_is_sequential(self) -> None:
        assert parse([]).eval_workers == 1

    @pytest.mark.parametrize(
        "backend_args",
        [["--backend", "reference"], ["--backend", "llm-demo"]],
        ids=["reference", "llm-demo"],
    )
    def test_workers_reach_the_controller_config(self, backend_args: list[str]) -> None:
        args = parse([*backend_args, "--eval-workers", "4"])
        controller = session_factory_from_args(args)()._factory(None)
        assert controller._config.evaluation_workers == 4

    def test_workers_reach_the_llm_session_config(self) -> None:
        args = parse(
            [
                "--backend", "llm",
                "--base-url", "http://localhost:11434/v1",
                "--model", "llama3.3",
                "--task", "do the thing",
                "--eval-workers", "6",
            ]
        )
        controller = session_factory_from_args(args)()._factory(None)
        assert controller._config.evaluation_workers == 6

    def test_non_positive_worker_count_is_rejected(self) -> None:
        with pytest.raises(SystemExit, match="eval-workers"):
            session_factory_from_args(parse(["--eval-workers", "0"]))


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


class TestApiKey:
    """The bearer token prefers the environment over the command line."""

    ARGS = [
        "--backend", "llm",
        "--base-url", "http://localhost:11434/v1",
        "--model", "llama3.3",
        "--task", "do the thing",
    ]

    def captured_spec(
        self, monkeypatch: pytest.MonkeyPatch, argv: list[str]
    ) -> LlmSessionSpec:
        specs: list[LlmSessionSpec] = []
        monkeypatch.setattr(serve, "build_llm_session", specs.append)
        session_factory_from_args(parse(argv))()
        return specs[0]

    def test_key_is_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(API_KEY_ENV_VAR, "env-token")
        assert self.captured_spec(monkeypatch, self.ARGS).api_key == "env-token"

    def test_no_key_when_neither_source_is_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(API_KEY_ENV_VAR, raising=False)
        assert self.captured_spec(monkeypatch, self.ARGS).api_key is None

    def test_explicit_flag_still_works_but_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(API_KEY_ENV_VAR, "env-token")
        spec = self.captured_spec(monkeypatch, [*self.ARGS, "--api-key", "flag-token"])
        assert spec.api_key == "flag-token"
        assert API_KEY_ENV_VAR in caplog.text
        assert "flag-token" not in caplog.text


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("127.8.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.20", False),
        ("my-workstation", False),
    ],
)
def test_loopback_detection(host: str, loopback: bool) -> None:
    assert is_loopback_host(host) is loopback


def test_concurrent_runs_are_capped_by_default() -> None:
    assert parse([]).max_concurrent_runs == 4


def test_non_positive_concurrency_cap_is_rejected() -> None:
    with pytest.raises(SystemExit, match="max-concurrent-runs"):
        session_factory_from_args(parse(["--max-concurrent-runs", "0"]))


class TestTemperatures:
    """Sampling temperatures travel from the CLI into the session spec."""

    def test_defaults_match_the_tuned_policy_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = TestApiKey().captured_spec(monkeypatch, TestApiKey.ARGS)
        assert (spec.temperature, spec.critic_temperature) == (0.7, 0.2)

    def test_flags_reach_the_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        argv = [*TestApiKey.ARGS, "--temperature", "0", "--critic-temperature", "0"]
        spec = TestApiKey().captured_spec(monkeypatch, argv)
        assert (spec.temperature, spec.critic_temperature) == (0.0, 0.0)

    @pytest.mark.parametrize("flag", ["--temperature", "--critic-temperature"])
    @pytest.mark.parametrize("value", ["-0.1", "2.5"])
    def test_out_of_range_values_are_rejected(self, flag: str, value: str) -> None:
        with pytest.raises(SystemExit, match=flag):
            session_factory_from_args(parse([*TestApiKey.ARGS, flag, value]))

"""Smoke-tests every documented entry point.

The demos are the project's front door: they are what a reader runs first and
what the README promises works without a model or a Docker daemon. Nothing
else imports their ``main`` functions, so a refactor elsewhere can break them
silently. Each test drives the real entry point and asserts on the output it
advertises.
"""

from __future__ import annotations

import pytest


class TestCoreDemos:
    """Entry points that exercise the search stack."""

    def test_reference_search(self, capsys: pytest.CaptureFixture[str]) -> None:
        from cognitivetree.demo import main

        main()
        out = capsys.readouterr().out
        assert "outcome     : succeeded" in out
        assert "north east east south west" in out

    def test_sandboxed_search(self, capsys: pytest.CaptureFixture[str]) -> None:
        from cognitivetree.sandbox.demo import main

        main()
        out = capsys.readouterr().out
        assert "execution backend:" in out
        assert "outcome  : succeeded" in out
        assert "max(low, min(value, high))" in out

    def test_critique_driven_backtracking(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cognitivetree.feedback.demo import main

        main()
        out = capsys.readouterr().out
        assert "revision attempts : 1" in out
        assert "outcome           : succeeded" in out
        assert "backtracking" in out


class TestLayerDemos:
    """Entry points for the LLM, observability, and persistence layers."""

    def test_offline_llm_stack(self, capsys: pytest.CaptureFixture[str]) -> None:
        from cognitivetree.llm.demo import main

        main()
        out = capsys.readouterr().out
        assert "outcome           : succeeded" in out
        assert "revision notes fed back into the generator prompt" in out

    def test_metrics_and_token_budget(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cognitivetree.observability.demo import main

        main()
        out = capsys.readouterr().out
        assert "run metrics" in out
        assert "llm tokens" in out
        # The second half must actually demonstrate the ceiling firing.
        assert "outcome          : budget_exhausted" in out

    def test_archive_and_reopen(self, capsys: pytest.CaptureFixture[str]) -> None:
        from cognitivetree.persistence.demo import main

        main()
        out = capsys.readouterr().out
        assert "live run outcome : timed_out" in out
        assert "--- reopened from disk ---" in out
        assert "wall-clock budget" in out
        assert "token accounting preserved from save time" in out


class TestCliEntryPoints:
    """Console-script entry points, invoked as a user would."""

    def test_benchmark_cli(self, capsys: pytest.CaptureFixture[str]) -> None:
        from cognitivetree.benchmark.run import main

        main(["--budgets", "5", "400"])
        out = capsys.readouterr().out
        assert "compute scaling curve" in out
        assert "100%" in out

    def test_streaming_server_cli_builds_without_serving(self) -> None:
        # main() blocks in serve_forever, so the reachable assertion is that
        # the parser and factory wiring the CLI depends on stay intact.
        from cognitivetree.ui.serve import build_parser, session_factory_from_args

        args = build_parser().parse_args(["--backend", "llm-demo"])
        session = session_factory_from_args(args)()
        assert session.task
        assert list(session.stream())[-1]["type"] == "result"

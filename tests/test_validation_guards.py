"""Covers the guards and error paths that only fire on bad input.

These branches exist so that misconfiguration fails loudly at construction
rather than producing a subtly wrong run, and so that infrastructure faults
are distinguishable from payload faults. Untested, they are only assertions
about intent; exercised, they are guarantees.
"""

from __future__ import annotations

import pytest

from cognitivetree.llm.client import ChatMessage, CompletionRequest
from cognitivetree.policies import Critique, FailureClass
from cognitivetree.sandbox.spec import (
    ExecutionRequest,
    ExecutionStatus,
    ResourceLimits,
    SandboxError,
    clip_output,
    decode_captured,
)
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor

REPLACEMENT = chr(0xFFFD)  # what undecodable bytes become


class TestResourceLimits:
    """Every ceiling rejects values that would disable the protection."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"memory_mb": 3}, "memory_mb"),
            ({"cpus": 0}, "cpus"),
            ({"cpus": -1.0}, "cpus"),
            ({"pids": 0}, "pids"),
            ({"timeout_seconds": 0}, "timeout_seconds"),
            ({"timeout_seconds": -5.0}, "timeout_seconds"),
            ({"output_limit_chars": 0}, "output_limit_chars"),
        ],
    )
    def test_invalid_limits_are_rejected(self, kwargs: dict, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            ResourceLimits(**kwargs)

    def test_minimum_viable_limits_are_accepted(self) -> None:
        limits = ResourceLimits(
            memory_mb=4, cpus=0.1, pids=1, timeout_seconds=0.1, output_limit_chars=1
        )
        assert limits.memory_mb == 4


class TestExecutionRequest:
    """Payload submissions validate before reaching an executor."""

    @pytest.mark.parametrize("code", ["", "   ", "\n\t "])
    def test_blank_payloads_are_rejected(self, code: str) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ExecutionRequest(code=code)

    @pytest.mark.parametrize("timeout", [0, -1.0])
    def test_non_positive_timeout_override_is_rejected(self, timeout: float) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            ExecutionRequest(code="pass", timeout_seconds=timeout)

    def test_absent_override_is_allowed(self) -> None:
        assert ExecutionRequest(code="pass").timeout_seconds is None


class TestCompletionRequest:
    """Completion parameters are validated before a call is attempted."""

    @pytest.mark.parametrize("temperature", [-0.1, 2.1])
    def test_temperature_must_be_in_range(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature"):
            CompletionRequest(
                messages=(ChatMessage(role="user", content="hi"),),
                temperature=temperature,
            )

    @pytest.mark.parametrize("max_tokens", [0, -1])
    def test_max_tokens_must_be_positive(self, max_tokens: int) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            CompletionRequest(
                messages=(ChatMessage(role="user", content="hi"),),
                max_tokens=max_tokens,
            )

    def test_range_boundaries_are_inclusive(self) -> None:
        for temperature in (0.0, 2.0):
            CompletionRequest(
                messages=(ChatMessage(role="user", content="hi"),),
                temperature=temperature,
            )


class TestCritiqueSeverity:
    """Severity feeds reward shaping, so it must stay normalized."""

    @pytest.mark.parametrize("severity", [-0.01, 1.01])
    def test_out_of_range_severity_is_rejected(self, severity: float) -> None:
        with pytest.raises(ValueError, match="severity"):
            Critique(
                failure_class=FailureClass.ASSERTION,
                summary="s",
                guidance="g",
                severity=severity,
            )

    def test_boundaries_are_accepted(self) -> None:
        for severity in (0.0, 1.0):
            Critique(
                failure_class=FailureClass.EXCEPTION,
                summary="s",
                guidance="g",
                severity=severity,
            )


class TestSubprocessExecutorErrorPaths:
    """Infrastructure faults stay distinguishable from payload faults."""

    def test_unspawnable_interpreter_raises_sandbox_error(self) -> None:
        executor = SubprocessExecutor(
            python_executable="definitely-not-an-interpreter-xyz"
        )
        with pytest.raises(SandboxError, match="failed to spawn interpreter"):
            executor.execute(ExecutionRequest(code="print(1)"))

    def test_partial_output_on_timeout_is_decoded(self) -> None:
        # A payload that prints before hanging exercises the timeout branch's
        # handling of whatever the child had already emitted.
        executor = SubprocessExecutor(limits=ResourceLimits(timeout_seconds=1.0))
        result = executor.execute(
            ExecutionRequest(code="print('partial', flush=True)\nwhile True: pass")
        )
        assert result.status is ExecutionStatus.TIMEOUT
        assert isinstance(result.stdout, str)

    @pytest.mark.parametrize(
        ("data", "truncated", "expected"),
        [
            (b"bytes", False, ("bytes", False)),
            (b"\xff\xfe", False, (REPLACEMENT * 2, False)),
            (b"abcdef", False, ("abcde", True)),
            (b"abc", True, ("abc", True)),
            # A multi-byte character cut by the capture cap decodes to a
            # replacement character instead of raising.
            ("é".encode()[:1], True, (REPLACEMENT, True)),
        ],
    )
    def test_captured_output_decoding_covers_every_shape(
        self, data: bytes, truncated: bool, expected: tuple[str, bool]
    ) -> None:
        assert decode_captured(data, truncated, limit_chars=5) == expected

    def test_timeout_carries_no_exit_code(self) -> None:
        # A child killed at the deadline is the only path that yields None.
        executor = SubprocessExecutor(limits=ResourceLimits(timeout_seconds=1.0))
        result = executor.execute(ExecutionRequest(code="while True: pass"))
        assert result.exit_code is None


class TestOutputClipping:
    """The cap that keeps runaway output from re-entering the framework."""

    def test_short_output_passes_through_unflagged(self) -> None:
        assert clip_output("short", 100) == ("short", False)

    def test_exact_limit_is_not_clipped(self) -> None:
        assert clip_output("abcde", 5) == ("abcde", False)

    def test_oversized_output_is_clipped_and_flagged(self) -> None:
        text, clipped = clip_output("abcdef", 5)
        assert text == "abcde"
        assert clipped


class TestServerUrl:
    """The URL property must not leak a byte repr into a link."""

    def test_byte_encoded_host_is_decoded(self) -> None:
        from cognitivetree.session import build_reference_session
        from cognitivetree.ui.server import StreamingUiServer

        server = StreamingUiServer(("127.0.0.1", 0), build_reference_session)
        try:
            port = server.server_address[1]
            # socketserver types the address broadly; simulate the byte-encoded
            # family the property guards against.
            server.server_address = (b"127.0.0.1", port)  # type: ignore[assignment]
            assert server.url == f"http://127.0.0.1:{port}/"
            assert "b'" not in server.url
        finally:
            server.server_close()

    def test_string_host_is_unchanged(self) -> None:
        from cognitivetree.session import build_reference_session
        from cognitivetree.ui.server import StreamingUiServer

        server = StreamingUiServer(("127.0.0.1", 0), build_reference_session)
        try:
            assert server.url.startswith("http://127.0.0.1:")
            assert server.url.endswith("/")
        finally:
            server.server_close()

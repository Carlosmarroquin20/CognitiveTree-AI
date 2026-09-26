"""Validates the host-process executor's containment behavior."""

import tracemalloc

import pytest

from cognitivetree.sandbox.spec import (
    ExecutionRequest,
    ExecutionStatus,
    ResourceLimits,
)
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor


def test_clean_run_captures_stdout() -> None:
    result = SubprocessExecutor().execute(ExecutionRequest(code="print('payload ok')"))
    assert result.ok
    assert result.status is ExecutionStatus.COMPLETED
    assert result.exit_code == 0
    assert result.stdout.strip() == "payload ok"
    assert result.duration_seconds > 0


def test_failing_run_reports_exit_code_and_stderr() -> None:
    result = SubprocessExecutor().execute(
        ExecutionRequest(code="raise ValueError('broken payload')")
    )
    assert not result.ok
    assert result.status is ExecutionStatus.COMPLETED
    assert result.exit_code == 1
    assert "broken payload" in result.stderr


def test_explicit_exit_code_is_preserved() -> None:
    result = SubprocessExecutor().execute(
        ExecutionRequest(code="import sys; sys.exit(7)")
    )
    assert result.exit_code == 7


def test_stdin_reaches_the_payload() -> None:
    result = SubprocessExecutor().execute(
        ExecutionRequest(code="print(input().upper())", stdin="quiet\n")
    )
    assert result.stdout.strip() == "QUIET"


def test_non_latin_output_survives_any_host_locale() -> None:
    # Under a cp1252 locale the child used to crash on this print, turning a
    # correct payload into a graded failure.
    result = SubprocessExecutor().execute(
        ExecutionRequest(code="print('café ✓ λ')")
    )
    assert result.ok, result.stderr
    assert result.stdout.strip() == "café ✓ λ"


def test_non_latin_stdin_reaches_the_payload() -> None:
    result = SubprocessExecutor().execute(
        ExecutionRequest(code="print(input()[::-1])", stdin="α✓\n")
    )
    assert result.ok, result.stderr
    assert result.stdout.strip() == "✓α"


def test_timeout_kills_the_payload() -> None:
    executor = SubprocessExecutor(limits=ResourceLimits(timeout_seconds=1.0))
    result = executor.execute(ExecutionRequest(code="while True: pass"))
    assert result.status is ExecutionStatus.TIMEOUT
    assert result.exit_code is None
    assert "deadline" in result.detail


def test_request_timeout_overrides_configured_limit() -> None:
    executor = SubprocessExecutor(limits=ResourceLimits(timeout_seconds=30.0))
    result = executor.execute(
        ExecutionRequest(code="while True: pass", timeout_seconds=1.0)
    )
    assert result.status is ExecutionStatus.TIMEOUT


def test_oversized_output_is_clipped_and_flagged() -> None:
    executor = SubprocessExecutor(limits=ResourceLimits(output_limit_chars=100))
    result = executor.execute(ExecutionRequest(code="print('x' * 10_000)"))
    assert result.truncated
    assert len(result.stdout) == 100


def test_blank_payload_is_rejected_at_request_construction() -> None:
    with pytest.raises(ValueError):
        ExecutionRequest(code="   ")


def test_output_flood_is_clipped_without_buffering_it() -> None:
    executor = SubprocessExecutor(limits=ResourceLimits(output_limit_chars=1000))
    code = "import sys\nfor _ in range(400): sys.stdout.write('x' * 250_000)"
    tracemalloc.start()
    try:
        result = executor.execute(ExecutionRequest(code=code))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert result.ok
    assert result.truncated
    assert result.stdout == "x" * 1000
    # 100 MB emitted; buffering it whole used to peak near 200 MB.
    assert peak < 5_000_000


def test_line_endings_are_normalized() -> None:
    result = SubprocessExecutor().execute(ExecutionRequest(code="print('a')\nprint('b')"))
    assert result.stdout == "a\nb\n"


def test_timeout_keeps_partial_output() -> None:
    executor = SubprocessExecutor(limits=ResourceLimits(timeout_seconds=1.0))
    code = "import time\nprint('started', flush=True)\ntime.sleep(60)"
    result = executor.execute(ExecutionRequest(code=code))
    assert result.status is ExecutionStatus.TIMEOUT
    assert result.stdout.strip() == "started"

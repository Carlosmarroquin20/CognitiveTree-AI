"""Validates bounded child-process execution."""

import sys
import time
import tracemalloc

import pytest

from cognitivetree.sandbox.process import run_bounded


def python(code: str) -> list[str]:
    return [sys.executable, "-I", "-c", code]


def test_small_output_is_captured_whole() -> None:
    run = run_bounded(python("print('hello'); import sys; sys.exit(3)"), b"", 10, 1024)
    assert run.returncode == 3
    assert run.stdout.strip() == b"hello"
    assert not run.stdout_truncated and not run.timed_out


def test_stdin_reaches_the_child() -> None:
    run = run_bounded(python("print(input().upper())"), b"quiet\n", 10, 1024)
    assert run.stdout.strip() == b"QUIET"


def test_oversized_stream_is_capped_and_flagged() -> None:
    code = "import sys; sys.stdout.write('x' * 1_000_000); sys.stderr.write('e' * 10)"
    run = run_bounded(python(code), b"", 10, 4096)
    assert run.returncode == 0
    assert run.stdout == b"x" * 4096
    assert run.stdout_truncated
    assert run.stderr == b"e" * 10 and not run.stderr_truncated


def test_host_memory_stays_bounded_by_the_cap() -> None:
    # 100 MB of output against a 64 KB cap: buffering it all, as
    # subprocess.run does, peaks near 200 MB on the host.
    code = "import sys\nfor _ in range(400): sys.stdout.write('x' * 250_000)"
    tracemalloc.start()
    try:
        run = run_bounded(python(code), b"", 30, 64_000)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert run.stdout_truncated
    assert peak < 5_000_000


def test_deadline_kills_the_child_and_keeps_partial_output() -> None:
    code = "import sys, time\nprint('started', flush=True)\ntime.sleep(60)"
    started = time.perf_counter()
    run = run_bounded(python(code), b"", 1.0, 1024)
    assert run.timed_out and run.returncode is None
    assert run.stdout.strip() == b"started"
    assert time.perf_counter() - started < 15


def test_child_ignoring_stdin_does_not_deadlock() -> None:
    run = run_bounded(python("print('done')"), b"x" * 5_000_000, 10, 1024)
    assert run.returncode == 0
    assert run.stdout.strip() == b"done"


def test_unspawnable_command_raises_os_error() -> None:
    with pytest.raises(OSError):
        run_bounded(["definitely-not-a-binary-xyz"], b"", 5, 1024)


def test_cap_must_be_positive() -> None:
    with pytest.raises(ValueError):
        run_bounded(python("pass"), b"", 5, 0)

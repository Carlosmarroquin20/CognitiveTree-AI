"""Validates payload delivery over stdin through the bootstrap."""

import subprocess
import sys

import pytest

from cognitivetree.sandbox.payload import BOOTSTRAP, frame_payload
from cognitivetree.sandbox.spec import ExecutionRequest
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor


def plain_stderr(code: str) -> str:
    """Returns the stderr that running ``code`` through ``-c`` produces."""
    return subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True
    ).stderr


def run(code: str, stdin: str = "") -> tuple[int | None, str, str]:
    result = SubprocessExecutor().execute(ExecutionRequest(code=code, stdin=stdin))
    return result.exit_code, result.stdout, result.stderr


def test_frame_carries_byte_length_code_then_stdin() -> None:
    assert frame_payload("print('λ')", "in\n") == b"11\nprint('\xce\xbb')in\n"


def test_bootstrap_fits_any_command_line() -> None:
    assert len(BOOTSTRAP) < 2_000


def test_payload_beyond_the_windows_command_line_limit_runs() -> None:
    code = "x = 1\n" + "# padding\n" * 4000 + "print(x)"
    assert len(code) > 32_767
    exit_code, stdout, _ = run(code)
    assert exit_code == 0
    assert stdout.strip() == "1"


def test_payload_stdin_follows_the_framed_code() -> None:
    exit_code, stdout, _ = run("print(input()[::-1])\nprint(input())", "abc\nsecond\n")
    assert exit_code == 0
    assert stdout.split() == ["cba", "second"]


@pytest.mark.parametrize(
    "code",
    [
        "def f():\n    raise ValueError('boom')\nf()",
        "def broken(:\n    pass",
        "assert 1 == 2, 'in-range value must pass through'",
    ],
    ids=["runtime-error", "syntax-error", "assertion"],
)
def test_failures_report_exactly_what_dash_c_reports(code: str) -> None:
    exit_code, _, stderr = run(code)
    assert exit_code == 1
    assert stderr == plain_stderr(code)


def test_exit_codes_are_preserved() -> None:
    assert run("import sys; sys.exit(7)")[0] == 7


def test_payload_runs_as_main_without_bootstrap_names() -> None:
    code = (
        "import pickle\n"
        "class Point:\n"
        "    pass\n"
        "pickle.dumps(Point())\n"
        "print(__name__, sorted(k for k in globals() if not k.startswith('__')))"
    )
    exit_code, stdout, stderr = run(code)
    assert exit_code == 0, stderr
    assert stdout.strip() == "__main__ ['Point', 'pickle']"

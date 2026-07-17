"""Host-interpreter executor providing fault containment without isolation.

The executor exists for development hosts and CI runners without a Docker
daemon. It contains payload faults (crashes, hangs, runaway output) at the
process boundary but offers no filesystem, network, or privilege isolation;
production deployments route through
:class:`~cognitivetree.sandbox.docker_executor.DockerSandboxExecutor`.
"""

from __future__ import annotations

import subprocess
import sys
from time import perf_counter

from cognitivetree.sandbox.spec import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ResourceLimits,
    SandboxError,
    clip_output,
)


class SubprocessExecutor:
    """Runs payloads with the host interpreter in an isolated-mode child process.

    ``python -I`` detaches the child from user site-packages and environment
    variables, which keeps runs reproducible but must not be mistaken for a
    security boundary.
    """

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        python_executable: str | None = None,
    ) -> None:
        self._limits = limits or ResourceLimits()
        self._python = python_executable or sys.executable

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Executes ``request`` in a child process under the configured deadline."""
        timeout = request.timeout_seconds or self._limits.timeout_seconds
        command = [self._python, "-I", "-c", request.code]
        started = perf_counter()
        try:
            completed = subprocess.run(
                command,
                input=request.stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                exit_code=None,
                stdout=_decode_stream(exc.stdout),
                stderr=_decode_stream(exc.stderr),
                duration_seconds=perf_counter() - started,
                detail=f"payload exceeded {timeout:.1f}s deadline",
            )
        except OSError as exc:
            raise SandboxError(f"failed to spawn interpreter: {exc}") from exc

        duration = perf_counter() - started
        stdout, out_clipped = clip_output(
            completed.stdout, self._limits.output_limit_chars
        )
        stderr, err_clipped = clip_output(
            completed.stderr, self._limits.output_limit_chars
        )
        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            truncated=out_clipped or err_clipped,
        )


def _decode_stream(stream: str | bytes | None) -> str:
    """Normalizes the partial output attached to a timeout exception."""
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return stream

"""Host-interpreter executor providing fault containment without isolation.

The executor exists for development hosts and CI runners without a Docker
daemon. It contains payload faults (crashes, hangs, runaway output) at the
process boundary but offers no filesystem, network, or privilege isolation;
production deployments route through
:class:`~cognitivetree.sandbox.docker_executor.DockerSandboxExecutor`.
"""

from __future__ import annotations

import sys
from time import perf_counter

from cognitivetree.sandbox.process import run_bounded
from cognitivetree.sandbox.spec import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ResourceLimits,
    SandboxError,
    capture_bytes_for,
    decode_captured,
)


class SubprocessExecutor:
    """Runs payloads with the host interpreter in an isolated-mode child process.

    ``python -I`` detaches the child from user site-packages and environment
    variables, which keeps runs reproducible but must not be mistaken for a
    security boundary. ``-X utf8`` pins the child's standard streams to UTF-8:
    on a host whose locale encoding is narrower (cp1252 on most Windows
    installs), a correct payload printing a non-Latin character would
    otherwise crash on output and be graded as a failure.
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
        command = [self._python, "-X", "utf8", "-I", "-c", request.code]
        limit = self._limits.output_limit_chars
        started = perf_counter()
        try:
            run = run_bounded(
                command,
                stdin=request.stdin.encode("utf-8"),
                timeout=timeout,
                capture_bytes=capture_bytes_for(limit),
            )
        except OSError as exc:
            raise SandboxError(f"failed to spawn interpreter: {exc}") from exc
        duration = perf_counter() - started

        stdout, out_dropped = decode_captured(run.stdout, run.stdout_truncated, limit)
        stderr, err_dropped = decode_captured(run.stderr, run.stderr_truncated, limit)
        # The host child writes platform line endings; normalizing keeps the
        # output identical across operating systems, as text mode used to.
        stdout = stdout.replace("\r\n", "\n")
        stderr = stderr.replace("\r\n", "\n")
        if run.timed_out:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                exit_code=None,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                truncated=out_dropped or err_dropped,
                detail=f"payload exceeded {timeout:.1f}s deadline",
            )
        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            exit_code=run.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            truncated=out_dropped or err_dropped,
        )

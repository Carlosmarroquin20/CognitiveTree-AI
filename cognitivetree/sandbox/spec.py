"""Execution contracts shared by every sandbox backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any


@unique
class ExecutionStatus(Enum):
    """Classifies how an execution attempt concluded.

    COMPLETED covers every run where the payload itself terminated, including
    nonzero exits and resource-limit kills; the exit code carries the detail.
    TIMEOUT marks payloads forcibly stopped at the deadline. SANDBOX_ERROR is
    reserved for infrastructure faults (daemon unreachable, image missing)
    where the payload never got a fair chance to run.
    """

    COMPLETED = "completed"
    TIMEOUT = "timeout"
    SANDBOX_ERROR = "sandbox_error"


class SandboxError(RuntimeError):
    """Raised when the execution infrastructure itself fails."""


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Hard caps applied to a single payload execution.

    Attributes:
        memory_mb: Memory ceiling; the swap ceiling is pinned to the same
            value so the payload cannot page its way past the cap.
        cpus: CPU quota expressed in whole or fractional cores.
        pids: Maximum concurrent processes, bounding fork bombs.
        timeout_seconds: Wall-clock deadline before the payload is killed.
        output_limit_chars: Per-stream cap on captured stdout / stderr;
            oversized output is clipped and flagged, never propagated whole.
    """

    memory_mb: int = 256
    cpus: float = 1.0
    pids: int = 64
    timeout_seconds: float = 10.0
    output_limit_chars: int = 64_000

    def __post_init__(self) -> None:
        if self.memory_mb < 4:
            raise ValueError("memory_mb must be at least 4")
        if self.cpus <= 0:
            raise ValueError("cpus must be positive")
        if self.pids < 1:
            raise ValueError("pids must be a positive integer")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.output_limit_chars < 1:
            raise ValueError("output_limit_chars must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Describes one payload submission to an executor.

    ``timeout_seconds`` overrides the executor's configured deadline when set;
    ``stdin`` is delivered to the payload's standard input verbatim.
    """

    code: str
    stdin: str = ""
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if not self.code.strip():
            raise ValueError("code must be a non-empty payload")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds override must be positive")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Immutable record of one execution attempt."""

    status: ExecutionStatus
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    truncated: bool = False
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Reports whether the payload ran to completion and exited cleanly."""
        return self.status is ExecutionStatus.COMPLETED and self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        """Serializes the result for storage in node metadata."""
        return {
            "status": self.status.value,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_seconds": round(self.duration_seconds, 4),
            "truncated": self.truncated,
            "detail": self.detail,
        }


def clip_output(text: str, limit: int) -> tuple[str, bool]:
    """Clips ``text`` to ``limit`` characters, reporting whether clipping occurred."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True

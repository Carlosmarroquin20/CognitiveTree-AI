"""Hardened Docker backend for untrusted payload execution.

Every run receives a fresh, disposable container with a defense-in-depth
profile: no network, read-only root filesystem, dropped capabilities, blocked
privilege escalation, an unprivileged user, and hard memory / CPU / process
caps. Payloads reach the interpreter as an exec-form argument — never through
a shell — and the container is force-removed on timeout.
"""

from __future__ import annotations

import contextlib
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from cognitivetree.sandbox.spec import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ResourceLimits,
    SandboxError,
    clip_output,
)

DEFAULT_IMAGE = "cognitivetree-sandbox:latest"

# Exit code emitted by the Docker CLI itself when the daemon or configuration
# fails before the payload starts; anything at or above it is an infra fault.
_DOCKER_CLI_ERROR = 125

_CONTAINER_NAME_PREFIX = "ctree-sbx"


class DockerUnavailableError(SandboxError):
    """Raised when the Docker CLI or daemon cannot service a request."""


@dataclass(frozen=True, slots=True)
class DockerSandboxConfig:
    """Static configuration for the Docker execution backend.

    Attributes:
        image: Sandbox image reference; built from the bundled Dockerfile.
        docker_binary: Docker CLI entry point, overridable for compatible
            runtimes such as Podman.
        limits: Resource caps applied to every container.
        tmpfs_mb: Size of the writable ``/tmp`` tmpfs, the only mutable path.
        kill_grace_seconds: Extra wall-clock allowance granted to the CLI for
            container startup and teardown beyond the payload deadline.
    """

    image: str = DEFAULT_IMAGE
    docker_binary: str = "docker"
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    tmpfs_mb: int = 16
    kill_grace_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.tmpfs_mb < 1:
            raise ValueError("tmpfs_mb must be a positive integer")
        if self.kill_grace_seconds <= 0:
            raise ValueError("kill_grace_seconds must be positive")


class DockerSandboxExecutor:
    """Executes payloads in single-use, network-isolated Docker containers."""

    def __init__(self, config: DockerSandboxConfig | None = None) -> None:
        self._config = config or DockerSandboxConfig()

    @staticmethod
    def is_available(docker_binary: str = "docker", timeout: float = 10.0) -> bool:
        """Probes whether the Docker daemon is reachable through ``docker_binary``."""
        try:
            probe = subprocess.run(
                [docker_binary, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return probe.returncode == 0 and bool(probe.stdout.strip())

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Runs ``request`` in a fresh container and returns its result record."""
        limits = self._config.limits
        timeout = request.timeout_seconds or limits.timeout_seconds
        name = f"{_CONTAINER_NAME_PREFIX}-{uuid.uuid4().hex[:12]}"
        command = self._build_command(name, request)
        started = perf_counter()
        try:
            completed = subprocess.run(
                command,
                input=request.stdin,
                capture_output=True,
                text=True,
                timeout=timeout + self._config.kill_grace_seconds,
            )
        except FileNotFoundError as exc:
            raise DockerUnavailableError(
                f"docker binary {self._config.docker_binary!r} not found"
            ) from exc
        except subprocess.TimeoutExpired:
            self._force_remove(name)
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                exit_code=None,
                duration_seconds=perf_counter() - started,
                detail=f"payload exceeded {timeout:.1f}s deadline; container removed",
            )

        duration = perf_counter() - started
        stdout, out_clipped = clip_output(completed.stdout, limits.output_limit_chars)
        stderr, err_clipped = clip_output(completed.stderr, limits.output_limit_chars)

        if completed.returncode >= _DOCKER_CLI_ERROR and _is_cli_fault(
            completed.returncode, stderr
        ):
            return ExecutionResult(
                status=ExecutionStatus.SANDBOX_ERROR,
                exit_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
                truncated=out_clipped or err_clipped,
                detail="docker CLI reported an infrastructure fault",
            )

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration,
            truncated=out_clipped or err_clipped,
        )

    def _build_command(self, name: str, request: ExecutionRequest) -> list[str]:
        """Assembles the hardened ``docker run`` invocation for one payload."""
        limits = self._config.limits
        return [
            self._config.docker_binary,
            "run",
            "--rm",
            "--interactive",
            "--name",
            name,
            "--network",
            "none",
            "--memory",
            f"{limits.memory_mb}m",
            "--memory-swap",
            f"{limits.memory_mb}m",
            "--cpus",
            str(limits.cpus),
            "--pids-limit",
            str(limits.pids),
            "--read-only",
            "--tmpfs",
            f"/tmp:rw,size={self._config.tmpfs_mb}m",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            "65534:65534",
            "--workdir",
            "/tmp",
            "--env",
            "PYTHONUNBUFFERED=1",
            self._config.image,
            "python",
            "-I",
            "-c",
            request.code,
        ]

    def _force_remove(self, name: str) -> None:
        """Removes a timed-out container, tolerating already-gone containers."""
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                [self._config.docker_binary, "rm", "-f", name],
                capture_output=True,
                text=True,
                timeout=30,
            )


def _is_cli_fault(returncode: int, stderr: str) -> bool:
    """Distinguishes Docker CLI faults from payloads that exit with 125+.

    A Python payload can legitimately call ``sys.exit(125)``; the CLI fault
    signature is the exit code paired with a daemon-side error message.
    """
    markers = ("docker:", "error during connect", "no such image", "unable to find image")
    lowered = stderr.lower()
    cli_fault = returncode == _DOCKER_CLI_ERROR and any(m in lowered for m in markers)
    exec_fault = returncode in (126, 127) and "docker" in lowered
    return cli_fault or exec_fault


def ensure_image(
    config: DockerSandboxConfig | None = None,
    build_if_missing: bool = False,
    build_timeout: float = 600.0,
) -> bool:
    """Verifies the sandbox image exists, optionally building it from the bundled context.

    Returns ``True`` when the image is present after the call. Raises
    :class:`DockerUnavailableError` when a requested build fails.
    """
    cfg = config or DockerSandboxConfig()
    inspect = subprocess.run(
        [cfg.docker_binary, "image", "inspect", cfg.image],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if inspect.returncode == 0:
        return True
    if not build_if_missing:
        return False
    context = Path(__file__).parent / "image"
    build = subprocess.run(
        [cfg.docker_binary, "build", "--tag", cfg.image, str(context)],
        capture_output=True,
        text=True,
        timeout=build_timeout,
    )
    if build.returncode != 0:
        raise DockerUnavailableError(
            f"sandbox image build failed: {build.stderr.strip()[-500:]}"
        )
    return True

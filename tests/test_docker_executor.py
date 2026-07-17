"""Validates the Docker backend: hardening profile statically, behavior live.

Integration tests execute only when a Docker daemon is reachable and the
sandbox image is present; they are skipped otherwise so the suite stays green
on hosts without Docker.
"""

import pytest

from cognitivetree.sandbox.docker_executor import (
    DockerSandboxConfig,
    DockerSandboxExecutor,
    ensure_image,
)
from cognitivetree.sandbox.spec import (
    ExecutionRequest,
    ExecutionStatus,
    ResourceLimits,
)

_DAEMON_UP = DockerSandboxExecutor.is_available()
_IMAGE_READY = _DAEMON_UP and ensure_image(build_if_missing=False)

docker_required = pytest.mark.skipif(
    not _IMAGE_READY, reason="Docker daemon or sandbox image unavailable"
)


class TestCommandProfile:
    """Static assertions on the generated `docker run` invocation."""

    def build(self) -> list[str]:
        executor = DockerSandboxExecutor(
            DockerSandboxConfig(limits=ResourceLimits(memory_mb=128, cpus=0.5, pids=32))
        )
        return executor._build_command("ctree-sbx-test", ExecutionRequest(code="pass"))

    def test_isolation_flags_present(self) -> None:
        command = self.build()
        joined = " ".join(command)
        assert "--network none" in joined
        assert "--read-only" in joined
        assert "--cap-drop ALL" in joined
        assert "--security-opt no-new-privileges" in joined
        assert "--user 65534:65534" in joined
        assert "--rm" in command

    def test_resource_caps_reflect_limits(self) -> None:
        joined = " ".join(self.build())
        assert "--memory 128m" in joined
        assert "--memory-swap 128m" in joined
        assert "--cpus 0.5" in joined
        assert "--pids-limit 32" in joined

    def test_payload_is_passed_exec_form_not_shell(self) -> None:
        command = self.build()
        assert command[-4:] == ["python", "-I", "-c", "pass"]
        assert "sh" not in command
        assert "bash" not in command


def test_availability_probe_returns_bool() -> None:
    assert isinstance(DockerSandboxExecutor.is_available(), bool)


def test_probe_with_bogus_binary_is_false_not_raising() -> None:
    assert DockerSandboxExecutor.is_available("definitely-not-docker-xyz") is False


@docker_required
class TestLiveSandbox:
    """Behavioral checks against real disposable containers."""

    def setup_method(self) -> None:
        self.executor = DockerSandboxExecutor(
            DockerSandboxConfig(limits=ResourceLimits(timeout_seconds=20.0))
        )

    def test_clean_execution_round_trip(self) -> None:
        result = self.executor.execute(
            ExecutionRequest(code="print('sandbox says hello')")
        )
        assert result.ok, result.stderr
        assert result.stdout.strip() == "sandbox says hello"

    def test_network_egress_is_blocked(self) -> None:
        code = (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
            "except OSError:\n"
            "    raise SystemExit(42)\n"
            "raise SystemExit(0)\n"
        )
        result = self.executor.execute(ExecutionRequest(code=code))
        assert result.status is ExecutionStatus.COMPLETED
        assert result.exit_code == 42

    def test_root_filesystem_is_read_only(self) -> None:
        code = (
            "try:\n"
            "    open('/etc/poison', 'w').write('x')\n"
            "except OSError:\n"
            "    raise SystemExit(42)\n"
            "raise SystemExit(0)\n"
        )
        result = self.executor.execute(ExecutionRequest(code=code))
        assert result.exit_code == 42

    def test_tmp_remains_writable_scratch(self) -> None:
        code = (
            "open('/tmp/scratch.txt', 'w').write('data')\n"
            "print(open('/tmp/scratch.txt').read())\n"
        )
        result = self.executor.execute(ExecutionRequest(code=code))
        assert result.ok, result.stderr
        assert result.stdout.strip() == "data"

    def test_runaway_payload_is_killed_at_deadline(self) -> None:
        result = self.executor.execute(
            ExecutionRequest(code="while True: pass", timeout_seconds=3.0)
        )
        assert result.status is ExecutionStatus.TIMEOUT
        assert result.exit_code is None

    def test_payload_runs_unprivileged(self) -> None:
        result = self.executor.execute(
            ExecutionRequest(code="import os; print(os.getuid())")
        )
        assert result.ok, result.stderr
        assert result.stdout.strip() == "65534"

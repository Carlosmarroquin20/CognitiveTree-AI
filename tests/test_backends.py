"""Validates executor selection against daemons that cannot serve the image."""

import subprocess
from collections.abc import Callable, Iterator

import pytest

from cognitivetree.sandbox import backends
from cognitivetree.sandbox.docker_executor import (
    DockerSandboxExecutor,
    DockerUnavailableError,
)
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor


def _fake_info(
    stdout: str, returncode: int = 0
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr="")

    return run


class TestAvailabilityProbe:
    def test_linux_daemon_is_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_info("linux\n"))
        assert DockerSandboxExecutor.is_available() is True

    def test_windows_container_daemon_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # GitHub's Windows runners answer the probe in Windows-container mode,
        # where the Linux sandbox image can neither be built nor run.
        monkeypatch.setattr(subprocess, "run", _fake_info("windows\n"))
        assert DockerSandboxExecutor.is_available() is False

    def test_failed_probe_is_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_info("", returncode=1))
        assert DockerSandboxExecutor.is_available() is False


@pytest.fixture
def fresh_backend_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(backends, "_image_ready", False)
    monkeypatch.setattr(backends, "_image_error", None)
    yield


@pytest.mark.usefixtures("fresh_backend_state")
class TestSelection:
    def test_unreachable_daemon_selects_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(backends, "docker_available", lambda: False)
        executor, description = backends.select_executor()
        assert isinstance(executor, SubprocessExecutor)
        assert "daemon unreachable" in description

    def test_image_build_failure_degrades_to_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[bool] = []

        def failing_build(*_: object, **__: object) -> bool:
            calls.append(True)
            raise DockerUnavailableError("sandbox image build failed: no matching manifest")

        monkeypatch.setattr(backends, "docker_available", lambda: True)
        monkeypatch.setattr(backends, "ensure_image", failing_build)

        executor, description = backends.select_executor()
        assert isinstance(executor, SubprocessExecutor)
        assert "image unavailable" in description

        # The failure is remembered, so later sessions skip the slow rebuild.
        backends.select_executor()
        assert len(calls) == 1

    def test_ready_image_selects_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(backends, "docker_available", lambda: True)
        monkeypatch.setattr(backends, "ensure_image", lambda *_, **__: True)
        executor, description = backends.select_executor()
        assert isinstance(executor, DockerSandboxExecutor)
        assert description.startswith("docker")

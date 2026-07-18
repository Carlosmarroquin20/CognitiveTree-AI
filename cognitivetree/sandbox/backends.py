"""Executor backend selection with cached daemon probing.

Selection prefers the hardened Docker backend and falls back to the
host-process executor when no daemon answers. Probe results are cached for a
short interval: a streaming server builds one session per connection, and
paying a multi-second daemon probe on every request would dominate latency on
hosts without Docker. The cache is deliberately time-bounded so a daemon
started mid-process is picked up within one interval.
"""

from __future__ import annotations

import threading
import time

from cognitivetree.sandbox.docker_executor import (
    DockerSandboxConfig,
    DockerSandboxExecutor,
    ensure_image,
)
from cognitivetree.sandbox.executor import CodeExecutor
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor

_PROBE_TTL_SECONDS = 30.0

_lock = threading.Lock()
_probe_expiry = 0.0
_probe_result = False
_image_ready = False


def docker_available(ttl_seconds: float = _PROBE_TTL_SECONDS) -> bool:
    """Reports daemon reachability, caching the probe for ``ttl_seconds``."""
    global _probe_expiry, _probe_result
    now = time.monotonic()
    with _lock:
        if now < _probe_expiry:
            return _probe_result
    result = DockerSandboxExecutor.is_available()
    with _lock:
        _probe_result = result
        _probe_expiry = time.monotonic() + ttl_seconds
        return _probe_result


def select_executor() -> tuple[CodeExecutor, str]:
    """Returns the strongest available execution backend and its description.

    The sandbox image is built at most once per process; subsequent
    selections reuse the verified image without re-inspecting it.
    """
    global _image_ready
    if docker_available():
        config = DockerSandboxConfig()
        if not _image_ready:
            ensure_image(config, build_if_missing=True)
            _image_ready = True
        return DockerSandboxExecutor(config), f"docker ({config.image})"
    return (
        SubprocessExecutor(),
        "subprocess fallback (no isolation; Docker daemon unreachable)",
    )

"""Executor backend selection with cached daemon probing.

Selection prefers the hardened Docker backend and falls back to the
host-process executor when no daemon answers. Probe results are cached for a
short interval: a streaming server builds one session per connection, and
paying a multi-second daemon probe on every request would dominate latency on
hosts without Docker. The cache is deliberately time-bounded so a daemon
started mid-process is picked up within one interval.
"""

from __future__ import annotations

import logging
import threading
import time

from cognitivetree.sandbox.docker_executor import (
    DockerSandboxConfig,
    DockerSandboxExecutor,
    DockerUnavailableError,
    ensure_image,
)
from cognitivetree.sandbox.executor import CodeExecutor
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor

_PROBE_TTL_SECONDS = 30.0

_lock = threading.Lock()
_probe_expiry = 0.0
_probe_result = False
_image_ready = False
_image_error: str | None = None

logger = logging.getLogger(__name__)


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
    selections reuse the verified image without re-inspecting it. A failed
    build is remembered for the same lifetime and degrades to the host
    executor: retrying a build that may take minutes on every session would
    stall each connection, and the failure cause rarely heals mid-process.
    """
    global _image_ready, _image_error
    if docker_available() and _image_error is None:
        config = DockerSandboxConfig()
        if not _image_ready:
            try:
                ensure_image(config, build_if_missing=True)
            except DockerUnavailableError as exc:
                _image_error = str(exc)
                logger.warning("docker backend disabled for this process: %s", exc)
            else:
                _image_ready = True
        if _image_ready:
            return DockerSandboxExecutor(config), f"docker ({config.image})"
    if _image_error is not None:
        reason = "sandbox image unavailable"
    else:
        reason = "Docker daemon unreachable"
    return (
        SubprocessExecutor(),
        f"subprocess fallback (no isolation; {reason})",
    )

"""Standard-library HTTP server streaming reasoning runs over SSE."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol, runtime_checkable

from cognitivetree.ui.events import format_sse
from cognitivetree.ui.page import PAGE_HTML

logger = logging.getLogger(__name__)


@runtime_checkable
class StreamingSession(Protocol):
    """The whole surface the server needs from whatever it is streaming.

    Both a live :class:`~cognitivetree.session.ReasoningSession` and an
    archived :class:`~cognitivetree.persistence.ReplaySession` satisfy it
    without sharing a base class, which is what lets the same endpoint serve
    a search in progress and one that finished hours ago.
    """

    @property
    def task(self) -> str:
        """Returns the task statement being streamed."""
        ...

    def stream(self) -> Iterator[dict[str, Any]]:
        """Yields the run as an ordered sequence of envelopes."""
        ...


SessionFactory = Callable[[], StreamingSession]

_CLIENT_DISCONNECTS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)

_BUSY_RETRY_AFTER_SECONDS = 5


class _Handler(BaseHTTPRequestHandler):
    """Routes the two-endpoint surface: the page and the event stream."""

    server: StreamingUiServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - fixed by http.server
        if self.path in ("/", "/index.html"):
            self._serve_page()
        elif self.path == "/stream":
            self._serve_stream()
        else:
            self.send_error(404, "unknown path")

    def _serve_page(self) -> None:
        body = PAGE_HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self) -> None:
        """Runs a fresh session and streams its envelopes until completion.

        Each connection gets its own run; a disconnect stops the transmission
        while the worker thread drains the remaining envelopes and finishes.
        A request beyond the server's concurrency cap is refused with 503
        before any session is built.
        """
        slots = self.server.run_slots
        if slots is not None and not slots.acquire(blocking=False):
            self._refuse_busy()
            return
        try:
            self._stream_session()
        finally:
            if slots is not None:
                slots.release()

    def _stream_session(self) -> None:
        session = self.server.session_factory()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for envelope in session.stream():
                self.wfile.write(format_sse(envelope))
                self.wfile.flush()
        except _CLIENT_DISCONNECTS:
            logger.info("stream client disconnected mid-run")

    def _refuse_busy(self) -> None:
        body = b"concurrent run limit reached; retry shortly\n"
        self.send_response(503)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Retry-After", str(_BUSY_RETRY_AFTER_SECONDS))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Redirects access logging to the module logger."""
        logger.debug("%s - %s", self.address_string(), format % args)


class StreamingUiServer(ThreadingHTTPServer):
    """Threaded HTTP server bound to a session factory.

    Every ``/stream`` request receives an independent reasoning run, so
    multiple observers can trigger and watch runs concurrently.
    ``max_concurrent_runs`` bounds how many run at once; each one may spawn
    sandboxed executions and spend model quota, so an unbounded server can be
    exhausted by opening connections. ``None`` leaves it unbounded.
    """

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        session_factory: SessionFactory,
        max_concurrent_runs: int | None = None,
    ) -> None:
        if max_concurrent_runs is not None and max_concurrent_runs < 1:
            raise ValueError("max_concurrent_runs must be a positive integer")
        super().__init__(address, _Handler)
        self.session_factory = session_factory
        self.run_slots = (
            threading.BoundedSemaphore(max_concurrent_runs)
            if max_concurrent_runs is not None
            else None
        )

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        # socketserver types the address broadly enough to cover byte-encoded
        # families; interpolating those raw would yield a b'...' host.
        if isinstance(host, bytes):
            host = host.decode("utf-8")
        return f"http://{host}:{port}/"

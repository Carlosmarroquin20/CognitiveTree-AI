"""Validates SSE framing and the live HTTP surface end-to-end."""

import http.client
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import EndlessRun

from cognitivetree.session import ReasoningSession, build_reference_session
from cognitivetree.state import SearchPhase
from cognitivetree.ui.events import format_sse
from cognitivetree.ui.server import StreamingUiServer


def test_sse_framing() -> None:
    frame = format_sse({"type": "phase", "phase": "expansion"})
    text = frame.decode("utf-8")
    assert text.startswith("event: phase\ndata: ")
    assert text.endswith("\n\n")
    payload = json.loads(text.split("data: ", 1)[1].strip())
    assert payload == {"type": "phase", "phase": "expansion"}


class LiveServer:
    """Runs the streaming server on an ephemeral port for one test."""

    def __enter__(self) -> "LiveServer":
        self.server = StreamingUiServer(("127.0.0.1", 0), build_reference_session)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def test_page_is_served() -> None:
    with LiveServer() as live:
        connection = http.client.HTTPConnection("127.0.0.1", live.port, timeout=30)
        connection.request("GET", "/")
        response = connection.getresponse()
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert "text/html" in response.getheader("Content-Type", "")
        assert "CognitiveTree-AI" in body
        assert "EventSource" in body
        connection.close()


def test_unknown_path_is_404() -> None:
    with LiveServer() as live:
        connection = http.client.HTTPConnection("127.0.0.1", live.port, timeout=30)
        connection.request("GET", "/nope")
        assert connection.getresponse().status == 404
        connection.close()


def test_stream_delivers_full_run_over_http() -> None:
    with LiveServer() as live:
        connection = http.client.HTTPConnection("127.0.0.1", live.port, timeout=120)
        connection.request("GET", "/stream")
        response = connection.getresponse()
        assert response.status == 200
        assert "text/event-stream" in response.getheader("Content-Type", "")

        events: list[tuple[str, dict]] = []
        current_event = ""
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                current_event = line[len("event: "):]
            elif line.startswith("data: "):
                events.append((current_event, json.loads(line[len("data: "):])))
            if events and events[-1][0] == "result":
                break
        connection.close()

        names = [name for name, _ in events]
        assert names[0] == "phase"
        assert "snapshot" in names
        assert names[-1] == "result"
        assert events[-1][1]["outcome"] == "succeeded"


def test_client_disconnect_mid_stream_leaves_the_server_healthy() -> None:
    # A browser closing the tab mid-run must not take the handler thread with
    # it; the server has to keep answering afterwards.
    with LiveServer() as live:
        aborted = http.client.HTTPConnection("127.0.0.1", live.port, timeout=60)
        aborted.request("GET", "/stream")
        response = aborted.getresponse()
        assert response.status == 200
        response.readline()
        aborted.close()

        survivor = http.client.HTTPConnection("127.0.0.1", live.port, timeout=60)
        survivor.request("GET", "/")
        assert survivor.getresponse().status == 200
        survivor.close()


class GatedSession:
    """Streams one envelope, then holds its run slot until released."""

    def __init__(self, release: threading.Event) -> None:
        self._release = release

    @property
    def task(self) -> str:
        return "gated"

    def stream(self) -> Iterator[dict[str, Any]]:
        yield {"type": "phase", "phase": "selection"}
        self._release.wait(timeout=30)
        yield {"type": "result", "outcome": "succeeded"}


def open_stream(port: int) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request("GET", "/stream")
    return connection, connection.getresponse()


def test_streams_beyond_the_concurrency_cap_are_refused() -> None:
    release = threading.Event()
    server = StreamingUiServer(
        ("127.0.0.1", 0), lambda: GatedSession(release), max_concurrent_runs=1
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        first, first_response = open_stream(port)
        assert first_response.status == 200
        first_response.readline()  # the run is now holding the only slot

        second, refused = open_stream(port)
        assert refused.status == 503
        assert refused.getheader("Retry-After") == "5"
        assert b"limit" in refused.read()
        second.close()

        release.set()
        first_response.read()  # drains the stream so the slot is returned
        first.close()

        third, admitted = open_stream(port)
        assert admitted.status == 200
        admitted.read()
        third.close()
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_concurrency_cap_must_be_positive() -> None:
    with pytest.raises(ValueError):
        StreamingUiServer(("127.0.0.1", 0), build_reference_session, max_concurrent_runs=0)


def test_disconnecting_client_cancels_the_live_run(endless_run: EndlessRun) -> None:
    run = endless_run
    server = StreamingUiServer(
        ("127.0.0.1", 0), lambda: ReasoningSession("endless", run.factory)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection, response = open_stream(server.server_address[1])
        assert response.status == 200
        response.readline()
        # The response holds its own reference to the socket, so both must
        # close before the server can observe the disconnect.
        response.close()
        connection.close()

        # The next envelope written to the closed socket fails, the handler
        # closes the stream, and the run stops at its following boundary.
        assert run.settled.wait(timeout=15)
        assert run.terminal is SearchPhase.CANCELLED
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_page_escapes_quotes_in_untrusted_node_content() -> None:
    # Node content is model output and lands inside a title="..." attribute.
    # Escaping that misses quotes lets content open an event-handler
    # attribute, which was verified exploitable in a browser before the fix.
    from cognitivetree.ui.page import PAGE_HTML

    assert '\'"\': "&quot;"' in PAGE_HTML
    assert '"\'": "&#39;"' in PAGE_HTML
    assert "div.innerHTML" not in PAGE_HTML


class TestSecurityHeaders:
    """The page ships a policy that pins its own script and stylesheet."""

    def test_page_carries_csp_referrer_and_nosniff(self) -> None:
        from cognitivetree.ui.page import PAGE_CSP

        with LiveServer() as live:
            connection = http.client.HTTPConnection("127.0.0.1", live.port, timeout=30)
            connection.request("GET", "/")
            response = connection.getresponse()
            response.read()
            connection.close()
        assert response.getheader("Content-Security-Policy") == PAGE_CSP
        assert response.getheader("Referrer-Policy") == "no-referrer"
        assert response.getheader("X-Content-Type-Options") == "nosniff"

    def test_error_responses_are_not_sniffed(self) -> None:
        with LiveServer() as live:
            connection = http.client.HTTPConnection("127.0.0.1", live.port, timeout=30)
            connection.request("GET", "/nope")
            response = connection.getresponse()
            response.read()
            connection.close()
        assert response.status == 404
        assert response.getheader("X-Content-Type-Options") == "nosniff"

    def test_policy_hashes_match_the_served_markup(self) -> None:
        import base64
        import hashlib
        import re

        from cognitivetree.ui.page import PAGE_CSP, PAGE_HTML

        for tag, directive in (("script", "script-src"), ("style", "style-src")):
            # One block per tag, or the hash would leave the others blocked.
            assert PAGE_HTML.count(f"<{tag}>") == 1
            body = re.search(f"<{tag}>(.*?)</{tag}>", PAGE_HTML, re.DOTALL)
            assert body is not None
            digest = base64.b64encode(hashlib.sha256(body.group(1).encode()).digest())
            assert f"{directive} 'sha256-{digest.decode()}'" in PAGE_CSP

    def test_policy_forbids_everything_it_does_not_name(self) -> None:
        from cognitivetree.ui.page import PAGE_CSP, PAGE_HTML

        assert PAGE_CSP.startswith("default-src 'none'")
        assert "unsafe-inline" not in PAGE_CSP
        assert "frame-ancestors 'none'" in PAGE_CSP
        # Inline style attributes would be refused under a hash-only style-src.
        assert "style=" not in PAGE_HTML

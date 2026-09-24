"""Validates SSE framing and the live HTTP surface end-to-end."""

import http.client
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from cognitivetree.session import build_reference_session
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

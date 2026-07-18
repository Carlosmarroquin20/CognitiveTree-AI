"""Validates SSE framing and the live HTTP surface end-to-end."""

import http.client
import json
import threading

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

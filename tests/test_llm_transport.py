"""Exercises the real HTTP transport against a local server.

Every other LLM test injects a fake transport, which leaves
:class:`UrllibTransport` — the code that actually runs against Ollama, vLLM,
llama.cpp, or LM Studio — unexercised. These tests stand up a standard-library
HTTP server on an ephemeral loopback port and drive the genuine request path
through it: no network, no model, but real sockets, real headers, and real
status handling.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from cognitivetree.llm.client import ChatMessage, CompletionRequest, LlmError
from cognitivetree.llm.openai_compatible import OpenAICompatibleClient, UrllibTransport

# Mutable per-test script the handler reads; each test rebinds it before use.
_SCRIPT: dict[str, object] = {}


class _Handler(BaseHTTPRequestHandler):
    """Replays a scripted response and records what it received."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - fixed by http.server
        length = int(self.headers.get("Content-Length", "0"))
        _SCRIPT["received_body"] = self.rfile.read(length)
        _SCRIPT["received_headers"] = dict(self.headers)
        _SCRIPT["received_path"] = self.path
        _SCRIPT["calls"] = int(_SCRIPT.get("calls", 0)) + 1

        statuses = _SCRIPT.get("statuses") or [200]
        index = min(int(_SCRIPT["calls"]) - 1, len(statuses) - 1)  # type: ignore[arg-type]
        status = statuses[index]  # type: ignore[index]
        body = str(_SCRIPT.get("body", "{}")).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Silences the default stderr access log."""


@pytest.fixture
def server() -> Iterator[str]:
    """Serves the scripted handler and yields its base URL."""
    _SCRIPT.clear()
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def completion_body(text: str = "hello from the model") -> str:
    return json.dumps(
        {
            "choices": [{"message": {"content": text}}],
            "model": "llama3.3",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }
    )


def request() -> CompletionRequest:
    return CompletionRequest(
        messages=(
            ChatMessage(role="system", content="system prompt"),
            ChatMessage(role="user", content="user prompt"),
        ),
        temperature=0.25,
        max_tokens=128,
    )


class TestUrllibTransport:
    """The default transport, driven over real sockets."""

    def test_successful_post_returns_status_and_body(self, server: str) -> None:
        _SCRIPT["body"] = '{"ok": true}'
        status, body = UrllibTransport().post(
            f"{server}/v1/chat/completions", {"a": 1}, {}, 10.0
        )
        assert status == 200
        assert json.loads(body) == {"ok": True}

    def test_payload_is_sent_as_json(self, server: str) -> None:
        _SCRIPT["body"] = "{}"
        UrllibTransport().post(
            f"{server}/v1/chat/completions", {"model": "llama3.3", "n": 2}, {}, 10.0
        )
        assert json.loads(_SCRIPT["received_body"]) == {"model": "llama3.3", "n": 2}
        assert _SCRIPT["received_headers"]["Content-Type"] == "application/json"

    def test_extra_headers_reach_the_server(self, server: str) -> None:
        _SCRIPT["body"] = "{}"
        UrllibTransport().post(
            f"{server}/v1/chat/completions",
            {},
            {"Authorization": "Bearer secret-token"},
            10.0,
        )
        assert _SCRIPT["received_headers"]["Authorization"] == "Bearer secret-token"

    def test_path_is_preserved(self, server: str) -> None:
        _SCRIPT["body"] = "{}"
        UrllibTransport().post(f"{server}/v1/chat/completions", {}, {}, 10.0)
        assert _SCRIPT["received_path"] == "/v1/chat/completions"

    @pytest.mark.parametrize("status", [400, 404, 429, 500, 503])
    def test_error_statuses_are_returned_not_raised(
        self, server: str, status: int
    ) -> None:
        # The client layer decides what is retryable, so the transport must
        # surface the status rather than turning it into an exception.
        _SCRIPT["statuses"] = [status]
        _SCRIPT["body"] = '{"error": "upstream"}'
        code, body = UrllibTransport().post(f"{server}/v1", {}, {}, 10.0)
        assert code == status
        assert b"upstream" in body

    def test_unreachable_host_raises_llm_error(self) -> None:
        # Port 1 on loopback refuses connections on every supported platform.
        with pytest.raises(LlmError, match="backend unreachable"):
            UrllibTransport().post("http://127.0.0.1:1/v1", {}, {}, 5.0)

    def test_malformed_url_raises_llm_error(self) -> None:
        with pytest.raises((LlmError, ValueError)):
            UrllibTransport().post("not-a-url", {}, {}, 5.0)


class TestClientOverRealHttp:
    """The full client stack against an endpoint that speaks the dialect."""

    def build(self, server: str, **kwargs: object) -> OpenAICompatibleClient:
        kwargs.setdefault("retry_backoff_seconds", 0.0)
        return OpenAICompatibleClient(
            base_url=f"{server}/v1", model="llama3.3", **kwargs  # type: ignore[arg-type]
        )

    def test_completion_round_trip(self, server: str) -> None:
        _SCRIPT["body"] = completion_body()
        response = self.build(server).complete(request())

        assert response.text == "hello from the model"
        assert response.model == "llama3.3"
        assert response.prompt_tokens == 11
        assert response.completion_tokens == 7

    def test_request_shape_matches_the_openai_dialect(self, server: str) -> None:
        _SCRIPT["body"] = completion_body()
        self.build(server).complete(request())

        sent = json.loads(_SCRIPT["received_body"])
        assert sent["model"] == "llama3.3"
        assert sent["temperature"] == 0.25
        assert sent["max_tokens"] == 128
        assert sent["messages"] == [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ]
        assert _SCRIPT["received_path"] == "/v1/chat/completions"

    def test_api_key_travels_as_a_bearer_token(self, server: str) -> None:
        _SCRIPT["body"] = completion_body()
        self.build(server, api_key="secret-token").complete(request())
        assert _SCRIPT["received_headers"]["Authorization"] == "Bearer secret-token"

    def test_server_error_is_retried_then_succeeds(self, server: str) -> None:
        _SCRIPT["statuses"] = [500, 200]
        _SCRIPT["body"] = completion_body()
        response = self.build(server, max_retries=1).complete(request())

        assert response.text == "hello from the model"
        assert _SCRIPT["calls"] == 2

    def test_rate_limiting_is_retried_then_succeeds(self, server: str) -> None:
        _SCRIPT["statuses"] = [429, 200]
        _SCRIPT["body"] = completion_body()
        response = self.build(server, max_retries=1).complete(request())

        assert response.text == "hello from the model"
        assert _SCRIPT["calls"] == 2

    def test_client_error_is_not_retried(self, server: str) -> None:
        _SCRIPT["statuses"] = [404]
        _SCRIPT["body"] = '{"error": "no such model"}'
        with pytest.raises(LlmError, match="404"):
            self.build(server, max_retries=3).complete(request())
        assert _SCRIPT["calls"] == 1

    def test_persistent_server_error_exhausts_retries(self, server: str) -> None:
        _SCRIPT["statuses"] = [503]
        _SCRIPT["body"] = "{}"
        with pytest.raises(LlmError, match="503"):
            self.build(server, max_retries=2).complete(request())
        assert _SCRIPT["calls"] == 3

    def test_non_json_body_is_reported_as_malformed(self, server: str) -> None:
        _SCRIPT["body"] = "<html>gateway timeout</html>"
        with pytest.raises(LlmError, match="malformed"):
            self.build(server).complete(request())

    def test_unreachable_endpoint_surfaces_as_llm_error(self) -> None:
        client = OpenAICompatibleClient(
            base_url="http://127.0.0.1:1/v1", model="llama3.3", max_retries=0
        )
        with pytest.raises(LlmError, match="unreachable"):
            client.complete(request())

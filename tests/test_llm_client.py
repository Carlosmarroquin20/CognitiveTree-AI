"""Validates request shaping, retries, and error mapping of the LLM client."""

import json

import pytest

from cognitivetree.llm.client import ChatMessage, CompletionRequest, LlmError
from cognitivetree.llm.openai_compatible import OpenAICompatibleClient


def ok_body(text: str = "completion text") -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"content": text}}],
            "model": "llama3.3",
            "usage": {"prompt_tokens": 12, "completion_tokens": 34},
        }
    ).encode()


class FakeTransport:
    """Replays queued responses while recording every request."""

    def __init__(self, outcomes: list) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def post(self, url, payload, headers, timeout):
        self.calls.append(
            {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
        )
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def request() -> CompletionRequest:
    return CompletionRequest(
        messages=(
            ChatMessage(role="system", content="system prompt"),
            ChatMessage(role="user", content="user prompt"),
        ),
        temperature=0.3,
        max_tokens=256,
    )


def client_with(transport: FakeTransport, **kwargs) -> OpenAICompatibleClient:
    defaults = {"base_url": "http://localhost:11434/v1", "model": "llama3.3"}
    defaults.update(kwargs)
    return OpenAICompatibleClient(transport=transport, **defaults)


def test_request_shaping_and_endpoint_join() -> None:
    transport = FakeTransport([(200, ok_body())])
    response = client_with(transport).complete(request())

    call = transport.calls[0]
    assert call["url"] == "http://localhost:11434/v1/chat/completions"
    assert call["payload"]["model"] == "llama3.3"
    assert call["payload"]["temperature"] == 0.3
    assert call["payload"]["max_tokens"] == 256
    assert call["payload"]["messages"][0] == {
        "role": "system",
        "content": "system prompt",
    }
    assert response.text == "completion text"
    assert response.prompt_tokens == 12
    assert response.completion_tokens == 34


def test_api_key_becomes_bearer_header() -> None:
    transport = FakeTransport([(200, ok_body())])
    client_with(transport, api_key="secret-token").complete(request())
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer secret-token"


def test_no_auth_header_without_api_key() -> None:
    transport = FakeTransport([(200, ok_body())])
    client_with(transport).complete(request())
    assert "Authorization" not in transport.calls[0]["headers"]


def test_server_error_is_retried_then_succeeds() -> None:
    transport = FakeTransport([(500, b"overloaded"), (200, ok_body())])
    response = client_with(transport, max_retries=1).complete(request())
    assert response.text == "completion text"
    assert len(transport.calls) == 2


def test_connection_error_is_retried_then_raised() -> None:
    transport = FakeTransport(
        [LlmError("backend unreachable"), LlmError("backend unreachable")]
    )
    with pytest.raises(LlmError, match="unreachable"):
        client_with(transport, max_retries=1).complete(request())
    assert len(transport.calls) == 2


def test_client_error_is_not_retried() -> None:
    transport = FakeTransport([(404, b"model not found")])
    with pytest.raises(LlmError, match="404"):
        client_with(transport, max_retries=3).complete(request())
    assert len(transport.calls) == 1


def test_malformed_body_raises_llm_error() -> None:
    transport = FakeTransport([(200, b"not json at all")])
    with pytest.raises(LlmError, match="malformed"):
        client_with(transport).complete(request())


def test_invalid_construction_is_rejected() -> None:
    with pytest.raises(ValueError):
        OpenAICompatibleClient(base_url=" ", model="m")
    with pytest.raises(ValueError):
        OpenAICompatibleClient(base_url="http://x", model="")
    with pytest.raises(ValueError):
        ChatMessage(role="tool", content="x")
    with pytest.raises(ValueError):
        CompletionRequest(messages=())

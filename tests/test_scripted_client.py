"""Validates the deterministic scripted LLM client."""

import pytest

from cognitivetree.llm.client import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LlmError,
)
from cognitivetree.llm.scripted import ScriptedLlmClient


def request(text: str = "user prompt") -> CompletionRequest:
    return CompletionRequest(
        messages=(
            ChatMessage(role="system", content="system prompt"),
            ChatMessage(role="user", content=text),
        )
    )


def test_responder_output_is_wrapped_with_usage() -> None:
    client = ScriptedLlmClient(lambda _: "one two three", model="scripted-llama")
    response = client.complete(request())
    assert response.text == "one two three"
    assert response.model == "scripted-llama"
    assert response.completion_tokens == 3
    assert response.prompt_tokens == 4  # "system prompt" + "user prompt"


def test_responder_may_return_full_response_verbatim() -> None:
    canned = CompletionResponse(text="verbatim", model="m", completion_tokens=9)
    client = ScriptedLlmClient(lambda _: canned)
    assert client.complete(request()) is canned


def test_requests_are_recorded_in_order() -> None:
    client = ScriptedLlmClient(lambda _: "ok")
    client.complete(request("first"))
    client.complete(request("second"))
    assert [r.messages[-1].content for r in client.requests] == ["first", "second"]


def test_from_sequence_replays_then_raises() -> None:
    client = ScriptedLlmClient.from_sequence(["a", "b"])
    assert client.complete(request()).text == "a"
    assert client.complete(request()).text == "b"
    with pytest.raises(LlmError, match="exhausted"):
        client.complete(request())


def test_responder_sees_request_content() -> None:
    seen: list[str] = []

    def responder(req: CompletionRequest) -> str:
        seen.append(req.messages[-1].content)
        return "ack"

    ScriptedLlmClient(responder).complete(request("inspect me"))
    assert seen == ["inspect me"]

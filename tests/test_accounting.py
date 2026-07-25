"""Validates transparent token accounting over an LLM client."""

from cognitivetree.llm.client import ChatMessage, CompletionRequest, CompletionResponse
from cognitivetree.observability import AccountingLlmClient, TokenUsage


class FixedClient:
    """Returns prearranged responses so accounting math is exact."""

    def __init__(self, responses: list[CompletionResponse]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _request() -> CompletionRequest:
    return CompletionRequest(messages=(ChatMessage(role="user", content="hi"),))


def test_usage_accumulates_across_calls() -> None:
    inner = FixedClient(
        [
            CompletionResponse(text="a", prompt_tokens=10, completion_tokens=3),
            CompletionResponse(text="b", prompt_tokens=7, completion_tokens=5),
        ]
    )
    client = AccountingLlmClient(inner)

    assert client.complete(_request()).text == "a"
    assert client.complete(_request()).text == "b"

    usage = client.usage
    assert usage.calls == 2
    assert usage.prompt_tokens == 17
    assert usage.completion_tokens == 8
    assert usage.total_tokens == 25


def test_wrapper_is_transparent() -> None:
    original = CompletionResponse(text="verbatim", model="m", completion_tokens=4)
    client = AccountingLlmClient(FixedClient([original]))
    assert client.complete(_request()) is original


def test_zero_usage_before_any_call() -> None:
    client = AccountingLlmClient(FixedClient([]))
    assert client.usage == TokenUsage(calls=0, prompt_tokens=0, completion_tokens=0)


def test_token_usage_serialization() -> None:
    usage = TokenUsage(calls=3, prompt_tokens=100, completion_tokens=40)
    assert usage.to_dict() == {
        "calls": 3,
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
    }

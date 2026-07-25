"""Transparent LLM client wrapper that tallies token consumption."""

from __future__ import annotations

from cognitivetree.llm.client import CompletionRequest, CompletionResponse, LlmClient
from cognitivetree.observability.metrics import TokenUsage


class AccountingLlmClient:
    """Wraps any ``LlmClient`` and accumulates its reported token usage.

    The wrapper is fully transparent: it forwards every request to the inner
    client and returns its response unchanged, recording the usage figures the
    backend reports along the way. Assembling a session around the wrapper
    yields token accounting for the whole run without touching the generator,
    the critic, or the search core.
    """

    def __init__(self, inner: LlmClient) -> None:
        self._inner = inner
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        response = self._inner.complete(request)
        self._calls += 1
        self._prompt_tokens += response.prompt_tokens
        self._completion_tokens += response.completion_tokens
        return response

    @property
    def usage(self) -> TokenUsage:
        """Returns the usage accumulated across every completion so far."""
        return TokenUsage(
            calls=self._calls,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )

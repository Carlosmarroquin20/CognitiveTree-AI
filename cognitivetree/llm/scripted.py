"""Deterministic ``LlmClient`` doubles for offline demos and tests.

A real model backend is not always available — CI runners, air-gapped hosts,
and quick demonstrations all need the reasoning loop to run without one. The
scripted client satisfies the :class:`~cognitivetree.llm.client.LlmClient`
contract with prearranged completions, so the entire adapter stack (prompt
assembly, candidate parsing, JSON critique parsing, revision-note injection)
executes through its production code paths against fully predictable output.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from cognitivetree.llm.client import (
    CompletionRequest,
    CompletionResponse,
    LlmError,
)

Responder = Callable[[CompletionRequest], "str | CompletionResponse"]


def _estimate_tokens(text: str) -> int:
    """Approximates a token count by whitespace segmentation.

    The estimate exists only to populate usage accounting with a plausible,
    monotonic value; it is not a tokenizer and must not be relied on for
    billing-grade counts.
    """
    return len(text.split())


class ScriptedLlmClient:
    """Replays scripted completions so the LLM stack runs without a model.

    Two construction modes cover the common needs. A *responder* maps each
    request to its completion, keeping multi-role conversations — generator
    versus critic, with or without revision notes — legible and independent of
    call order. :meth:`from_sequence` replays fixed completions by arrival
    order for simpler cases. Every request is retained on :attr:`requests` for
    inspection.
    """

    def __init__(self, responder: Responder, model: str = "scripted") -> None:
        self._responder = responder
        self._model = model
        self.requests: list[CompletionRequest] = []

    @classmethod
    def from_sequence(
        cls, completions: Sequence[str], model: str = "scripted"
    ) -> ScriptedLlmClient:
        """Builds a client that returns ``completions`` in order, then raises."""
        pending = list(completions)

        def responder(_: CompletionRequest) -> str:
            if not pending:
                raise LlmError("scripted completion sequence exhausted")
            return pending.pop(0)

        return cls(responder, model=model)

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Records ``request`` and returns the scripted completion for it."""
        self.requests.append(request)
        produced = self._responder(request)
        if isinstance(produced, CompletionResponse):
            return produced
        return CompletionResponse(
            text=produced,
            model=self._model,
            prompt_tokens=sum(_estimate_tokens(m.content) for m in request.messages),
            completion_tokens=_estimate_tokens(produced),
        )

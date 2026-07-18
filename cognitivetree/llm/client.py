"""Model-agnostic completion contracts for LLM backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class LlmError(RuntimeError):
    """Raised when a completion cannot be obtained from the backend."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Single message in a chat-completion conversation."""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported message role {self.role!r}")


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    """Parameters of one completion call."""

    messages: tuple[ChatMessage, ...]
    temperature: float = 0.7
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("messages must not be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must lie within [0.0, 2.0]")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")


@dataclass(frozen=True, slots=True)
class CompletionResponse:
    """Completion text together with backend accounting."""

    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class LlmClient(Protocol):
    """Produces a completion for a fully specified request.

    Implementations raise :class:`LlmError` for every failure mode — network,
    protocol, or payload — so callers never handle transport-specific
    exceptions.
    """

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Returns the backend completion for ``request``."""
        ...

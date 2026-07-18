"""LLM backend adapters implementing the core policy contracts.

The layer speaks the OpenAI-compatible chat-completions dialect, which covers
Ollama, vLLM, llama.cpp server, and LM Studio for open-source models such as
Llama 3.3 and Qwen 2.5. Everything is built on the standard library; the only
integration surface with the search core is the policy contracts, so the
adapters remain swappable and unit-testable through injected transports.
"""

from cognitivetree.llm.client import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LlmClient,
    LlmError,
)
from cognitivetree.llm.critic import LlmCritic
from cognitivetree.llm.generator import LlmThoughtGenerator
from cognitivetree.llm.openai_compatible import OpenAICompatibleClient

__all__ = [
    "ChatMessage",
    "CompletionRequest",
    "CompletionResponse",
    "LlmClient",
    "LlmCritic",
    "LlmError",
    "LlmThoughtGenerator",
    "OpenAICompatibleClient",
]

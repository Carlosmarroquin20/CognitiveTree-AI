"""LLM-backed implementation of the ThoughtGenerator contract."""

from __future__ import annotations

from cognitivetree.llm.client import ChatMessage, CompletionRequest, LlmClient
from cognitivetree.llm.parsing import split_candidates
from cognitivetree.llm.prompts import (
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_USER_TEMPLATE,
    REVISION_BLOCK_TEMPLATE,
)
from cognitivetree.node import ThoughtNode

_REVISION_NOTES_KEY = "revision_notes"


class LlmThoughtGenerator:
    """Expands nodes by sampling candidate thoughts from a chat model.

    The prompt carries the task (root content), the reasoning path down to
    the node, and — when the revision policy reopened the node — the compiled
    revision notes, so critique feedback conditions the next wave of
    candidates. Backend failures propagate as
    :class:`~cognitivetree.llm.client.LlmError`; generation is essential to
    the search, so the controller converts the raised error into a FAILED
    run instead of silently exhausting the tree.
    """

    def __init__(
        self,
        client: LlmClient,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        """Returns up to ``k`` model-proposed continuations of ``node``."""
        request = CompletionRequest(
            messages=(
                ChatMessage(role="system", content=GENERATOR_SYSTEM_PROMPT),
                ChatMessage(role="user", content=self._render_user_message(node, k)),
            ),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        response = self._client.complete(request)
        return split_candidates(response.text)[:k]

    def _render_user_message(self, node: ThoughtNode, k: int) -> str:
        """Builds the expansion prompt for ``node``."""
        path = node.path_from_root()
        task = path[0].content
        steps = path[1:]
        if steps:
            rendered_path = "\n".join(
                f"{index}. {step.content}" for index, step in enumerate(steps, start=1)
            )
        else:
            rendered_path = "(no thoughts yet; propose opening thoughts)"

        notes = str(node.metadata.get(_REVISION_NOTES_KEY, "")).strip()
        revision_block = (
            REVISION_BLOCK_TEMPLATE.format(notes=notes) if notes else ""
        )
        return GENERATOR_USER_TEMPLATE.format(
            task=task, path=rendered_path, revision_block=revision_block, k=k
        )

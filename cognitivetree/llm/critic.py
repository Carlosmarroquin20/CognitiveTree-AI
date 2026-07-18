"""LLM-backed implementation of the Critic contract."""

from __future__ import annotations

import json
import logging

from cognitivetree.llm.client import ChatMessage, CompletionRequest, LlmClient, LlmError
from cognitivetree.llm.parsing import extract_json_object
from cognitivetree.llm.prompts import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE
from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Critique, FailureClass
from cognitivetree.sandbox.evaluation import METADATA_KEY as EXECUTION_KEY

logger = logging.getLogger(__name__)

_STDERR_TAIL_CHARS = 1500


class LlmCritic:
    """Diagnoses failed thoughts with a chat model.

    The critic degrades instead of failing: backend errors and unparseable
    completions return ``None`` so a network blip never aborts a long search.
    Deployments chain it behind the deterministic
    :class:`~cognitivetree.feedback.execution_critic.ExecutionTraceCritic`
    (see :class:`~cognitivetree.feedback.composite.ChainedCritic`) or use it
    alone when semantic diagnosis is worth a model call per failure.
    """

    def __init__(
        self,
        client: LlmClient,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens

    def critique(self, node: ThoughtNode) -> Critique | None:
        """Returns a model-produced critique for ``node``, or ``None``."""
        request = CompletionRequest(
            messages=(
                ChatMessage(role="system", content=CRITIC_SYSTEM_PROMPT),
                ChatMessage(role="user", content=self._render_user_message(node)),
            ),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        try:
            response = self._client.complete(request)
        except LlmError as exc:
            logger.warning("critic degraded: completion failed: %s", exc)
            return None
        return self._parse_critique(response.text)

    def _render_user_message(self, node: ThoughtNode) -> str:
        """Builds the critique prompt from the node and its execution record."""
        record = node.metadata.get(EXECUTION_KEY)
        if record is None:
            execution = "(no execution record; the thought was rejected unexecuted)"
        else:
            compact = dict(record)
            stderr = str(compact.get("stderr", ""))
            if len(stderr) > _STDERR_TAIL_CHARS:
                compact["stderr"] = "..." + stderr[-_STDERR_TAIL_CHARS:]
            execution = json.dumps(compact, indent=2)
        return CRITIC_USER_TEMPLATE.format(content=node.content, execution=execution)

    def _parse_critique(self, text: str) -> Critique | None:
        """Maps the completion JSON onto a validated Critique."""
        document = extract_json_object(text)
        if document is None:
            logger.warning("critic degraded: completion carried no JSON object")
            return None
        try:
            failure_class = FailureClass(str(document.get("failure_class", "")))
        except ValueError:
            failure_class = FailureClass.EXCEPTION
        summary = str(document.get("summary", "")).strip()
        guidance = str(document.get("guidance", "")).strip()
        if not guidance:
            logger.warning("critic degraded: completion carried no guidance")
            return None
        try:
            severity = float(document.get("severity", 0.7))
        except (TypeError, ValueError):
            severity = 0.7
        severity = min(1.0, max(0.0, severity))
        return Critique(
            failure_class=failure_class,
            summary=summary or "model critique without summary",
            guidance=guidance,
            severity=severity,
        )

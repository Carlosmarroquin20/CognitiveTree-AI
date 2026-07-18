"""Revision policy governing when saturated nodes earn re-expansion."""

from __future__ import annotations

from cognitivetree.node import NodeStatus, ThoughtNode
from cognitivetree.policies import CRITIQUE_METADATA_KEY

REVISION_NOTES_KEY = "revision_notes"
REVISION_ATTEMPTS_KEY = "revision_attempts"


def compile_revision_notes(node: ThoughtNode) -> str:
    """Aggregates the critiques of ``node``'s failed children into guidance notes.

    Notes are deduplicated by guidance text while preserving first-seen order,
    so repeated failure modes surface once. Generators read the compiled notes
    from ``node.metadata["revision_notes"]`` when producing revised candidates;
    LLM-backed generators inject them into the prompt verbatim.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for child in node.children:
        critique = child.metadata.get(CRITIQUE_METADATA_KEY)
        if not critique:
            continue
        guidance = critique.get("guidance", "").strip()
        if not guidance or guidance in seen:
            continue
        seen.add(guidance)
        lines.append(f"- [{critique.get('failure_class', 'unknown')}] {guidance}")
    return "\n".join(lines)


class BoundedRevisionPolicy:
    """Grants a limited number of critique-informed re-expansions per node.

    A node qualifies for revision only when every child has failed (none live,
    none terminal), the attempt budget is not exhausted, and — unless
    ``require_guidance`` is disabled — at least one child carries a critique
    to learn from. Granting a revision is a committing act: the policy stores
    the compiled notes and increments the attempt counter before returning.
    """

    def __init__(self, max_attempts: int = 1, require_guidance: bool = True) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self._max_attempts = max_attempts
        self._require_guidance = require_guidance

    def revise(self, node: ThoughtNode) -> bool:
        """Prepares ``node`` for re-expansion and reports whether to proceed."""
        if not node.children:
            return False
        if any(child.is_live for child in node.children):
            return False
        if any(child.status is NodeStatus.TERMINAL for child in node.children):
            return False
        attempts = int(node.metadata.get(REVISION_ATTEMPTS_KEY, 0))
        if attempts >= self._max_attempts:
            return False
        notes = compile_revision_notes(node)
        if self._require_guidance and not notes:
            return False
        node.metadata[REVISION_NOTES_KEY] = notes
        node.metadata[REVISION_ATTEMPTS_KEY] = attempts + 1
        return True

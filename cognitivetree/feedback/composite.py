"""Composition helpers for critic policies."""

from __future__ import annotations

from typing import Sequence

from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Critic, Critique


class ChainedCritic:
    """Delegates to an ordered sequence of critics, returning the first verdict.

    The canonical deployment chains the deterministic execution-trace critic
    first (free, covers mechanical failures) and an LLM critic second, so
    model calls are spent only on failures the trace cannot explain.
    """

    def __init__(self, critics: Sequence[Critic]) -> None:
        if not critics:
            raise ValueError("critics must contain at least one critic")
        self._critics = tuple(critics)

    def critique(self, node: ThoughtNode) -> Critique | None:
        """Returns the first non-``None`` critique in chain order."""
        for critic in self._critics:
            verdict = critic.critique(node)
            if verdict is not None:
                return verdict
        return None

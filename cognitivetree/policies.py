"""Policy contracts decoupling search control from model-specific behavior.

The controller depends only on these protocols. Phase 2 supplies an evaluator
backed by sandboxed code execution; Phase 3 layers critique and reward models
on top; LLM-backed generators (Llama, Qwen) attach here without any change to
the search core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cognitivetree.node import ThoughtNode


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Verdict produced by an evaluator for a single thought node.

    Attributes:
        score: Normalized quality signal in ``[0.0, 1.0]``.
        is_terminal: Marks the thought as a completed line of reasoning; the
            controller accepts or prunes it based on the acceptance threshold,
            since completed thoughts cannot be extended further.
        rationale: Free-form justification retained on the node for tracing.
    """

    score: float
    is_terminal: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must lie within [0.0, 1.0]")


@runtime_checkable
class ThoughtGenerator(Protocol):
    """Produces candidate continuation thoughts for a frontier node."""

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        """Returns up to ``k`` candidate thoughts extending ``node``.

        Implementations may return fewer than ``k`` candidates, or an empty
        list when the node admits no viable continuation; the controller
        treats an empty result as a dead end and backtracks.
        """
        ...


@runtime_checkable
class ThoughtEvaluator(Protocol):
    """Scores a thought node and decides whether it completes the task."""

    def evaluate(self, node: ThoughtNode) -> Evaluation:
        """Returns the evaluation verdict for ``node``."""
        ...

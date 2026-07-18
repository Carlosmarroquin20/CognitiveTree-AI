"""Policy contracts decoupling search control from model-specific behavior.

The controller depends only on these protocols. Phase 2 supplies an evaluator
backed by sandboxed code execution; Phase 3 adds the critique, reward, and
revision contracts that drive semantic backtracking; LLM-backed generators
(Llama, Qwen) attach here without any change to the search core.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Protocol, runtime_checkable

from cognitivetree.node import ThoughtNode

CRITIQUE_METADATA_KEY = "critique"


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


@unique
class FailureClass(Enum):
    """Taxonomy of diagnosable failure modes for a rejected thought."""

    NO_EXECUTION = "no_execution"
    TIMEOUT = "timeout"
    SYNTAX = "syntax"
    ASSERTION = "assertion"
    EXCEPTION = "exception"


@dataclass(frozen=True, slots=True)
class Critique:
    """Structured diagnosis of a failed thought, oriented toward revision.

    Attributes:
        failure_class: Coarse classification of what went wrong.
        summary: One-line diagnosis of the observed failure.
        guidance: Actionable directive a generator can follow when producing
            revised candidates; phrased as an instruction, not a description.
        severity: Failure gravity in ``[0.0, 1.0]``; feeds reward shaping.
    """

    failure_class: FailureClass
    summary: str
    guidance: str
    severity: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must lie within [0.0, 1.0]")

    def to_dict(self) -> dict[str, Any]:
        """Serializes the critique for storage in node metadata."""
        return {
            "failure_class": self.failure_class.value,
            "summary": self.summary,
            "guidance": self.guidance,
            "severity": round(self.severity, 4),
        }


@runtime_checkable
class Critic(Protocol):
    """Diagnoses why a rejected thought failed and how to revise it."""

    def critique(self, node: ThoughtNode) -> Critique | None:
        """Returns a critique for ``node``, or ``None`` when there is nothing
        actionable to report."""
        ...


@runtime_checkable
class RevisionPolicy(Protocol):
    """Decides whether a saturated node earns another expansion attempt.

    The search controller consults the policy at the moment structural
    backtracking would otherwise prune a node whose children have all failed.
    A ``True`` return commits the revision: the policy is expected to have
    prepared the node (revision notes, attempt accounting) before returning.
    """

    def revise(self, node: ThoughtNode) -> bool:
        """Prepares ``node`` for re-expansion and reports whether to proceed."""
        ...


@runtime_checkable
class RewardModel(Protocol):
    """Shapes the value that backpropagates through the tree for one node.

    Shaping affects search guidance only; solution acceptance always operates
    on the raw evaluator score.
    """

    def shape(
        self, node: ThoughtNode, base_score: float, critique: Critique | None
    ) -> float:
        """Returns the shaped backpropagation value for ``node``."""
        ...

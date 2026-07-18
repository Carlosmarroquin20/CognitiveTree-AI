"""Composite reward shaping for backpropagated search values."""

from __future__ import annotations

from dataclasses import dataclass

from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Critique

REWARD_METADATA_KEY = "reward"


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Relative weights of the shaped-reward components.

    Attributes:
        evaluation: Weight of the raw evaluator score.
        critique: Weight of the critique term, computed as ``1 - severity``
            (a node with no critique contributes the full term).
        depth: Weight of the shallowness term, which decays linearly to zero
            at ``depth_horizon`` and biases search toward shorter chains.
        depth_horizon: Depth at which the shallowness term bottoms out.
    """

    evaluation: float = 0.7
    critique: float = 0.2
    depth: float = 0.1
    depth_horizon: int = 16

    def __post_init__(self) -> None:
        if min(self.evaluation, self.critique, self.depth) < 0.0:
            raise ValueError("reward weights must be non-negative")
        if self.evaluation + self.critique + self.depth <= 0.0:
            raise ValueError("at least one reward weight must be positive")
        if self.depth_horizon < 1:
            raise ValueError("depth_horizon must be a positive integer")


class RewardShaper:
    """Implements the :class:`~cognitivetree.policies.RewardModel` contract.

    The shaped value is the weighted mean of the evaluation, critique, and
    shallowness terms, normalized back into ``[0.0, 1.0]``. The component
    breakdown is stored under ``node.metadata["reward"]`` so every
    backpropagated value remains auditable after the run.
    """

    def __init__(self, weights: RewardWeights | None = None) -> None:
        self._weights = weights or RewardWeights()

    def shape(
        self, node: ThoughtNode, base_score: float, critique: Critique | None
    ) -> float:
        """Returns the shaped backpropagation value for ``node``."""
        w = self._weights
        critique_term = 1.0 if critique is None else 1.0 - critique.severity
        depth_term = max(0.0, 1.0 - node.depth / w.depth_horizon)
        total_weight = w.evaluation + w.critique + w.depth
        shaped = (
            w.evaluation * base_score
            + w.critique * critique_term
            + w.depth * depth_term
        ) / total_weight
        node.metadata[REWARD_METADATA_KEY] = {
            "base_score": round(base_score, 6),
            "critique_term": round(critique_term, 6),
            "depth_term": round(depth_term, 6),
            "shaped": round(shaped, 6),
        }
        return shaped

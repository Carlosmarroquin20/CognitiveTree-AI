"""Configuration surface for the search runtime."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Carries every tunable parameter that governs tree construction and traversal.

    Instances are immutable so a single configuration can be shared across the
    controller, tree, and policy layers without defensive copying.

    Attributes:
        max_iterations: Upper bound on select-expand-evaluate-backpropagate cycles.
        max_depth: Maximum node depth; nodes at this depth are never expanded.
        branching_factor: Number of candidate thoughts requested per expansion.
        exploration_weight: UCT exploration coefficient; higher values favor
            less-visited branches over exploitation of high-value ones.
        accept_threshold: Minimum evaluator score at which a terminal thought is
            accepted as a solution.
        prune_threshold: Score below which a thought is pruned from the frontier.
        seed: Seed for the controller RNG used in tie-breaking; reserved for
            stochastic policies in later phases. ``None`` yields nondeterminism.
    """

    max_iterations: int = 64
    max_depth: int = 8
    branching_factor: int = 3
    exploration_weight: float = 1.414
    accept_threshold: float = 0.95
    prune_threshold: float = 0.15
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if self.branching_factor < 1:
            raise ValueError("branching_factor must be a positive integer")
        if self.exploration_weight < 0.0:
            raise ValueError("exploration_weight must be non-negative")
        if not 0.0 <= self.prune_threshold < self.accept_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0.0 <= prune_threshold < accept_threshold <= 1.0"
            )

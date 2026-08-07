"""Configuration surface for the search runtime."""

from __future__ import annotations

from dataclasses import dataclass

# A serialized tree nests one JSON object per level, and ``json.dumps``
# encodes that nesting recursively: past roughly 496 levels it raises
# RecursionError, which would strand a finished run — unable to be archived,
# streamed to the UI, or replayed. The cap sits far below that cliff and far
# above any plausible reasoning chain (the default depth is 8), so the failure
# surfaces at construction with an actionable message instead of after the
# search has already spent its budget.
MAX_SUPPORTED_DEPTH = 256


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Carries every tunable parameter that governs tree construction and traversal.

    Instances are immutable so a single configuration can be shared across the
    controller, tree, and policy layers without defensive copying.

    Attributes:
        max_iterations: Upper bound on select-expand-evaluate-backpropagate cycles.
        max_depth: Maximum node depth; nodes at this depth are never expanded.
            Capped at :data:`MAX_SUPPORTED_DEPTH` so every reachable run stays
            serializable — see the note on that constant.
        branching_factor: Number of candidate thoughts requested per expansion.
        exploration_weight: UCT exploration coefficient; higher values favor
            less-visited branches over exploitation of high-value ones.
        accept_threshold: Minimum evaluator score at which a terminal thought is
            accepted as a solution.
        prune_threshold: Score below which a thought is pruned from the frontier.
        seed: Seed for the controller RNG used in tie-breaking; reserved for
            stochastic policies in later phases. ``None`` yields nondeterminism.
        max_wall_seconds: Global wall-clock budget for the entire run. ``None``
            (the default) disables the budget, matching every prior release.
            The controller checks the deadline once per iteration, at the
            boundary before expansion begins — the same granularity at which
            ``max_iterations`` already bounds work — so it cannot preempt an
            in-flight generator, critic, or sandboxed execution call; a single
            slow iteration can overshoot the deadline by its own duration.
            This budget is independent of, and typically tighter than, the
            per-call timeouts already enforced by the sandbox executor and the
            LLM client.
        evaluation_workers: Number of candidate evaluations to run
            concurrently within one expansion batch. ``1`` (the default) keeps
            evaluation strictly sequential, matching every prior release.
            Raising it parallelizes the dominant cost of a run — sandboxed
            execution — without affecting results: verdicts are always applied
            in candidate order, so a seeded run stays bit-for-bit
            reproducible. It is opt-in because a custom
            :class:`~cognitivetree.policies.ThoughtEvaluator` is not required
            to be thread-safe; the bundled sandbox evaluators are.
    """

    max_iterations: int = 64
    max_depth: int = 8
    branching_factor: int = 3
    exploration_weight: float = 1.414
    accept_threshold: float = 0.95
    prune_threshold: float = 0.15
    seed: int | None = None
    max_wall_seconds: float | None = None
    evaluation_workers: int = 1

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if self.max_depth < 1:
            raise ValueError("max_depth must be a positive integer")
        if self.max_depth > MAX_SUPPORTED_DEPTH:
            raise ValueError(
                f"max_depth must not exceed {MAX_SUPPORTED_DEPTH}; deeper trees "
                "cannot be serialized to JSON for archiving or streaming"
            )
        if self.branching_factor < 1:
            raise ValueError("branching_factor must be a positive integer")
        if self.exploration_weight < 0.0:
            raise ValueError("exploration_weight must be non-negative")
        if not 0.0 <= self.prune_threshold < self.accept_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0.0 <= prune_threshold < accept_threshold <= 1.0"
            )
        if self.max_wall_seconds is not None and self.max_wall_seconds <= 0.0:
            raise ValueError("max_wall_seconds must be positive when provided")
        if self.evaluation_workers < 1:
            raise ValueError("evaluation_workers must be a positive integer")

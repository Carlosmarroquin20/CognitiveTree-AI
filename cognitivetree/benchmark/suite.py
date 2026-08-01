"""Benchmark task contracts and the bundled reference suite."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from cognitivetree.config import SearchConfig
from cognitivetree.demo import SequencePuzzleGenerator
from cognitivetree.node import ThoughtNode
from cognitivetree.observability.metrics import TokenUsage
from cognitivetree.policies import Evaluation
from cognitivetree.search import TreeSearchController

VOCABULARY: tuple[str, ...] = ("north", "south", "east", "west")

TASK_STATEMENT = "Recover the hidden movement sequence."


@dataclass(frozen=True, slots=True)
class TaskSetup:
    """One task instantiated for a single run.

    ``usage`` is supplied by tasks whose policies consume tokens, letting the
    runner report consumption alongside search effort; synthetic tasks leave
    it unset.
    """

    controller: TreeSearchController
    usage: Callable[[], TokenUsage] | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """A single benchmark case.

    Attributes:
        name: Identifier reported in results.
        statement: Task text placed at the tree root.
        build: Constructs a fresh :class:`TaskSetup` from the effective
            config. A factory rather than a prebuilt controller, because each
            repeat needs its own tree, policies, and accounting.
        config_overrides: Fields forced onto the base config for this task —
            search depth and branching are intrinsic to a task, not to the
            configuration under evaluation.
        expected_solution: Known-correct answer. When set, a run counts as
            solved only if it both succeeded and returned this exact text,
            so an evaluator that wrongly accepts cannot inflate the score.
    """

    name: str
    statement: str
    build: Callable[[SearchConfig], TaskSetup]
    config_overrides: Mapping[str, Any] = field(default_factory=dict)
    expected_solution: str | None = None

    def resolve_config(self, base: SearchConfig) -> SearchConfig:
        """Applies this task's intrinsic overrides to ``base``."""
        return replace(base, **dict(self.config_overrides))

    def is_solved_by(self, outcome_succeeded: bool, solution: str | None) -> bool:
        """Reports whether a finished run counts as a solve."""
        if not outcome_succeeded:
            return False
        if self.expected_solution is None:
            return True
        return solution == self.expected_solution


@dataclass(frozen=True, slots=True)
class TerminalOnlyEvaluator:
    """Scores a sequence only once it reaches full length.

    Partial candidates are indistinguishable from one another, which is what
    makes the suite a real search problem: nothing can be pruned early, so the
    controller must actually explore rather than follow a gradient. The graded
    evaluator in :mod:`cognitivetree.demo` solves the same targets in one
    iteration per token and would leave every configuration tied at 100%.
    """

    target: tuple[str, ...]

    def evaluate(self, node: ThoughtNode) -> Evaluation:
        tokens = tuple(node.content.split())
        if len(tokens) < len(self.target):
            return Evaluation(score=0.5, rationale="incomplete sequence")
        if tokens == self.target:
            return Evaluation(score=1.0, is_terminal=True, rationale="exact match")
        return Evaluation(score=0.0, is_terminal=True, rationale="wrong sequence")


def sequence_task(target: tuple[str, ...]) -> BenchmarkTask:
    """Builds a hidden-sequence recovery task for ``target``."""

    def build(config: SearchConfig) -> TaskSetup:
        return TaskSetup(
            controller=TreeSearchController(
                config=config,
                generator=SequencePuzzleGenerator(vocabulary=VOCABULARY),
                evaluator=TerminalOnlyEvaluator(target=target),
            )
        )

    return BenchmarkTask(
        name=f"sequence-{len(target)}-{'-'.join(t[0] for t in target)}",
        statement=TASK_STATEMENT,
        build=build,
        # Depth is the sequence length; branching must cover the vocabulary or
        # some tokens would never be proposed.
        config_overrides={
            "max_depth": len(target),
            "branching_factor": len(VOCABULARY),
        },
        expected_solution=" ".join(target),
    )


DEFAULT_TARGETS: tuple[tuple[str, ...], ...] = (
    ("north", "east"),
    ("south", "west"),
    ("west", "north", "east"),
    ("east", "south", "south"),
    ("west", "north", "east", "south"),
    ("east", "east", "west", "north"),
)


def default_suite() -> tuple[BenchmarkTask, ...]:
    """Returns the bundled suite, graded from two to four tokens.

    The search space grows as ``4 ** length``, so the longer targets separate
    configurations that the short ones cannot.
    """
    return tuple(sequence_task(target) for target in DEFAULT_TARGETS)

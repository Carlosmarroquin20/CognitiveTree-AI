"""Deterministic reference domain exercising the full search loop.

The domain is a hidden-sequence recovery puzzle: the solver must reconstruct a
target token sequence by extending partial sequences one token at a time. The
evaluator scores prefix consistency, prunes divergent branches, and accepts an
exact reconstruction. The domain is intentionally trivial for an LLM but ideal
for validating search mechanics: it produces real branching, real pruning, and
real backtracking with zero model dependencies.

Run with: ``python -m cognitivetree.demo``
"""

from __future__ import annotations

from dataclasses import dataclass

from cognitivetree.config import SearchConfig
from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Evaluation
from cognitivetree.search import SearchEvent, TreeSearchController


@dataclass(frozen=True, slots=True)
class SequencePuzzleGenerator:
    """Proposes every vocabulary token as a continuation of the current sequence."""

    vocabulary: tuple[str, ...]

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        prefix = node.content if not node.is_root else ""
        candidates = [
            f"{prefix} {token}".strip() for token in self.vocabulary
        ]
        return candidates[:k]


@dataclass(frozen=True, slots=True)
class SequencePuzzleEvaluator:
    """Scores partial sequences by prefix consistency against the hidden target.

    Divergent sequences score 0.0 and fall below any sane pruning threshold.
    Consistent prefixes score proportionally to their coverage of the target,
    offset so they stay above the pruning threshold while incomplete. An exact
    reconstruction scores 1.0 and is flagged terminal.
    """

    target: tuple[str, ...]

    def evaluate(self, node: ThoughtNode) -> Evaluation:
        tokens = tuple(node.content.split())
        if len(tokens) > len(self.target) or tokens != self.target[: len(tokens)]:
            return Evaluation(score=0.0, rationale="diverged from target sequence")
        if tokens == self.target:
            return Evaluation(
                score=1.0, is_terminal=True, rationale="exact reconstruction"
            )
        coverage = len(tokens) / len(self.target)
        return Evaluation(
            score=0.2 + 0.7 * coverage,
            rationale=f"consistent prefix covering {coverage:.0%} of target",
        )


def main() -> None:
    """Runs the reference puzzle end-to-end and prints the search trace."""
    target = ("north", "east", "east", "south", "west")
    vocabulary = ("north", "south", "east", "west")

    config = SearchConfig(
        max_iterations=128,
        max_depth=len(target),
        branching_factor=len(vocabulary),
        exploration_weight=1.414,
        accept_threshold=0.95,
        prune_threshold=0.15,
        seed=7,
    )
    controller = TreeSearchController(
        config=config,
        generator=SequencePuzzleGenerator(vocabulary=vocabulary),
        evaluator=SequencePuzzleEvaluator(target=target),
        on_event=_print_event,
    )

    result = controller.run(task="Recover the hidden movement sequence.")

    print()
    print(result.tree.render())
    print()
    print(f"outcome     : {result.outcome.value}")
    print(f"iterations  : {result.iterations}")
    print(f"nodes       : {result.node_count}")
    print(f"transitions : {len(result.phase_history)}")
    print(f"solution    : {result.solution}")
    print(f"target      : {' '.join(target)}")


def _print_event(event: SearchEvent) -> None:
    node = event.node_id or "-"
    detail = f" | {event.detail}" if event.detail else ""
    print(f"[iter {event.iteration:03d}] {event.phase.value:<16} node={node}{detail}")


if __name__ == "__main__":
    main()

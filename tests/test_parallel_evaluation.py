"""Validates concurrent candidate evaluation and the invariants it must keep."""

import threading
import time

import pytest

from cognitivetree.config import SearchConfig
from cognitivetree.demo import SequencePuzzleEvaluator, SequencePuzzleGenerator
from cognitivetree.node import NodeStatus, ThoughtNode
from cognitivetree.policies import Evaluation
from cognitivetree.search import SearchOutcome, TreeSearchController

TARGET = ("north", "east", "east", "south")
VOCABULARY = ("north", "south", "east", "west")

BRANCHES = 4


class FixedGenerator:
    """Emits a fixed fan-out at the root and nothing deeper."""

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        if not node.is_root:
            return []
        return [f"candidate-{i}" for i in range(k)]


def strip_ids(payload):
    """Drops the per-run random node identifiers from a serialized tree.

    Node ids are ``uuid4``-based and therefore differ between any two runs,
    sequential or not; everything else about the tree must match.
    """
    if isinstance(payload, dict):
        return {k: strip_ids(v) for k, v in payload.items() if k != "id"}
    if isinstance(payload, list):
        return [strip_ids(item) for item in payload]
    return payload


def puzzle_controller(workers: int) -> TreeSearchController:
    return TreeSearchController(
        config=SearchConfig(
            max_iterations=128,
            max_depth=len(TARGET),
            branching_factor=len(VOCABULARY),
            seed=7,
            evaluation_workers=workers,
        ),
        generator=SequencePuzzleGenerator(vocabulary=VOCABULARY),
        evaluator=SequencePuzzleEvaluator(target=TARGET),
    )


class TestDeterminismParity:
    """Parallel evaluation must not alter what a seeded run produces."""

    @pytest.mark.parametrize("workers", [2, 4, 8])
    def test_results_match_the_sequential_run(self, workers: int) -> None:
        sequential = puzzle_controller(1).run("recover the sequence")
        parallel = puzzle_controller(workers).run("recover the sequence")

        assert parallel.outcome is sequential.outcome
        assert parallel.iterations == sequential.iterations
        assert parallel.node_count == sequential.node_count
        assert parallel.solution == sequential.solution
        assert [t.target for t in parallel.phase_history] == [
            t.target for t in sequential.phase_history
        ]

    def test_tree_is_structurally_identical(self) -> None:
        sequential = puzzle_controller(1).run("recover the sequence")
        parallel = puzzle_controller(4).run("recover the sequence")
        assert strip_ids(parallel.tree.to_dict(include_metadata=True)) == strip_ids(
            sequential.tree.to_dict(include_metadata=True)
        )

    def test_backpropagated_statistics_match(self) -> None:
        sequential = puzzle_controller(1).run("recover the sequence")
        parallel = puzzle_controller(4).run("recover the sequence")
        assert parallel.tree.root.visits == sequential.tree.root.visits
        assert parallel.tree.root.value_sum == pytest.approx(
            sequential.tree.root.value_sum
        )


class TestActualConcurrency:
    """Proof that the workers genuinely overlap, without timing heuristics."""

    def test_candidates_evaluate_simultaneously(self) -> None:
        # Every evaluation blocks on the barrier until all of them arrive.
        # Under sequential evaluation the first call would wait alone and the
        # barrier would break on timeout, so passing proves real overlap.
        barrier = threading.Barrier(BRANCHES, timeout=10.0)

        class RendezvousEvaluator:
            def evaluate(self, node: ThoughtNode) -> Evaluation:
                barrier.wait()
                return Evaluation(score=0.5)

        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=1,
                max_depth=1,
                branching_factor=BRANCHES,
                seed=1,
                evaluation_workers=BRANCHES,
            ),
            generator=FixedGenerator(),
            evaluator=RendezvousEvaluator(),
        )
        result = controller.run("task")

        assert result.node_count == BRANCHES + 1
        assert not barrier.broken

    def test_sequential_default_does_not_overlap(self) -> None:
        # The mirror image: with one worker the same barrier must break,
        # confirming the default path really is serial.
        barrier = threading.Barrier(BRANCHES, timeout=0.5)

        class RendezvousEvaluator:
            def evaluate(self, node: ThoughtNode) -> Evaluation:
                barrier.wait()
                return Evaluation(score=0.5)

        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=1, max_depth=1, branching_factor=BRANCHES, seed=1
            ),
            generator=FixedGenerator(),
            evaluator=RendezvousEvaluator(),
        )
        result = controller.run("task")

        assert barrier.broken
        assert result.outcome is SearchOutcome.FAILED


class TestOrderingGuarantees:
    """Verdicts apply in candidate order regardless of completion order."""

    def test_out_of_order_completion_still_applies_in_candidate_order(self) -> None:
        # The first candidate finishes last, inverting completion order.
        class InvertedLatencyEvaluator:
            def evaluate(self, node: ThoughtNode) -> Evaluation:
                index = int(node.content.rsplit("-", 1)[1])
                time.sleep((BRANCHES - index) * 0.02)
                return Evaluation(score=0.1 * (index + 1), rationale=node.content)

        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=1,
                max_depth=1,
                branching_factor=BRANCHES,
                seed=1,
                prune_threshold=0.0,
                evaluation_workers=BRANCHES,
            ),
            generator=FixedGenerator(),
            evaluator=InvertedLatencyEvaluator(),
        )
        result = controller.run("task")

        children = result.tree.root.children
        assert [c.content for c in children] == [
            f"candidate-{i}" for i in range(BRANCHES)
        ]
        assert [c.rationale for c in children] == [c.content for c in children]
        assert [round(c.score, 4) for c in children] == [
            round(0.1 * (i + 1), 4) for i in range(BRANCHES)
        ]

    def test_first_terminal_candidate_wins_the_solution(self) -> None:
        class TwoWinnersEvaluator:
            def evaluate(self, node: ThoughtNode) -> Evaluation:
                index = int(node.content.rsplit("-", 1)[1])
                if index in (1, 2):
                    # The later winner finishes first, which must not matter.
                    time.sleep(0.05 if index == 1 else 0.0)
                    return Evaluation(score=1.0, is_terminal=True)
                return Evaluation(score=0.5)

        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=1,
                max_depth=1,
                branching_factor=BRANCHES,
                seed=1,
                evaluation_workers=BRANCHES,
            ),
            generator=FixedGenerator(),
            evaluator=TwoWinnersEvaluator(),
        )
        result = controller.run("task")

        assert result.outcome is SearchOutcome.SUCCEEDED
        assert result.best_path[-1].content == "candidate-1"


class TestFailurePropagation:
    """Exceptions from workers surface deterministically."""

    def test_earliest_failing_candidate_reports_its_error(self) -> None:
        class SelectivelyExplodingEvaluator:
            def evaluate(self, node: ThoughtNode) -> Evaluation:
                index = int(node.content.rsplit("-", 1)[1])
                if index == 1:
                    time.sleep(0.05)  # fails later in wall-clock terms
                    raise RuntimeError("failure from candidate one")
                if index == 3:
                    raise RuntimeError("failure from candidate three")
                return Evaluation(score=0.5)

        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=1,
                max_depth=1,
                branching_factor=BRANCHES,
                seed=1,
                evaluation_workers=BRANCHES,
            ),
            generator=FixedGenerator(),
            evaluator=SelectivelyExplodingEvaluator(),
        )
        result = controller.run("task")

        assert result.outcome is SearchOutcome.FAILED
        assert "failure from candidate one" in result.error

    def test_worker_failure_does_not_escape_the_run(self) -> None:
        class ExplodingEvaluator:
            def evaluate(self, node: ThoughtNode) -> Evaluation:
                raise RuntimeError("evaluator down")

        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=4,
                max_depth=1,
                branching_factor=BRANCHES,
                seed=1,
                evaluation_workers=BRANCHES,
            ),
            generator=FixedGenerator(),
            evaluator=ExplodingEvaluator(),
        )
        result = controller.run("task")
        assert result.outcome is SearchOutcome.FAILED
        assert "evaluator down" in result.error


class TestConfiguration:
    """Worker-count validation and edge cases."""

    @pytest.mark.parametrize("workers", [0, -1])
    def test_non_positive_worker_counts_are_rejected(self, workers: int) -> None:
        with pytest.raises(ValueError, match="evaluation_workers"):
            SearchConfig(evaluation_workers=workers)

    def test_default_is_sequential(self) -> None:
        assert SearchConfig().evaluation_workers == 1

    def test_worker_count_above_batch_size_is_harmless(self) -> None:
        result = puzzle_controller(64).run("recover the sequence")
        assert result.outcome is SearchOutcome.SUCCEEDED

    def test_single_candidate_batch_uses_the_sequential_path(self) -> None:
        class SoleCandidateGenerator:
            def generate(self, node: ThoughtNode, k: int) -> list[str]:
                return ["only"] if node.is_root else []

        class RecordingEvaluator:
            def __init__(self) -> None:
                self.threads: set[str] = set()

            def evaluate(self, node: ThoughtNode) -> Evaluation:
                self.threads.add(threading.current_thread().name)
                return Evaluation(score=0.5)

        evaluator = RecordingEvaluator()
        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=1, max_depth=1, seed=1, evaluation_workers=8
            ),
            generator=SoleCandidateGenerator(),
            evaluator=evaluator,
        )
        controller.run("task")

        # A lone candidate must not be handed to a pool at all.
        assert evaluator.threads == {threading.current_thread().name}

    def test_pruned_children_keep_independent_metadata(self) -> None:
        class TaggingEvaluator:
            def evaluate(self, node: ThoughtNode) -> Evaluation:
                node.metadata["seen_by"] = node.content
                return Evaluation(score=0.0)

        controller = TreeSearchController(
            config=SearchConfig(
                max_iterations=1,
                max_depth=1,
                branching_factor=BRANCHES,
                seed=1,
                evaluation_workers=BRANCHES,
            ),
            generator=FixedGenerator(),
            evaluator=TaggingEvaluator(),
        )
        result = controller.run("task")

        for child in result.tree.root.children:
            assert child.metadata["seen_by"] == child.content
            assert child.status is NodeStatus.PRUNED

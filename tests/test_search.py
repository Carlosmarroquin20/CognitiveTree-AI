"""End-to-end validation of the search controller against deterministic policies."""


from cognitivetree.config import SearchConfig
from cognitivetree.demo import SequencePuzzleEvaluator, SequencePuzzleGenerator
from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Evaluation
from cognitivetree.search import SearchOutcome, TreeSearchController
from cognitivetree.state import SearchPhase

TARGET = ("north", "east", "east", "south")
VOCABULARY = ("north", "south", "east", "west")


def build_controller(**overrides) -> TreeSearchController:
    params = {
        "max_iterations": 128,
        "max_depth": len(TARGET),
        "branching_factor": len(VOCABULARY),
        "seed": 7,
    }
    params.update(overrides)
    return TreeSearchController(
        config=SearchConfig(**params),
        generator=SequencePuzzleGenerator(vocabulary=VOCABULARY),
        evaluator=SequencePuzzleEvaluator(target=TARGET),
    )


def test_controller_solves_reference_puzzle() -> None:
    result = build_controller().run("recover the sequence")

    assert result.outcome is SearchOutcome.SUCCEEDED
    assert result.solution == " ".join(TARGET)
    assert [node.depth for node in result.best_path] == list(range(len(TARGET) + 1))
    assert result.phase_history[-1].target is SearchPhase.SUCCEEDED


def test_search_is_deterministic_under_fixed_seed() -> None:
    first = build_controller(seed=11).run("recover the sequence")
    second = build_controller(seed=11).run("recover the sequence")

    assert first.iterations == second.iterations
    assert first.node_count == second.node_count
    assert [t.target for t in first.phase_history] == [
        t.target for t in second.phase_history
    ]


def test_depth_cap_exhausts_unreachable_target() -> None:
    result = build_controller(max_depth=2).run("recover the sequence")

    assert result.outcome is SearchOutcome.EXHAUSTED
    assert result.solution is None
    assert all(node.depth <= 2 for node in result.tree.nodes())


def test_iteration_budget_is_respected() -> None:
    result = build_controller(max_iterations=2).run("recover the sequence")

    assert result.outcome is SearchOutcome.EXHAUSTED
    assert result.iterations == 2


class ExplodingGenerator:
    """Simulates a policy fault to exercise the failure path."""

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        raise RuntimeError("backend unavailable")


def test_policy_fault_yields_failed_outcome() -> None:
    controller = TreeSearchController(
        config=SearchConfig(seed=1),
        generator=ExplodingGenerator(),
        evaluator=SequencePuzzleEvaluator(target=TARGET),
    )
    result = controller.run("recover the sequence")

    assert result.outcome is SearchOutcome.FAILED
    assert "backend unavailable" in result.error
    assert result.phase_history[-1].target is SearchPhase.FAILED


class DeadEndGenerator:
    """Returns no candidates so every expansion forces a backtrack."""

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        return []


def test_empty_expansion_backtracks_then_exhausts() -> None:
    controller = TreeSearchController(
        config=SearchConfig(seed=1),
        generator=DeadEndGenerator(),
        evaluator=SequencePuzzleEvaluator(target=TARGET),
    )
    result = controller.run("recover the sequence")

    assert result.outcome is SearchOutcome.EXHAUSTED
    phases = [t.target for t in result.phase_history]
    assert SearchPhase.BACKTRACKING in phases


class AlwaysWrongEvaluator:
    """Scores every thought below the pruning threshold."""

    def evaluate(self, node: ThoughtNode) -> Evaluation:
        return Evaluation(score=0.0, rationale="rejected")


def test_universal_pruning_collapses_tree_to_exhaustion() -> None:
    controller = TreeSearchController(
        config=SearchConfig(max_iterations=50, max_depth=4, branching_factor=2, seed=3),
        generator=SequencePuzzleGenerator(vocabulary=VOCABULARY),
        evaluator=AlwaysWrongEvaluator(),
    )
    result = controller.run("recover the sequence")

    assert result.outcome is SearchOutcome.EXHAUSTED
    # A single expansion prunes every child, saturating the root immediately.
    assert result.iterations < 5


def test_events_mirror_phase_history() -> None:
    events = []
    controller = TreeSearchController(
        config=SearchConfig(
            max_iterations=128,
            max_depth=len(TARGET),
            branching_factor=len(VOCABULARY),
            seed=7,
        ),
        generator=SequencePuzzleGenerator(vocabulary=VOCABULARY),
        evaluator=SequencePuzzleEvaluator(target=TARGET),
        on_event=events.append,
    )
    result = controller.run("recover the sequence")

    assert [e.phase for e in events] == [t.target for t in result.phase_history]


def test_duplicate_candidates_are_deduplicated() -> None:
    class RepeatingGenerator:
        def generate(self, node: ThoughtNode, k: int) -> list[str]:
            return ["same thought", "same thought", "  ", "other thought"]

    class NeutralEvaluator:
        def evaluate(self, node: ThoughtNode) -> Evaluation:
            return Evaluation(score=0.5)

    controller = TreeSearchController(
        config=SearchConfig(max_iterations=1, branching_factor=4, seed=1),
        generator=RepeatingGenerator(),
        evaluator=NeutralEvaluator(),
    )
    result = controller.run("task")

    assert [c.content for c in result.tree.root.children] == [
        "same thought",
        "other thought",
    ]

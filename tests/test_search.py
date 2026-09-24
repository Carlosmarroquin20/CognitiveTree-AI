"""End-to-end validation of the search controller against deterministic policies."""

import threading
import time

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


class SlowGenerator:
    """Sleeps past the configured wall-clock deadline on its first call only.

    A single sleep is enough: the deadline is checked once per iteration, so
    overshooting it during iteration 1 guarantees detection at the top of
    iteration 2, regardless of scheduling jitter on the test runner.
    """

    def __init__(self, delay_seconds: float) -> None:
        self._delay_seconds = delay_seconds
        self._called = False

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        if not self._called:
            self._called = True
            time.sleep(self._delay_seconds)
        return [f"{node.content} thought-{i}" for i in range(k)]


class NeverTerminalEvaluator:
    """Scores every thought mid-range so the search never reaches SUCCEEDED.

    Isolates the wall-clock budget as the sole possible cause of termination:
    nothing here can trigger EXHAUSTED via pruning or SUCCEEDED via acceptance.
    """

    def evaluate(self, node: ThoughtNode) -> Evaluation:
        return Evaluation(score=0.5)


def test_wall_clock_budget_stops_a_slow_run() -> None:
    controller = TreeSearchController(
        config=SearchConfig(
            max_iterations=50,
            max_depth=50,
            branching_factor=2,
            max_wall_seconds=0.02,
            seed=1,
        ),
        generator=SlowGenerator(delay_seconds=0.15),
        evaluator=NeverTerminalEvaluator(),
    )
    result = controller.run("task")

    assert result.outcome is SearchOutcome.TIMED_OUT
    assert result.iterations == 1
    assert result.phase_history[-1].target is SearchPhase.TIMED_OUT
    assert "wall-clock budget" in result.phase_history[-1].note


def test_generous_wall_clock_budget_does_not_interfere() -> None:
    result = build_controller(max_wall_seconds=60.0).run("recover the sequence")

    assert result.outcome is SearchOutcome.SUCCEEDED
    assert result.solution == " ".join(TARGET)


def test_wall_clock_budget_defaults_to_unbounded() -> None:
    assert SearchConfig().max_wall_seconds is None


class CancellingGenerator:
    """Sets the cancellation event while generating its ``trigger_call``-th batch."""

    def __init__(self, cancel_event: threading.Event, trigger_call: int) -> None:
        self._cancel_event = cancel_event
        self._trigger_call = trigger_call
        self.calls = 0

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        self.calls += 1
        if self.calls == self._trigger_call:
            self._cancel_event.set()
        return [f"{node.content} thought-{i}" for i in range(k)]


def cancellable_controller(
    generator: CancellingGenerator, **overrides: object
) -> TreeSearchController:
    params: dict[str, object] = {
        "max_iterations": 50,
        "max_depth": 50,
        "branching_factor": 2,
        "seed": 1,
    }
    params.update(overrides)
    return TreeSearchController(
        config=SearchConfig(**params),  # type: ignore[arg-type]
        generator=generator,
        evaluator=NeverTerminalEvaluator(),
    )


def test_cancellation_stops_the_run_at_the_next_iteration_boundary() -> None:
    cancel = threading.Event()
    generator = CancellingGenerator(cancel, trigger_call=3)
    result = cancellable_controller(generator).run("task", cancel_event=cancel)

    # The iteration that observed the signal completes; no later one starts.
    assert result.outcome is SearchOutcome.CANCELLED
    assert result.iterations == 3
    assert generator.calls == 3
    assert result.phase_history[-1].target is SearchPhase.CANCELLED
    assert "cancelled" in result.phase_history[-1].note
    assert result.solution is None


def test_pre_set_cancellation_does_no_work() -> None:
    cancel = threading.Event()
    cancel.set()
    generator = CancellingGenerator(cancel, trigger_call=0)
    result = cancellable_controller(generator).run("task", cancel_event=cancel)

    assert result.outcome is SearchOutcome.CANCELLED
    assert result.iterations == 0
    assert generator.calls == 0
    assert result.node_count == 1


def test_cancellation_takes_precedence_over_the_deadline() -> None:
    cancel = threading.Event()
    cancel.set()
    controller = cancellable_controller(
        CancellingGenerator(cancel, trigger_call=0), max_wall_seconds=1e-9
    )
    time.sleep(0.01)
    assert controller.run("task", cancel_event=cancel).outcome is SearchOutcome.CANCELLED


def test_unset_cancellation_event_leaves_the_run_unchanged() -> None:
    baseline = build_controller(seed=11).run("recover the sequence")
    observed = build_controller(seed=11).run(
        "recover the sequence", cancel_event=threading.Event()
    )
    assert observed.outcome is baseline.outcome
    assert observed.iterations == baseline.iterations
    assert [t.target for t in observed.phase_history] == [
        t.target for t in baseline.phase_history
    ]

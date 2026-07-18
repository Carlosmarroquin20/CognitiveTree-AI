"""End-to-end validation of the critique-driven backtracking loop."""

from dataclasses import dataclass

from cognitivetree.config import SearchConfig
from cognitivetree.feedback.demo import (
    BROKEN_WAVE,
    GuidanceSensitiveGenerator,
    REVISED_CANDIDATE,
)
from cognitivetree.feedback.execution_critic import ExecutionTraceCritic
from cognitivetree.feedback.revision import (
    REVISION_ATTEMPTS_KEY,
    REVISION_NOTES_KEY,
    BoundedRevisionPolicy,
)
from cognitivetree.feedback.rewards import REWARD_METADATA_KEY, RewardShaper
from cognitivetree.node import ThoughtNode
from cognitivetree.sandbox.demo import VALIDATION_HARNESS
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor
from cognitivetree.search import SearchOutcome, TreeSearchController
from cognitivetree.state import SearchPhase


@dataclass(frozen=True, slots=True)
class AlwaysBrokenGenerator:
    """Reproposes the same broken wave regardless of revision notes."""

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        if not node.is_root:
            return []
        return list(BROKEN_WAVE[:k])


def build_controller(generator, **hooks) -> TreeSearchController:
    return TreeSearchController(
        config=SearchConfig(
            max_iterations=8,
            max_depth=1,
            branching_factor=len(BROKEN_WAVE),
            seed=7,
        ),
        generator=generator,
        evaluator=CodeExecutionEvaluator(
            executor=SubprocessExecutor(),
            validation_harness=VALIDATION_HARNESS,
        ),
        **hooks,
    )


def test_fail_critique_revise_succeed_cycle() -> None:
    controller = build_controller(
        GuidanceSensitiveGenerator(),
        critic=ExecutionTraceCritic(),
        revision_policy=BoundedRevisionPolicy(max_attempts=1),
        reward_model=RewardShaper(),
    )
    result = controller.run("Implement clamp(value, low, high) correctly.")
    root = result.tree.root

    assert result.outcome is SearchOutcome.SUCCEEDED
    assert result.iterations == 2
    assert result.solution == REVISED_CANDIDATE
    assert root.metadata[REVISION_ATTEMPTS_KEY] == 1
    assert "Revise the implementation" in root.metadata[REVISION_NOTES_KEY]

    backtracks = [
        t for t in result.phase_history if t.target is SearchPhase.BACKTRACKING
    ]
    assert any("revision" in t.note for t in backtracks)

    first_wave = [c for c in root.children if c.content in BROKEN_WAVE]
    assert len(first_wave) == len(BROKEN_WAVE)
    assert all(c.metadata.get("critique") for c in first_wave)


def test_revision_budget_exhaustion_leads_to_exhausted_outcome() -> None:
    controller = build_controller(
        AlwaysBrokenGenerator(),
        critic=ExecutionTraceCritic(),
        revision_policy=BoundedRevisionPolicy(max_attempts=1),
    )
    result = controller.run("Implement clamp(value, low, high) correctly.")
    root = result.tree.root

    assert result.outcome is SearchOutcome.EXHAUSTED
    assert root.metadata[REVISION_ATTEMPTS_KEY] == 1
    # The revised expansion reproposed only known-failed candidates, which
    # deduplication reduced to an empty batch.
    assert len(root.children) == len(BROKEN_WAVE)


def test_without_hooks_behavior_matches_phase_one_pruning() -> None:
    controller = build_controller(AlwaysBrokenGenerator())
    result = controller.run("Implement clamp(value, low, high) correctly.")
    root = result.tree.root

    assert result.outcome is SearchOutcome.EXHAUSTED
    assert REVISION_ATTEMPTS_KEY not in root.metadata
    assert all("critique" not in c.metadata for c in root.children)


def test_reward_shaping_alters_backpropagated_values() -> None:
    shaped = build_controller(
        AlwaysBrokenGenerator(),
        critic=ExecutionTraceCritic(),
        reward_model=RewardShaper(),
    ).run("Implement clamp(value, low, high) correctly.")
    raw = build_controller(AlwaysBrokenGenerator()).run(
        "Implement clamp(value, low, high) correctly."
    )

    assert shaped.tree.root.visits == raw.tree.root.visits
    assert shaped.tree.root.value_sum > raw.tree.root.value_sum
    assert all(
        REWARD_METADATA_KEY in c.metadata for c in shaped.tree.root.children
    )

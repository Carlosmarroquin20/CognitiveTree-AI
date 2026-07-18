"""End-to-end demonstration of the critique-driven backtracking loop.

The scenario forces the full Phase 3 cycle: the generator's first wave of
candidates is entirely broken, every child fails its sandboxed validation and
is pruned, the critic distills the assertion failures into revision notes, the
revision policy reopens the root, and the generator — now reading the notes —
produces the corrected implementation, which is accepted.

The guidance-sensitive generator stands in for an LLM: it reacts to the same
revision notes an LLM prompt would receive in Phase 4, but deterministically.

Run with: ``python -m cognitivetree.feedback.demo``
"""

from __future__ import annotations

from dataclasses import dataclass

from cognitivetree.config import SearchConfig
from cognitivetree.feedback.execution_critic import ExecutionTraceCritic
from cognitivetree.feedback.revision import (
    REVISION_ATTEMPTS_KEY,
    REVISION_NOTES_KEY,
    BoundedRevisionPolicy,
)
from cognitivetree.feedback.rewards import RewardShaper
from cognitivetree.node import ThoughtNode
from cognitivetree.sandbox.backends import select_executor
from cognitivetree.sandbox.demo import VALIDATION_HARNESS
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.search import SearchEvent, TreeSearchController

BROKEN_WAVE: tuple[str, ...] = (
    "Clamp by capping at the upper bound only.\n"
    "```python\ndef clamp(value, low, high):\n    return min(value, high)\n```",
    "Clamp by always returning the lower bound.\n"
    "```python\ndef clamp(value, low, high):\n    return low\n```",
    "Clamp by comparing against the wrong bound first.\n"
    "```python\ndef clamp(value, low, high):\n    return min(low, max(value, high))\n```",
)

REVISED_CANDIDATE = (
    "Revised after critique: raise the value to the lower bound, then cap at "
    "the upper bound.\n"
    "```python\ndef clamp(value, low, high):\n    return max(low, min(value, high))\n```"
)


@dataclass(frozen=True, slots=True)
class GuidanceSensitiveGenerator:
    """Proposes broken candidates first and a corrected one once notes exist.

    The branch on revision notes mirrors how an LLM generator conditions on
    critique context; determinism keeps the demonstration reproducible.
    """

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        if not node.is_root:
            return []
        notes = str(node.metadata.get(REVISION_NOTES_KEY, ""))
        if not notes:
            return list(BROKEN_WAVE[:k])
        return [REVISED_CANDIDATE]


def main() -> None:
    """Runs the fail-critique-revise-succeed cycle and prints its trace."""
    executor, backend = select_executor()
    print(f"execution backend: {backend}")
    print()

    controller = TreeSearchController(
        config=SearchConfig(
            max_iterations=16,
            max_depth=1,
            branching_factor=len(BROKEN_WAVE),
            seed=7,
        ),
        generator=GuidanceSensitiveGenerator(),
        evaluator=CodeExecutionEvaluator(
            executor=executor,
            validation_harness=VALIDATION_HARNESS,
        ),
        critic=ExecutionTraceCritic(),
        revision_policy=BoundedRevisionPolicy(max_attempts=1),
        reward_model=RewardShaper(),
        on_event=_print_event,
    )

    result = controller.run(task="Implement clamp(value, low, high) correctly.")
    root = result.tree.root

    print()
    print(result.tree.render())
    print()
    print("critiques recorded on failed candidates:")
    for child in root.children:
        critique = child.metadata.get("critique")
        if critique:
            print(f"- [{critique['failure_class']}] {critique['summary']}")
    print()
    print("revision notes handed to the generator:")
    for line in str(root.metadata.get(REVISION_NOTES_KEY, "")).splitlines():
        print(f"  {line}")
    print()
    print(f"revision attempts : {root.metadata.get(REVISION_ATTEMPTS_KEY, 0)}")
    print(f"outcome           : {result.outcome.value}")
    print(f"iterations        : {result.iterations}")
    if result.solution:
        print()
        print(result.solution)


def _print_event(event: SearchEvent) -> None:
    node = event.node_id or "-"
    detail = f" | {event.detail}" if event.detail else ""
    print(f"[iter {event.iteration:03d}] {event.phase.value:<16} node={node}{detail}")


if __name__ == "__main__":
    main()

"""End-to-end demonstration: tree search grounded by sandboxed execution.

The task is to produce a correct ``clamp`` implementation. A deterministic
generator proposes four candidate implementations (three subtly broken); the
:class:`CodeExecutionEvaluator` executes each candidate against a hidden
validation harness, prunes the failures, and accepts the implementation whose
harness passes. The demo prefers the Docker backend and falls back to the
host-process executor when no daemon is reachable.

Run with: ``python -m cognitivetree.sandbox.demo``
"""

from __future__ import annotations

from dataclasses import dataclass

from cognitivetree.config import SearchConfig
from cognitivetree.node import ThoughtNode
from cognitivetree.sandbox.docker_executor import (
    DockerSandboxConfig,
    DockerSandboxExecutor,
    ensure_image,
)
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.sandbox.executor import CodeExecutor
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor
from cognitivetree.search import TreeSearchController

CANDIDATE_IMPLEMENTATIONS: tuple[str, ...] = (
    "Clamp by capping at the upper bound only.\n"
    "```python\ndef clamp(value, low, high):\n    return min(value, high)\n```",
    "Clamp by nesting min and max with the bounds swapped.\n"
    "```python\ndef clamp(value, low, high):\n    return min(low, max(value, high))\n```",
    "Clamp by raising the value to the lower bound, then capping at the upper bound.\n"
    "```python\ndef clamp(value, low, high):\n    return max(low, min(value, high))\n```",
    "Clamp by always returning the lower bound.\n"
    "```python\ndef clamp(value, low, high):\n    return low\n```",
)

VALIDATION_HARNESS = """
assert clamp(5, 0, 10) == 5, "in-range value must pass through"
assert clamp(-3, 0, 10) == 0, "below-range value must rise to low"
assert clamp(42, 0, 10) == 10, "above-range value must cap at high"
assert clamp(7, 7, 7) == 7, "degenerate range must be stable"
print("validation harness: all checks passed")
"""


@dataclass(frozen=True, slots=True)
class CandidateImplementationGenerator:
    """Proposes the fixed candidate set at the root and nothing deeper."""

    candidates: tuple[str, ...] = CANDIDATE_IMPLEMENTATIONS

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        if not node.is_root:
            return []
        return list(self.candidates[:k])


def select_executor() -> tuple[CodeExecutor, str]:
    """Returns the strongest available execution backend and its description."""
    if DockerSandboxExecutor.is_available():
        config = DockerSandboxConfig()
        ensure_image(config, build_if_missing=True)
        return DockerSandboxExecutor(config), f"docker ({config.image})"
    return (
        SubprocessExecutor(),
        "subprocess fallback (no isolation; Docker daemon unreachable)",
    )


def main() -> None:
    """Searches for a correct implementation using execution-grounded scoring."""
    executor, backend = select_executor()
    print(f"execution backend: {backend}")

    config = SearchConfig(
        max_iterations=16,
        max_depth=1,
        branching_factor=len(CANDIDATE_IMPLEMENTATIONS),
        seed=7,
    )
    controller = TreeSearchController(
        config=config,
        generator=CandidateImplementationGenerator(),
        evaluator=CodeExecutionEvaluator(
            executor=executor,
            validation_harness=VALIDATION_HARNESS,
        ),
    )

    result = controller.run(task="Implement clamp(value, low, high) correctly.")

    print()
    print(result.tree.render())
    print()
    for child in result.tree.root.children:
        record = child.metadata.get("execution", {})
        print(
            f"- node {child.id} exit={record.get('exit_code')} "
            f"status={record.get('status')} | {child.rationale}"
        )
    print()
    print(f"outcome  : {result.outcome.value}")
    print(f"solution : {'found' if result.solution else 'none'}")
    if result.solution:
        print()
        print(result.solution)


if __name__ == "__main__":
    main()

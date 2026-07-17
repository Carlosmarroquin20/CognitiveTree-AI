"""Full-stack check: tree search converging on execution-validated code."""

from cognitivetree.config import SearchConfig
from cognitivetree.sandbox.demo import (
    CANDIDATE_IMPLEMENTATIONS,
    VALIDATION_HARNESS,
    CandidateImplementationGenerator,
)
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor
from cognitivetree.search import SearchOutcome, TreeSearchController


def test_search_selects_the_only_correct_implementation() -> None:
    controller = TreeSearchController(
        config=SearchConfig(
            max_iterations=8,
            max_depth=1,
            branching_factor=len(CANDIDATE_IMPLEMENTATIONS),
            seed=7,
        ),
        generator=CandidateImplementationGenerator(),
        evaluator=CodeExecutionEvaluator(
            executor=SubprocessExecutor(),
            validation_harness=VALIDATION_HARNESS,
        ),
    )

    result = controller.run(task="Implement clamp(value, low, high) correctly.")

    assert result.outcome is SearchOutcome.SUCCEEDED
    assert result.solution is not None
    assert "max(low, min(value, high))" in result.solution

    solved = result.best_path[-1]
    execution = solved.metadata["execution"]
    assert execution["exit_code"] == 0
    assert "all checks passed" in execution["stdout"]

    failures = [c for c in result.tree.root.children if c is not solved]
    assert len(failures) == 3
    assert all(c.metadata["execution"]["exit_code"] != 0 for c in failures)

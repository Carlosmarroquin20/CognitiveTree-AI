"""Validates the bridge from execution verdicts to search evaluations."""

import pytest

from cognitivetree.node import ThoughtNode
from cognitivetree.sandbox.evaluation import METADATA_KEY, CodeExecutionEvaluator
from cognitivetree.sandbox.spec import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    SandboxError,
)


class ScriptedExecutor:
    """Returns a canned result while recording every submitted request."""

    def __init__(self, result: ExecutionResult) -> None:
        self.result = result
        self.requests: list[ExecutionRequest] = []

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        self.requests.append(request)
        return self.result


def completed(exit_code: int, stderr: str = "") -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.COMPLETED, exit_code=exit_code, stderr=stderr
    )


def node_with_code(code: str = "print('x')") -> ThoughtNode:
    return ThoughtNode(content=f"Reasoning step.\n```python\n{code}\n```")


def test_clean_execution_yields_terminal_success() -> None:
    executor = ScriptedExecutor(completed(0))
    verdict = CodeExecutionEvaluator(executor).evaluate(node_with_code())
    assert verdict.score == 1.0
    assert verdict.is_terminal


def test_success_can_be_non_terminal() -> None:
    executor = ScriptedExecutor(completed(0))
    evaluator = CodeExecutionEvaluator(executor, terminal_on_success=False)
    assert not evaluator.evaluate(node_with_code()).is_terminal


def test_failed_execution_yields_prunable_score_with_stderr_rationale() -> None:
    executor = ScriptedExecutor(
        completed(1, stderr="Traceback...\nAssertionError: bounds violated")
    )
    verdict = CodeExecutionEvaluator(executor).evaluate(node_with_code())
    assert verdict.score == pytest.approx(0.05)
    assert not verdict.is_terminal
    assert "AssertionError: bounds violated" in verdict.rationale
    assert "exit code 1" in verdict.rationale


def test_timeout_yields_zero_score() -> None:
    executor = ScriptedExecutor(
        ExecutionResult(
            status=ExecutionStatus.TIMEOUT, exit_code=None, detail="deadline hit"
        )
    )
    verdict = CodeExecutionEvaluator(executor).evaluate(node_with_code())
    assert verdict.score == 0.0
    assert verdict.rationale == "deadline hit"


def test_sandbox_fault_raises_instead_of_scoring() -> None:
    executor = ScriptedExecutor(
        ExecutionResult(
            status=ExecutionStatus.SANDBOX_ERROR,
            exit_code=125,
            detail="daemon unreachable",
        )
    )
    with pytest.raises(SandboxError):
        CodeExecutionEvaluator(executor).evaluate(node_with_code())


def test_narrative_thought_gets_neutral_score_without_execution() -> None:
    executor = ScriptedExecutor(completed(0))
    verdict = CodeExecutionEvaluator(executor).evaluate(
        ThoughtNode(content="Plan: derive invariants first, code later.")
    )
    assert verdict.score == pytest.approx(0.3)
    assert not verdict.is_terminal
    assert executor.requests == []


def test_execution_record_lands_in_node_metadata() -> None:
    executor = ScriptedExecutor(completed(0))
    node = node_with_code()
    CodeExecutionEvaluator(executor).evaluate(node)
    record = node.metadata[METADATA_KEY]
    assert record["status"] == "completed"
    assert record["exit_code"] == 0


def test_validation_harness_is_appended_to_payload() -> None:
    executor = ScriptedExecutor(completed(0))
    evaluator = CodeExecutionEvaluator(
        executor, validation_harness="assert result == 4"
    )
    evaluator.evaluate(node_with_code("result = 2 + 2"))
    submitted = executor.requests[0].code
    assert submitted.startswith("result = 2 + 2")
    assert submitted.rstrip().endswith("assert result == 4")


def test_inconsistent_scores_are_rejected() -> None:
    with pytest.raises(ValueError):
        CodeExecutionEvaluator(ScriptedExecutor(completed(0)), success_score=0.2,
                               failure_score=0.5)

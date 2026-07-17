"""Bridge from sandboxed execution verdicts to the search core's evaluator contract."""

from __future__ import annotations

from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Evaluation
from cognitivetree.sandbox.executor import CodeExecutor
from cognitivetree.sandbox.extraction import extract_python_payload
from cognitivetree.sandbox.spec import (
    ExecutionRequest,
    ExecutionStatus,
    SandboxError,
)

METADATA_KEY = "execution"


class CodeExecutionEvaluator:
    """Grades thought nodes by executing their embedded code payload.

    The evaluator extracts the first fenced Python block from the thought
    content, appends the configured validation harness, and submits the
    combined program to the executor. The full execution record lands in
    ``node.metadata["execution"]`` for downstream critique (Phase 3) and UI
    streaming (Phase 4). Infrastructure faults surface as
    :class:`SandboxError` so the search run terminates as FAILED instead of
    silently pruning healthy branches.
    """

    def __init__(
        self,
        executor: CodeExecutor,
        validation_harness: str = "",
        success_score: float = 1.0,
        failure_score: float = 0.05,
        timeout_score: float = 0.0,
        missing_payload_score: float = 0.3,
        terminal_on_success: bool = True,
    ) -> None:
        """Configures scoring behavior around the supplied executor.

        Args:
            executor: Backend that runs the combined payload.
            validation_harness: Code appended after the payload; expected to
                raise (or exit nonzero) when the payload is functionally
                wrong, turning correctness checks into exit-code signals.
            success_score: Score for a clean exit; kept at or above the
                controller's acceptance threshold so passing payloads solve
                the task when ``terminal_on_success`` holds.
            failure_score: Score for payloads that ran and failed; kept below
                typical pruning thresholds so broken branches collapse.
            timeout_score: Score for payloads killed at the deadline.
            missing_payload_score: Score for thoughts without a code block;
                kept mid-range so purely narrative planning steps survive as
                interior nodes without ever being accepted.
            terminal_on_success: Marks cleanly executing payloads as terminal
                solution candidates.
        """
        if not 0.0 <= failure_score <= success_score <= 1.0:
            raise ValueError("scores must satisfy 0 <= failure <= success <= 1")
        self._executor = executor
        self._harness = validation_harness.rstrip()
        self._success_score = success_score
        self._failure_score = failure_score
        self._timeout_score = timeout_score
        self._missing_payload_score = missing_payload_score
        self._terminal_on_success = terminal_on_success

    def evaluate(self, node: ThoughtNode) -> Evaluation:
        """Returns the execution-grounded verdict for ``node``."""
        payload = extract_python_payload(node.content)
        if payload is None:
            return Evaluation(
                score=self._missing_payload_score,
                rationale="no executable payload found",
            )

        program = payload if not self._harness else f"{payload}\n\n{self._harness}\n"
        result = self._executor.execute(ExecutionRequest(code=program))
        node.metadata[METADATA_KEY] = result.to_dict()

        if result.status is ExecutionStatus.SANDBOX_ERROR:
            raise SandboxError(
                f"sandbox infrastructure fault: {result.detail or result.stderr[:200]}"
            )
        if result.status is ExecutionStatus.TIMEOUT:
            return Evaluation(
                score=self._timeout_score,
                rationale=result.detail or "payload timed out",
            )
        if result.ok:
            return Evaluation(
                score=self._success_score,
                is_terminal=self._terminal_on_success,
                rationale="payload and validation harness executed cleanly",
            )
        return Evaluation(
            score=self._failure_score,
            rationale=_failure_rationale(result.exit_code, result.stderr),
        )


def _failure_rationale(exit_code: int | None, stderr: str) -> str:
    """Condenses a failed run into a single-line rationale."""
    last_line = ""
    for line in reversed(stderr.strip().splitlines()):
        if line.strip():
            last_line = line.strip()
            break
    suffix = f": {last_line}" if last_line else ""
    return f"payload failed with exit code {exit_code}{suffix}"

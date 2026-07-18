"""Validates failure classification and guidance synthesis from execution records."""

from cognitivetree.feedback.execution_critic import ExecutionTraceCritic
from cognitivetree.node import ThoughtNode
from cognitivetree.policies import FailureClass
from cognitivetree.sandbox.evaluation import METADATA_KEY


def node_with_record(record: dict | None) -> ThoughtNode:
    node = ThoughtNode(content="candidate thought")
    if record is not None:
        node.metadata[METADATA_KEY] = record
    return node


def completed_record(exit_code: int, stderr: str = "") -> dict:
    return {"status": "completed", "exit_code": exit_code, "stderr": stderr}


def test_missing_record_yields_no_execution_critique() -> None:
    critique = ExecutionTraceCritic().critique(node_with_record(None))
    assert critique is not None
    assert critique.failure_class is FailureClass.NO_EXECUTION
    assert "fenced" in critique.guidance


def test_clean_execution_yields_no_critique() -> None:
    assert ExecutionTraceCritic().critique(node_with_record(completed_record(0))) is None


def test_timeout_record_yields_timeout_critique() -> None:
    critique = ExecutionTraceCritic().critique(
        node_with_record({"status": "timeout", "exit_code": None, "stderr": ""})
    )
    assert critique is not None
    assert critique.failure_class is FailureClass.TIMEOUT
    assert critique.severity == 0.9


def test_assertion_message_passes_through_to_guidance() -> None:
    stderr = (
        "Traceback (most recent call last):\n"
        '  File "<string>", line 9, in <module>\n'
        "AssertionError: below-range value must rise to low\n"
    )
    critique = ExecutionTraceCritic().critique(
        node_with_record(completed_record(1, stderr))
    )
    assert critique is not None
    assert critique.failure_class is FailureClass.ASSERTION
    assert "below-range value must rise to low" in critique.guidance
    assert critique.guidance.startswith("Revise the implementation")


def test_bare_assertion_gets_generic_guidance() -> None:
    critique = ExecutionTraceCritic().critique(
        node_with_record(completed_record(1, "AssertionError"))
    )
    assert critique is not None
    assert critique.failure_class is FailureClass.ASSERTION
    assert "assertion" in critique.guidance.lower()


def test_syntax_error_is_classified_separately() -> None:
    stderr = '  File "<string>", line 1\nSyntaxError: invalid syntax\n'
    critique = ExecutionTraceCritic().critique(
        node_with_record(completed_record(1, stderr))
    )
    assert critique is not None
    assert critique.failure_class is FailureClass.SYNTAX
    assert critique.severity == 0.8


def test_runtime_exception_names_the_error() -> None:
    stderr = "Traceback...\nNameError: name 'clam' is not defined\n"
    critique = ExecutionTraceCritic().critique(
        node_with_record(completed_record(1, stderr))
    )
    assert critique is not None
    assert critique.failure_class is FailureClass.EXCEPTION
    assert "NameError" in critique.guidance


def test_silent_nonzero_exit_reports_exit_code() -> None:
    critique = ExecutionTraceCritic().critique(
        node_with_record(completed_record(7, ""))
    )
    assert critique is not None
    assert critique.failure_class is FailureClass.EXCEPTION
    assert "exit" in critique.summary.lower() and "7" in critique.summary

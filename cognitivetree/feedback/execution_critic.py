"""Deterministic critic that diagnoses failures from sandbox execution records."""

from __future__ import annotations

import re

from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Critique, FailureClass
from cognitivetree.sandbox.evaluation import METADATA_KEY as EXECUTION_KEY

_ERROR_LINE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception|Exit|Interrupt|Warning))(?::\s*(?P<message>.*))?$")


class ExecutionTraceCritic:
    """Turns execution records into revision-oriented critiques.

    The critic reads the record stored by
    :class:`~cognitivetree.sandbox.evaluation.CodeExecutionEvaluator` under
    ``node.metadata["execution"]`` and classifies the failure from the exit
    status and the final traceback line. Assertion messages authored as
    requirement statements (the validation-harness convention) pass through
    verbatim into guidance, giving the generator a precise repair directive.
    """

    def critique(self, node: ThoughtNode) -> Critique | None:
        """Diagnoses ``node``; returns ``None`` for clean executions."""
        record = node.metadata.get(EXECUTION_KEY)
        if record is None:
            return Critique(
                failure_class=FailureClass.NO_EXECUTION,
                summary="thought carries no executable payload",
                guidance=(
                    "Produce a complete implementation inside a fenced "
                    "```python``` block."
                ),
                severity=0.4,
            )
        if record.get("status") == "timeout":
            return Critique(
                failure_class=FailureClass.TIMEOUT,
                summary="payload exceeded its execution deadline",
                guidance=(
                    "Remove unbounded loops or reduce algorithmic complexity; "
                    "the payload must terminate within its time budget."
                ),
                severity=0.9,
            )
        if record.get("exit_code") == 0:
            return None
        return self._diagnose_stderr(
            str(record.get("stderr", "")), record.get("exit_code")
        )

    def _diagnose_stderr(self, stderr: str, exit_code: object) -> Critique:
        """Classifies a nonzero exit from the final meaningful stderr line."""
        line = _last_meaningful_line(stderr)
        match = _ERROR_LINE.match(line)

        if match is None:
            guidance = "Revise the implementation so the payload exits cleanly."
            if line:
                guidance = (
                    "Inspect the failure output and revise the implementation "
                    f"to exit cleanly: {line}"
                )
            return Critique(
                failure_class=FailureClass.EXCEPTION,
                summary=f"payload exited with code {exit_code}",
                guidance=guidance,
                severity=0.7,
            )

        name = match.group("name")
        message = (match.group("message") or "").strip()

        if name == "AssertionError":
            if message:
                summary = f"validation check failed: {message}"
                guidance = (
                    "Revise the implementation so that this requirement holds: "
                    f"{message}"
                )
            else:
                summary = "a validation check failed without a message"
                guidance = (
                    "Revise the implementation until every validation "
                    "assertion passes."
                )
            return Critique(
                failure_class=FailureClass.ASSERTION,
                summary=summary,
                guidance=guidance,
                severity=0.6,
            )

        if name in ("SyntaxError", "IndentationError", "TabError"):
            detail = f"{name}: {message}" if message else name
            return Critique(
                failure_class=FailureClass.SYNTAX,
                summary=f"payload failed to parse: {detail}",
                guidance="Repair the syntax so the payload parses before resubmitting.",
                severity=0.8,
            )

        detail = f"{name}: {message}" if message else name
        suffix = f" ({message})" if message else ""
        return Critique(
            failure_class=FailureClass.EXCEPTION,
            summary=f"payload raised {detail}",
            guidance=f"Handle or eliminate the {name} raised at runtime{suffix}.",
            severity=0.7,
        )


def _last_meaningful_line(stderr: str) -> str:
    """Returns the final non-blank line of a traceback, or an empty string."""
    for line in reversed(stderr.strip().splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""

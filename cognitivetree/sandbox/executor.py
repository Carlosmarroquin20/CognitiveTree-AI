"""Backend protocol through which the framework submits code for execution."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cognitivetree.sandbox.spec import ExecutionRequest, ExecutionResult


@runtime_checkable
class CodeExecutor(Protocol):
    """Executes an isolated payload and reports the observed outcome.

    Implementations translate payload-level failures (nonzero exits, timeouts,
    resource kills) into :class:`ExecutionResult` fields and reserve raised
    :class:`~cognitivetree.sandbox.spec.SandboxError` exclusively for
    infrastructure faults.
    """

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        """Runs ``request`` to completion and returns its result record."""
        ...

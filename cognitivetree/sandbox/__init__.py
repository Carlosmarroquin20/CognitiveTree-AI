"""Sandboxed execution layer for validating intermediate code artifacts.

The layer mirrors the Phase 1 policy architecture: :mod:`spec` defines the
execution contracts, :mod:`executor` the backend protocol, and the concrete
backends (:mod:`docker_executor`, :mod:`subprocess_executor`) remain
interchangeable behind it. :mod:`evaluation` bridges execution verdicts into
the search core's ``ThoughtEvaluator`` contract.
"""

from cognitivetree.sandbox.docker_executor import (
    DockerSandboxConfig,
    DockerSandboxExecutor,
    DockerUnavailableError,
)
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.sandbox.executor import CodeExecutor
from cognitivetree.sandbox.extraction import extract_python_payload
from cognitivetree.sandbox.spec import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    ResourceLimits,
    SandboxError,
)
from cognitivetree.sandbox.subprocess_executor import SubprocessExecutor

__all__ = [
    "CodeExecutionEvaluator",
    "CodeExecutor",
    "DockerSandboxConfig",
    "DockerSandboxExecutor",
    "DockerUnavailableError",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStatus",
    "ResourceLimits",
    "SandboxError",
    "SubprocessExecutor",
    "extract_python_payload",
]

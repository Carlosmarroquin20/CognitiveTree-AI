"""Critique, reward shaping, and revision policies for semantic backtracking.

The subpackage implements the Phase 3 contracts declared in
:mod:`cognitivetree.policies`: :class:`ExecutionTraceCritic` diagnoses failed
executions from sandbox records, :class:`RewardShaper` composes the value that
backpropagates through the tree, and :class:`BoundedRevisionPolicy` decides
when a saturated node earns a revised expansion instead of structural pruning.
Every implementation is deterministic; LLM-backed critics attach through the
same :class:`~cognitivetree.policies.Critic` contract in Phase 4.
"""

from cognitivetree.feedback.composite import ChainedCritic
from cognitivetree.feedback.execution_critic import ExecutionTraceCritic
from cognitivetree.feedback.revision import (
    REVISION_ATTEMPTS_KEY,
    REVISION_NOTES_KEY,
    BoundedRevisionPolicy,
    compile_revision_notes,
)
from cognitivetree.feedback.rewards import RewardShaper, RewardWeights

__all__ = [
    "BoundedRevisionPolicy",
    "ChainedCritic",
    "ExecutionTraceCritic",
    "REVISION_ATTEMPTS_KEY",
    "REVISION_NOTES_KEY",
    "RewardShaper",
    "RewardWeights",
    "compile_revision_notes",
]

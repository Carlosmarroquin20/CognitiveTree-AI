"""CognitiveTree-AI: autonomous reasoning through validated tree search.

The package is organized in strict layers. ``config``, ``state``, ``node``, and
``tree`` form the structural core; ``policies`` defines the contracts through
which model backends and execution sandboxes attach in later phases; ``search``
hosts the controller that drives the MCTS / Tree-of-Thoughts loop.
"""

from cognitivetree.config import SearchConfig
from cognitivetree.node import NodeStatus, ThoughtNode
from cognitivetree.observability import (
    AccountingLlmClient,
    RunMetrics,
    TokenBudget,
    TokenUsage,
)
from cognitivetree.persistence import (
    ArchiveFormatError,
    ReplaySession,
    RunArchive,
    load_run,
    save_run,
)
from cognitivetree.policies import (
    Critic,
    Critique,
    Evaluation,
    FailureClass,
    RevisionPolicy,
    RewardModel,
    StopCondition,
    ThoughtEvaluator,
    ThoughtGenerator,
)
from cognitivetree.search import (
    SearchEvent,
    SearchOutcome,
    SearchResult,
    TreeSearchController,
)
from cognitivetree.session import (
    LlmSessionSpec,
    ReasoningSession,
    build_llm_session,
    build_reference_session,
)
from cognitivetree.state import (
    InvalidTransitionError,
    PhaseTransition,
    SearchPhase,
    SearchStateMachine,
)
from cognitivetree.tree import ThoughtTree

__version__ = "0.1.0"

__all__ = [
    "AccountingLlmClient",
    "ArchiveFormatError",
    "Critic",
    "Critique",
    "Evaluation",
    "FailureClass",
    "InvalidTransitionError",
    "LlmSessionSpec",
    "NodeStatus",
    "PhaseTransition",
    "ReasoningSession",
    "ReplaySession",
    "RunArchive",
    "RunMetrics",
    "RevisionPolicy",
    "RewardModel",
    "SearchConfig",
    "SearchEvent",
    "SearchOutcome",
    "SearchPhase",
    "SearchResult",
    "SearchStateMachine",
    "StopCondition",
    "ThoughtEvaluator",
    "ThoughtGenerator",
    "ThoughtNode",
    "ThoughtTree",
    "TokenBudget",
    "TokenUsage",
    "TreeSearchController",
    "build_llm_session",
    "build_reference_session",
    "load_run",
    "save_run",
]

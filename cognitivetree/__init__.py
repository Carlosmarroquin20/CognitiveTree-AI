"""CognitiveTree-AI: autonomous reasoning through validated tree search.

The package is organized in strict layers. ``config``, ``state``, ``node``, and
``tree`` form the structural core; ``policies`` defines the contracts through
which model backends and execution sandboxes attach in later phases; ``search``
hosts the controller that drives the MCTS / Tree-of-Thoughts loop.
"""

from cognitivetree.config import SearchConfig
from cognitivetree.node import NodeStatus, ThoughtNode
from cognitivetree.policies import Evaluation, ThoughtEvaluator, ThoughtGenerator
from cognitivetree.search import (
    SearchEvent,
    SearchOutcome,
    SearchResult,
    TreeSearchController,
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
    "Evaluation",
    "InvalidTransitionError",
    "NodeStatus",
    "PhaseTransition",
    "SearchConfig",
    "SearchEvent",
    "SearchOutcome",
    "SearchPhase",
    "SearchResult",
    "SearchStateMachine",
    "ThoughtEvaluator",
    "ThoughtGenerator",
    "ThoughtNode",
    "ThoughtTree",
    "TreeSearchController",
]

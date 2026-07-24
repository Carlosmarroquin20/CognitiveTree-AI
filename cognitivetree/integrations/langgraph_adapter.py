"""LangGraph embedding of the native reasoning engine.

The native FSM-supervised MCTS controller remains the execution engine; the
adapter packages a complete reasoning run as a single LangGraph node so the
framework composes into larger LangGraph applications (agent pipelines,
checkpointed workflows) without re-hosting the search loop node-by-node.
Re-implementing selection or backpropagation as LangGraph nodes would add
graph-runtime overhead per phase while duplicating logic the state machine
already guards; embedding at run granularity keeps one source of truth.

Requires the optional dependency: ``pip install cognitivetree-ai[langgraph]``.
"""

from __future__ import annotations

from typing import TypedDict

from cognitivetree.session import ControllerFactory


class ReasoningState(TypedDict, total=False):
    """Graph state contract shared with surrounding LangGraph nodes."""

    task: str
    outcome: str
    solution: str | None
    iterations: int
    node_count: int
    error: str


def build_reasoning_graph(controller_factory: ControllerFactory):
    """Compiles a one-node LangGraph that executes a full reasoning run.

    The returned graph consumes ``{"task": ...}`` and populates the outcome
    fields of :class:`ReasoningState`. Raises ``RuntimeError`` when langgraph
    is not installed.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError(
            "langgraph is not installed; install the optional extra: "
            "pip install cognitivetree-ai[langgraph]"
        ) from exc

    def reason(state: ReasoningState) -> ReasoningState:
        result = controller_factory(None).run(state["task"])
        return {
            "outcome": result.outcome.value,
            "solution": result.solution,
            "iterations": result.iterations,
            "node_count": result.node_count,
            "error": result.error,
        }

    graph = StateGraph(ReasoningState)
    graph.add_node("reason", reason)
    graph.add_edge(START, "reason")
    graph.add_edge("reason", END)
    return graph.compile()

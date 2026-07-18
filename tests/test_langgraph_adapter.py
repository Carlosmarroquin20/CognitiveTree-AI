"""Validates the optional LangGraph embedding; skipped without langgraph."""

import pytest

from cognitivetree.config import SearchConfig
from cognitivetree.demo import SequencePuzzleEvaluator, SequencePuzzleGenerator
from cognitivetree.integrations.langgraph_adapter import build_reasoning_graph
from cognitivetree.search import TreeSearchController

TARGET = ("north", "east", "south")
VOCABULARY = ("north", "south", "east", "west")


def controller_factory(sink):
    return TreeSearchController(
        config=SearchConfig(
            max_iterations=64,
            max_depth=len(TARGET),
            branching_factor=len(VOCABULARY),
            seed=7,
        ),
        generator=SequencePuzzleGenerator(vocabulary=VOCABULARY),
        evaluator=SequencePuzzleEvaluator(target=TARGET),
        on_event=sink,
    )


def test_adapter_raises_actionable_error_without_langgraph(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("langgraph"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="langgraph is not installed"):
        build_reasoning_graph(controller_factory)


def test_compiled_graph_executes_full_run() -> None:
    pytest.importorskip("langgraph")
    graph = build_reasoning_graph(controller_factory)
    state = graph.invoke({"task": "Recover the hidden movement sequence."})
    assert state["outcome"] == "succeeded"
    assert state["solution"] == " ".join(TARGET)
    assert state["iterations"] > 0

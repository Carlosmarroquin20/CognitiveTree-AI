"""Shared fixtures for the test suite."""

import threading
import time

import pytest

from cognitivetree.config import SearchConfig
from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Evaluation
from cognitivetree.search import SearchEvent, TreeSearchController
from cognitivetree.session import EventSink
from cognitivetree.state import TERMINAL_PHASES, SearchPhase


class EndlessRun:
    """Wires a run that never settles on its own, recording how it ends.

    Every thought scores mid-range and the iteration budget is far beyond
    what the test waits for, so only cancellation can stop the run early.
    """

    def __init__(self) -> None:
        self.generated = 0
        self.terminal: SearchPhase | None = None
        self.settled = threading.Event()

    def generate(self, node: ThoughtNode, k: int) -> list[str]:
        self.generated += 1
        time.sleep(0.005)
        return [f"{node.content}.{i}" for i in range(k)]

    def evaluate(self, node: ThoughtNode) -> Evaluation:
        return Evaluation(score=0.5)

    def factory(self, sink: EventSink | None) -> TreeSearchController:
        def recording_sink(event: SearchEvent) -> None:
            if sink is not None:
                sink(event)
            if event.phase in TERMINAL_PHASES:
                self.terminal = event.phase
                self.settled.set()

        return TreeSearchController(
            config=SearchConfig(
                max_iterations=100_000, max_depth=200, branching_factor=2, seed=3
            ),
            generator=self,
            evaluator=self,
            on_event=recording_sink,
        )


@pytest.fixture
def endless_run() -> EndlessRun:
    return EndlessRun()

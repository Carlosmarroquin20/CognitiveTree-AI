"""Assembly and lifecycle management for complete reasoning runs.

A :class:`ReasoningSession` binds a task to a fully wired controller and
exposes the run either synchronously (:meth:`ReasoningSession.run`) or as an
ordered stream of JSON-compatible envelopes (:meth:`ReasoningSession.stream`)
consumed by the streaming UI and any other event subscriber.

Two assembly factories cover the supported deployments:
:func:`build_reference_session` wires the deterministic clamp scenario (no
model required, full critique loop), and :func:`build_llm_session` wires an
OpenAI-compatible backend serving an open-source model.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from cognitivetree.config import SearchConfig
from cognitivetree.feedback.composite import ChainedCritic
from cognitivetree.feedback.execution_critic import ExecutionTraceCritic
from cognitivetree.feedback.revision import BoundedRevisionPolicy
from cognitivetree.feedback.rewards import RewardShaper
from cognitivetree.llm.critic import LlmCritic
from cognitivetree.llm.generator import LlmThoughtGenerator
from cognitivetree.llm.openai_compatible import OpenAICompatibleClient
from cognitivetree.policies import Critic
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.search import SearchEvent, SearchResult, TreeSearchController
from cognitivetree.state import TERMINAL_PHASES, SearchPhase
from cognitivetree.ui.events import (
    phase_envelope,
    result_envelope,
    snapshot_envelope,
)

EventSink = Callable[[SearchEvent], None]
ControllerFactory = Callable[[Optional[EventSink]], TreeSearchController]

_SNAPSHOT_PHASES = frozenset({SearchPhase.BACKPROPAGATION}) | TERMINAL_PHASES


class ReasoningSession:
    """Owns one task run end-to-end.

    Each :meth:`run` or :meth:`stream` call builds a fresh controller through
    the injected factory, so a session object can be reused and concurrent
    streams never share mutable search state.
    """

    def __init__(self, task: str, controller_factory: ControllerFactory) -> None:
        if not task.strip():
            raise ValueError("task must be a non-empty statement")
        self._task = task
        self._factory = controller_factory

    @property
    def task(self) -> str:
        return self._task

    def run(self) -> SearchResult:
        """Executes the task synchronously without event streaming."""
        return self._factory(None).run(self._task)

    def stream(self) -> Iterator[dict]:
        """Executes the task on a worker thread, yielding envelopes in order.

        The stream carries one ``phase`` envelope per state transition,
        ``snapshot`` envelopes at backpropagation and terminal phases, and a
        closing ``result`` envelope. Envelope production happens on the worker
        thread; this generator only drains the queue, so consumers may block
        freely (as an SSE connection does) without stalling the search.
        """
        envelopes: queue.Queue[dict | None] = queue.Queue()
        controller: TreeSearchController | None = None

        def sink(event: SearchEvent) -> None:
            envelopes.put(phase_envelope(event))
            if event.phase in _SNAPSHOT_PHASES and controller is not None:
                tree = controller.active_tree
                if tree is not None:
                    envelopes.put(snapshot_envelope(tree))

        controller = self._factory(sink)

        def work() -> None:
            try:
                result = controller.run(self._task)
                envelopes.put(result_envelope(result))
            finally:
                envelopes.put(None)

        worker = threading.Thread(
            target=work, name="cognitivetree-session", daemon=True
        )
        worker.start()
        while True:
            envelope = envelopes.get()
            if envelope is None:
                break
            yield envelope
        worker.join(timeout=10.0)


def build_reference_session() -> ReasoningSession:
    """Wires the deterministic reference scenario end-to-end.

    The scenario reuses the guidance-sensitive clamp generator, sandboxed
    validation, the execution-trace critic, and bounded revision, exercising
    every Phase 1-3 mechanism without any model dependency. It exists for
    demonstrations, UI development, and smoke verification.
    """
    from cognitivetree.feedback.demo import GuidanceSensitiveGenerator
    from cognitivetree.sandbox.backends import select_executor
    from cognitivetree.sandbox.demo import VALIDATION_HARNESS

    executor, _ = select_executor()

    def factory(sink: Optional[EventSink]) -> TreeSearchController:
        return TreeSearchController(
            config=SearchConfig(
                max_iterations=16, max_depth=1, branching_factor=3, seed=7
            ),
            generator=GuidanceSensitiveGenerator(),
            evaluator=CodeExecutionEvaluator(
                executor=executor, validation_harness=VALIDATION_HARNESS
            ),
            critic=ExecutionTraceCritic(),
            revision_policy=BoundedRevisionPolicy(max_attempts=1),
            reward_model=RewardShaper(),
            on_event=sink,
        )

    return ReasoningSession(
        task="Implement clamp(value, low, high) correctly.",
        controller_factory=factory,
    )


@dataclass(frozen=True, slots=True)
class LlmSessionSpec:
    """Deployment parameters for an LLM-backed reasoning session.

    Attributes:
        task: Task statement placed at the tree root.
        base_url: OpenAI-compatible endpoint root, e.g.
            ``http://localhost:11434/v1`` (Ollama) or
            ``http://localhost:8000/v1`` (vLLM).
        model: Served model identifier, e.g. ``llama3.3`` or
            ``Qwen/Qwen2.5-Coder-32B-Instruct``.
        validation_harness: Assertions appended to every extracted payload.
        api_key: Bearer token when the endpoint requires one.
        temperature: Sampling temperature for the generator.
        use_llm_critic: Chains an LLM critic behind the execution-trace
            critic for failures the traceback cannot explain.
        config: Search parameters for the run.
        revision_attempts: Revision budget per saturated node.
    """

    task: str
    base_url: str
    model: str
    validation_harness: str = ""
    api_key: str | None = None
    temperature: float = 0.7
    use_llm_critic: bool = False
    config: SearchConfig = SearchConfig(seed=None)
    revision_attempts: int = 1


def build_llm_session(spec: LlmSessionSpec) -> ReasoningSession:
    """Wires a session around an OpenAI-compatible model endpoint."""
    from cognitivetree.sandbox.backends import select_executor

    client = OpenAICompatibleClient(
        base_url=spec.base_url, model=spec.model, api_key=spec.api_key
    )
    executor, _ = select_executor()

    critic: Critic = ExecutionTraceCritic()
    if spec.use_llm_critic:
        critic = ChainedCritic([ExecutionTraceCritic(), LlmCritic(client)])

    def factory(sink: Optional[EventSink]) -> TreeSearchController:
        return TreeSearchController(
            config=spec.config,
            generator=LlmThoughtGenerator(client, temperature=spec.temperature),
            evaluator=CodeExecutionEvaluator(
                executor=executor, validation_harness=spec.validation_harness
            ),
            critic=critic,
            revision_policy=BoundedRevisionPolicy(max_attempts=spec.revision_attempts),
            reward_model=RewardShaper(),
            on_event=sink,
        )

    return ReasoningSession(task=spec.task, controller_factory=factory)

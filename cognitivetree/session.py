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

import logging
import queue
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cognitivetree.config import SearchConfig
from cognitivetree.feedback.composite import ChainedCritic
from cognitivetree.feedback.execution_critic import ExecutionTraceCritic
from cognitivetree.feedback.revision import BoundedRevisionPolicy
from cognitivetree.feedback.rewards import RewardShaper
from cognitivetree.llm.caching import CachingLlmClient, CompletionCache
from cognitivetree.llm.client import LlmClient
from cognitivetree.llm.critic import LlmCritic
from cognitivetree.llm.generator import LlmThoughtGenerator
from cognitivetree.llm.openai_compatible import OpenAICompatibleClient
from cognitivetree.observability.accounting import AccountingLlmClient
from cognitivetree.observability.budget import TokenBudget
from cognitivetree.observability.metrics import RunMetrics, TokenUsage
from cognitivetree.persistence.archive import save_run
from cognitivetree.policies import Critic
from cognitivetree.sandbox.evaluation import CodeExecutionEvaluator
from cognitivetree.search import (
    SearchEvent,
    SearchOutcome,
    SearchResult,
    TreeSearchController,
)
from cognitivetree.state import TERMINAL_PHASES, SearchPhase
from cognitivetree.ui.events import (
    metrics_envelope,
    phase_envelope,
    result_envelope,
    snapshot_envelope,
)

EventSink = Callable[[SearchEvent], None]
ControllerFactory = Callable[[EventSink | None], TreeSearchController]
UsageSource = Callable[[], TokenUsage]

_SNAPSHOT_PHASES = frozenset({SearchPhase.BACKPROPAGATION}) | TERMINAL_PHASES

logger = logging.getLogger(__name__)


class ReasoningSession:
    """Owns one task run end-to-end.

    Each :meth:`run` or :meth:`stream` call builds a fresh controller through
    the injected factory, so a session object can be reused and concurrent
    streams never share mutable search state.

    With ``archive_dir`` set, every finished run, cancelled and failed ones
    included, is saved there as a run archive that the replay backend can
    reopen. A run that could not be archived still returns its result: the
    write failure is logged rather than raised, since losing the archive is
    better than losing the run.

    ``usage`` reports the cumulative token tally of the session's client.
    Each run's metrics carry the difference across that run, so streamed
    metrics and archives include token consumption. Runs executing
    concurrently on one session share the tally, which blurs the split
    between them; the server avoids this by building a session per
    connection.
    """

    def __init__(
        self,
        task: str,
        controller_factory: ControllerFactory,
        archive_dir: str | Path | None = None,
        usage: UsageSource | None = None,
    ) -> None:
        if not task.strip():
            raise ValueError("task must be a non-empty statement")
        self._task = task
        self._factory = controller_factory
        self._archive_dir = Path(archive_dir) if archive_dir is not None else None
        self._usage = usage

    @property
    def task(self) -> str:
        return self._task

    def run(self) -> SearchResult:
        """Executes the task synchronously without event streaming."""
        baseline = self._usage() if self._usage is not None else None
        result = self._factory(None).run(self._task)
        self._settle(result, self._metrics(result, baseline))
        return result

    def stream(self) -> Iterator[dict[str, Any]]:
        """Executes the task on a worker thread, yielding envelopes in order.

        The stream carries one ``phase`` envelope per state transition,
        ``snapshot`` envelopes at backpropagation and terminal phases, and a
        closing ``result`` envelope. Envelope production happens on the worker
        thread; this generator only drains the queue, so consumers may block
        freely (as an SSE connection does) without stalling the search.

        Closing the generator before the ``result`` envelope cancels the run:
        a consumer that stops reading has no use for the rest of it, and
        letting it continue would keep spending model quota and sandbox time.
        The run stops at its next iteration boundary, in the ``cancelled``
        phase, without this generator waiting for it.
        """
        envelopes: queue.Queue[dict[str, Any] | None] = queue.Queue()
        cancel = threading.Event()
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
                baseline = self._usage() if self._usage is not None else None
                result = controller.run(self._task, cancel_event=cancel)
                metrics = self._metrics(result, baseline)
                self._settle(result, metrics)
                envelopes.put(metrics_envelope(metrics.to_dict()))
                envelopes.put(result_envelope(result))
            finally:
                envelopes.put(None)

        worker = threading.Thread(
            target=work, name="cognitivetree-session", daemon=True
        )
        worker.start()
        finished = False
        try:
            while True:
                envelope = envelopes.get()
                if envelope is None:
                    finished = True
                    break
                yield envelope
        finally:
            if not finished:
                cancel.set()
        worker.join(timeout=10.0)

    def _metrics(self, result: SearchResult, baseline: TokenUsage | None) -> RunMetrics:
        """Summarizes ``result`` with the tokens consumed since ``baseline``."""
        spent = (
            self._usage() - baseline
            if self._usage is not None and baseline is not None
            else None
        )
        return RunMetrics.from_result(result, token_usage=spent)

    def _settle(self, result: SearchResult, metrics: RunMetrics) -> None:
        """Archives a finished run and reports it as one structured log event.

        The event is the operator's record of the run: with JSON logging its
        fields are queryable without opening the archive. Failed runs log at
        warning level, since they signal a broken backend rather than a hard
        task.
        """
        archive = self._archive(result, metrics)
        usage = metrics.token_usage
        logger.log(
            logging.WARNING if result.outcome is SearchOutcome.FAILED else logging.INFO,
            "run %s after %d iterations",
            result.outcome.value,
            result.iterations,
            extra={
                "event": "run_finished",
                "outcome": result.outcome.value,
                "iterations": result.iterations,
                "nodes": result.node_count,
                "wall_seconds": round(metrics.wall_time_seconds, 3),
                "tokens": usage.total_tokens if usage is not None else None,
                "llm_calls": usage.calls if usage is not None else None,
                "archive": str(archive) if archive is not None else None,
                "error": result.error or None,
            },
        )

    def _archive(self, result: SearchResult, metrics: RunMetrics) -> Path | None:
        """Saves ``result`` under the archive directory, when one is set.

        Names sort chronologically and carry the outcome, so a directory
        listing reads as a run log; the random suffix keeps concurrent runs
        finishing in the same second from overwriting each other.
        """
        if self._archive_dir is None:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = f"{stamp}-{result.outcome.value}-{uuid.uuid4().hex[:6]}.json"
        try:
            return save_run(result, self._archive_dir / name, metrics=metrics.to_dict())
        except OSError as exc:
            logger.warning("run archive could not be written: %s", exc)
            return None


def build_reference_session(
    max_wall_seconds: float | None = None,
    evaluation_workers: int = 1,
    archive_dir: str | Path | None = None,
) -> ReasoningSession:
    """Wires the deterministic reference scenario end-to-end.

    The scenario reuses the guidance-sensitive clamp generator, sandboxed
    validation, the execution-trace critic, and bounded revision, exercising
    every Phase 1-3 mechanism without any model dependency. It exists for
    demonstrations, UI development, and smoke verification.

    ``max_wall_seconds`` threads through to the run's global time budget;
    ``None`` (the default) leaves the search unbounded, as before.
    ``archive_dir`` saves every run there; see :class:`ReasoningSession`.
    """
    from cognitivetree.feedback.demo import GuidanceSensitiveGenerator
    from cognitivetree.sandbox.backends import select_executor
    from cognitivetree.sandbox.demo import VALIDATION_HARNESS

    executor, _ = select_executor()

    def factory(sink: EventSink | None) -> TreeSearchController:
        return TreeSearchController(
            config=SearchConfig(
                max_iterations=16,
                max_depth=1,
                branching_factor=3,
                seed=7,
                max_wall_seconds=max_wall_seconds,
                evaluation_workers=evaluation_workers,
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
        archive_dir=archive_dir,
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
        critic_temperature: Sampling temperature for the LLM critic. Both
            default to the values the policies were tuned with; ``0`` makes
            a role deterministic, which reproducible runs and completion
            caching both need.
        use_llm_critic: Chains an LLM critic behind the execution-trace
            critic for failures the traceback cannot explain.
        config: Search parameters for the run.
        revision_attempts: Revision budget per saturated node.
        max_tokens: Total-token ceiling for the run; the search stops with
            outcome ``budget_exhausted`` once crossed. ``None`` leaves
            consumption unbounded.
        max_llm_calls: Completion-count ceiling, applied on the same terms.
        cache_completions: Replays completions for repeated identical
            requests. Only temperature-0 requests are cached, so this takes
            effect only for a role configured with temperature ``0``.
    """

    task: str
    base_url: str
    model: str
    validation_harness: str = ""
    api_key: str | None = None
    temperature: float = 0.7
    critic_temperature: float = 0.2
    use_llm_critic: bool = False
    config: SearchConfig = SearchConfig(seed=None)
    revision_attempts: int = 1
    max_tokens: int | None = None
    max_llm_calls: int | None = None
    cache_completions: bool = False

    def __post_init__(self) -> None:
        # Validated here so a bad value fails at startup; the completion
        # request would otherwise reject it on the first call, mid-run.
        for name in ("temperature", "critic_temperature"):
            if not 0.0 <= getattr(self, name) <= 2.0:
                raise ValueError(f"{name} must lie within [0.0, 2.0]")


def build_llm_session(
    spec: LlmSessionSpec,
    client: LlmClient | None = None,
    completion_cache: CompletionCache | None = None,
    archive_dir: str | Path | None = None,
) -> ReasoningSession:
    """Wires a session around a chat-completion backend.

    ``client`` overrides the endpoint constructed from the spec, which lets a
    deterministic double (see :class:`~cognitivetree.llm.scripted.ScriptedLlmClient`)
    drive the full assembly offline; ``spec.base_url`` and ``spec.model`` are
    ignored in that case.

    The client is always wrapped for accounting, which feeds each run's token
    usage into its metrics and, when the spec sets a consumption ceiling,
    becomes the run's stop condition. An already-accounting client is reused
    rather than double-wrapped, so its caller keeps a handle on the same
    totals the budget enforces. Each run receives a fresh budget, so the
    ceiling applies per run even though the client's totals keep
    accumulating across a reused session.

    With ``spec.cache_completions`` the cache sits outside the accounting
    layer, so budgets count only real backend calls and tokens.
    ``completion_cache`` supplies a store to share across sessions; without
    one, the session gets a private store that persists across its runs.
    ``archive_dir`` saves every run there; see :class:`ReasoningSession`.
    """
    from cognitivetree.sandbox.backends import select_executor

    if client is None:
        client = OpenAICompatibleClient(
            base_url=spec.base_url, model=spec.model, api_key=spec.api_key
        )

    # Always accounted: the wrapper is transparent and cheap, and it is what
    # puts token usage into streamed metrics and archives.
    accounting = client if isinstance(client, AccountingLlmClient) else AccountingLlmClient(client)
    client = accounting
    budgeted = spec.max_tokens is not None or spec.max_llm_calls is not None
    if spec.cache_completions:
        client = CachingLlmClient(client, completion_cache)

    executor, _ = select_executor()

    critic: Critic = ExecutionTraceCritic()
    if spec.use_llm_critic:
        critic = ChainedCritic(
            [
                ExecutionTraceCritic(),
                LlmCritic(client, temperature=spec.critic_temperature),
            ]
        )

    def factory(sink: EventSink | None) -> TreeSearchController:
        stop_condition = (
            TokenBudget(
                accounting,
                max_total_tokens=spec.max_tokens,
                max_calls=spec.max_llm_calls,
            )
            if budgeted
            else None
        )
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
            stop_condition=stop_condition,
        )

    return ReasoningSession(
        task=spec.task,
        controller_factory=factory,
        archive_dir=archive_dir,
        usage=lambda: accounting.usage,
    )

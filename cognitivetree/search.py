"""MCTS-driven Tree-of-Thoughts search controller.

The controller executes the canonical four-step MCTS cycle — selection,
expansion, evaluation, backpropagation — under the supervision of the
:class:`~cognitivetree.state.SearchStateMachine`. Backtracking operates on
two levels: structurally, a node whose children have all failed is pruned so
effort returns to the nearest viable ancestor; semantically, an optional
:class:`~cognitivetree.policies.RevisionPolicy` can intercept that pruning
and reopen the node for re-expansion with critique-derived revision notes,
turning failure diagnoses into amended guidance for the generator.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, unique

from cognitivetree.config import SearchConfig
from cognitivetree.node import NodeStatus, ThoughtNode
from cognitivetree.policies import (
    CRITIQUE_METADATA_KEY,
    Critic,
    Critique,
    Evaluation,
    RevisionPolicy,
    RewardModel,
    StopCondition,
    ThoughtEvaluator,
    ThoughtGenerator,
)
from cognitivetree.state import PhaseTransition, SearchPhase, SearchStateMachine
from cognitivetree.tree import ThoughtTree


@unique
class SearchOutcome(Enum):
    """Final disposition of a search run."""

    SUCCEEDED = "succeeded"
    EXHAUSTED = "exhausted"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class SearchEvent:
    """Point-in-time notification emitted as the controller changes phase."""

    iteration: int
    phase: SearchPhase
    node_id: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Aggregated outcome of a completed search run."""

    outcome: SearchOutcome
    best_path: tuple[ThoughtNode, ...]
    iterations: int
    node_count: int
    phase_history: tuple[PhaseTransition, ...]
    tree: ThoughtTree
    error: str = ""

    @property
    def solution(self) -> str | None:
        """Returns the accepted solution content, or ``None`` when unsolved."""
        if self.outcome is not SearchOutcome.SUCCEEDED or not self.best_path:
            return None
        return self.best_path[-1].content


class TreeSearchController:
    """Coordinates selection, expansion, evaluation, and backpropagation.

    The controller is stateless across runs: :meth:`run` builds a fresh tree
    and state machine per invocation, so one configured instance can serve
    many tasks sequentially.
    """

    def __init__(
        self,
        config: SearchConfig,
        generator: ThoughtGenerator,
        evaluator: ThoughtEvaluator,
        on_event: Callable[[SearchEvent], None] | None = None,
        critic: Critic | None = None,
        revision_policy: RevisionPolicy | None = None,
        reward_model: RewardModel | None = None,
        stop_condition: StopCondition | None = None,
    ) -> None:
        """Wires the search loop to its policies.

        ``critic``, ``revision_policy``, ``reward_model``, and
        ``stop_condition`` are optional; when omitted the controller
        reproduces the plain Phase 1 behavior of structural pruning and
        raw-score backpropagation.
        """
        self._config = config
        self._generator = generator
        self._evaluator = evaluator
        self._on_event = on_event
        self._critic = critic
        self._revision_policy = revision_policy
        self._reward_model = reward_model
        self._stop_condition = stop_condition
        self._rng = random.Random(config.seed)
        self._active_tree: ThoughtTree | None = None

    @property
    def active_tree(self) -> ThoughtTree | None:
        """Returns the tree of the in-progress or most recent run.

        Event sinks read this property to snapshot live search state; the
        reference is assigned before the first transition of a run, and sinks
        execute synchronously on the run's thread, so no locking is required.
        """
        return self._active_tree

    def run(
        self, task: str, cancel_event: threading.Event | None = None
    ) -> SearchResult:
        """Executes the search loop for ``task`` until a terminal phase is reached.

        Every early-stop mechanism is evaluated once per iteration, before
        that iteration's expansion begins, so none interrupts a policy call
        mid-flight: a set ``cancel_event`` stops the run in the ``CANCELLED``
        phase, ``config.max_wall_seconds`` stops it in ``TIMED_OUT``, and a
        satisfied ``stop_condition`` stops it in ``BUDGET_EXHAUSTED``. They are
        tested in that order, so a run that crosses several limits in the same
        iteration reports the first. Cancellation leads because once the
        caller has stopped waiting, no other limit matters.
        """
        tree = ThoughtTree(task)
        self._active_tree = tree
        machine = SearchStateMachine()
        iteration = 0
        error = ""
        deadline = (
            time.monotonic() + self._config.max_wall_seconds
            if self._config.max_wall_seconds is not None
            else None
        )

        self._advance(machine, SearchPhase.SELECTION, iteration, None, "search started")
        try:
            while iteration < self._config.max_iterations:
                if cancel_event is not None and cancel_event.is_set():
                    self._advance(
                        machine,
                        SearchPhase.CANCELLED,
                        iteration,
                        None,
                        "run cancelled by its caller",
                    )
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    self._advance(
                        machine,
                        SearchPhase.TIMED_OUT,
                        iteration,
                        None,
                        f"wall-clock budget of {self._config.max_wall_seconds:g}s exhausted",
                    )
                    break
                if self._stop_condition is not None:
                    reason = self._stop_condition.check()
                    if reason:
                        self._advance(
                            machine,
                            SearchPhase.BUDGET_EXHAUSTED,
                            iteration,
                            None,
                            reason,
                        )
                        break
                iteration += 1
                revived: list[str] = []
                node = self._select(tree, revived)
                if node is None:
                    self._advance(
                        machine, SearchPhase.EXHAUSTED, iteration, None, "frontier empty"
                    )
                    break
                if revived:
                    self._advance(
                        machine,
                        SearchPhase.BACKTRACKING,
                        iteration,
                        node.id,
                        "reopened node for revision attempt",
                    )
                    self._advance(
                        machine,
                        SearchPhase.SELECTION,
                        iteration,
                        node.id,
                        "revision notes prepared",
                    )

                self._advance(machine, SearchPhase.EXPANSION, iteration, node.id, "")
                children = self._expand(tree, node)
                if not children:
                    self._advance(
                        machine,
                        SearchPhase.BACKTRACKING,
                        iteration,
                        node.id,
                        "expansion produced no candidates",
                    )
                    tree.prune_subtree(node)
                    self._advance(
                        machine, SearchPhase.SELECTION, iteration, None, "backtracked"
                    )
                    continue

                self._advance(
                    machine,
                    SearchPhase.EVALUATION,
                    iteration,
                    node.id,
                    f"{len(children)} candidates",
                )
                solution, valued = self._evaluate(children)

                self._advance(machine, SearchPhase.BACKPROPAGATION, iteration, node.id, "")
                self._backpropagate(valued)

                if solution is not None:
                    self._advance(
                        machine,
                        SearchPhase.SUCCEEDED,
                        iteration,
                        solution.id,
                        f"accepted with score {solution.score:.3f}",
                    )
                    break

                self._advance(
                    machine, SearchPhase.SELECTION, iteration, None, "iteration complete"
                )
        except Exception as exc:  # noqa: BLE001 - policy faults must not escape the run
            error = f"{type(exc).__name__}: {exc}"
            self._advance(machine, SearchPhase.FAILED, iteration, None, error)

        # Guards against non-terminal exit when the loop breaks via iteration
        # budget exhaustion while the machine still sits in SELECTION; a wall-
        # clock timeout already transitions to TIMED_OUT before breaking, so
        # this is a no-op in that case.
        if not machine.is_terminal:
            self._advance(
                machine, SearchPhase.EXHAUSTED, iteration, None, "iteration budget spent"
            )

        outcome = _OUTCOME_BY_PHASE[machine.phase]
        return SearchResult(
            outcome=outcome,
            best_path=tuple(tree.best_path()),
            iterations=iteration,
            node_count=len(tree),
            phase_history=machine.history,
            tree=tree,
            error=error,
        )

    def _select(
        self, tree: ThoughtTree, revived: list[str] | None = None
    ) -> ThoughtNode | None:
        """Descends from the root via UCT to the next node worth expanding.

        Dead interior nodes discovered during descent are pruned and the walk
        restarts from the root; each restart removes at least one node from
        the frontier, which bounds the loop. A saturated node is offered to
        the revision policy before pruning — a granted revision returns the
        node for re-expansion and reports it through ``revived``. Returns
        ``None`` when the tree holds no expandable node.
        """
        while True:
            node = tree.root
            if not node.is_live:
                return None
            while True:
                if not node.children:
                    if node.depth < self._config.max_depth:
                        return node
                    # Depth-capped leaf: a dead end that selection removes
                    # before restarting the descent.
                    tree.prune_subtree(node)
                    break
                live_children = [c for c in node.children if c.is_live]
                if not live_children:
                    if (
                        self._revision_policy is not None
                        and self._revision_policy.revise(node)
                    ):
                        # Semantic backtracking: the saturated node returns to
                        # the frontier carrying critique-derived notes.
                        if revived is not None:
                            revived.append(node.id)
                        return node
                    # Fully saturated interior node: structural backtracking
                    # collapses it so effort returns to a viable ancestor.
                    tree.prune_subtree(node)
                    break
                node = max(
                    live_children,
                    key=lambda c: (
                        c.uct_score(self._config.exploration_weight),
                        self._rng.random(),
                    ),
                )

    def _expand(self, tree: ThoughtTree, node: ThoughtNode) -> list[ThoughtNode]:
        """Materializes generator candidates as child nodes of ``node``.

        Existing children seed the deduplication set so a revised expansion
        cannot resubmit a candidate that already failed.
        """
        candidates = self._generator.generate(node, self._config.branching_factor)
        children: list[ThoughtNode] = []
        seen: set[str] = {child.content for child in node.children}
        for content in candidates[: self._config.branching_factor]:
            text = content.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            children.append(tree.add_child(node, text))
        return children

    def _evaluate(
        self, children: list[ThoughtNode]
    ) -> tuple[ThoughtNode | None, list[tuple[ThoughtNode, float]]]:
        """Scores each fresh child; returns the first accepted solution and
        the per-child backpropagation values.

        Terminal verdicts below the acceptance threshold are pruned rather
        than kept live, because a completed line of reasoning cannot be
        extended by further expansion. Pruned children are offered to the
        critic, whose diagnosis lands in node metadata and feeds both reward
        shaping and later revision notes. Acceptance always operates on the
        raw evaluator score; the reward model shapes only the value that
        backpropagates.
        """
        solution: ThoughtNode | None = None
        valued: list[tuple[ThoughtNode, float]] = []
        for child, verdict in zip(children, self._collect_verdicts(children), strict=True):
            if verdict.is_terminal and verdict.score >= self._config.accept_threshold:
                status = NodeStatus.TERMINAL
            elif verdict.is_terminal or verdict.score < self._config.prune_threshold:
                status = NodeStatus.PRUNED
            else:
                status = NodeStatus.EVALUATED
            child.apply_evaluation(verdict.score, status, verdict.rationale)

            critique: Critique | None = None
            if self._critic is not None and status is NodeStatus.PRUNED:
                critique = self._critic.critique(child)
                if critique is not None:
                    child.metadata[CRITIQUE_METADATA_KEY] = critique.to_dict()

            value = verdict.score
            if self._reward_model is not None:
                value = self._reward_model.shape(child, verdict.score, critique)
            valued.append((child, value))

            if status is NodeStatus.TERMINAL and solution is None:
                solution = child
        return solution, valued

    def _collect_verdicts(self, children: list[ThoughtNode]) -> list[Evaluation]:
        """Scores every candidate, optionally concurrently, in candidate order.

        Evaluation dominates the cost of a run — each verdict may spawn a
        sandboxed container — and the candidates within one batch are
        independent, so they parallelize cleanly. Two properties are preserved
        regardless of worker count, which is what keeps a seeded run
        reproducible and its failures diagnosable:

        * Verdicts are returned in candidate order, never completion order, so
          acceptance and pruning see the same sequence a sequential run would.
        * When several candidates fail, the exception belonging to the
          earliest candidate is the one that propagates.

        Both follow from :meth:`concurrent.futures.Executor.map`, which yields
        in submission order and re-raises at the first failing position. The
        single-worker path stays a plain loop so default runs take on no
        thread-pool machinery at all.
        """
        workers = min(self._config.evaluation_workers, len(children))
        if workers <= 1:
            return [self._evaluator.evaluate(child) for child in children]
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ctree-eval"
        ) as pool:
            return list(pool.map(self._evaluator.evaluate, children))

    def _backpropagate(self, valued: list[tuple[ThoughtNode, float]]) -> None:
        """Propagates each child's shaped value through its ancestor chain."""
        for child, value in valued:
            for node in child.path_from_root():
                node.record_visit(value)

    def _advance(
        self,
        machine: SearchStateMachine,
        target: SearchPhase,
        iteration: int,
        node_id: str | None,
        detail: str,
    ) -> None:
        """Transitions the state machine and mirrors the change to the event sink."""
        machine.transition(target, detail)
        if self._on_event is not None:
            self._on_event(
                SearchEvent(
                    iteration=iteration, phase=target, node_id=node_id, detail=detail
                )
            )


_OUTCOME_BY_PHASE: dict[SearchPhase, SearchOutcome] = {
    SearchPhase.SUCCEEDED: SearchOutcome.SUCCEEDED,
    SearchPhase.EXHAUSTED: SearchOutcome.EXHAUSTED,
    SearchPhase.FAILED: SearchOutcome.FAILED,
    SearchPhase.TIMED_OUT: SearchOutcome.TIMED_OUT,
    SearchPhase.BUDGET_EXHAUSTED: SearchOutcome.BUDGET_EXHAUSTED,
    SearchPhase.CANCELLED: SearchOutcome.CANCELLED,
}

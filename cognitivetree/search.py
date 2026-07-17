"""MCTS-driven Tree-of-Thoughts search controller.

The controller executes the canonical four-step MCTS cycle — selection,
expansion, evaluation, backpropagation — under the supervision of the
:class:`~cognitivetree.state.SearchStateMachine`. Dead branches collapse
upward through structural backtracking: when every child of a node is pruned
or failed, selection prunes the node itself and restarts from the root, so
search effort flows back to the nearest viable ancestor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, unique
from typing import Callable, Optional

from cognitivetree.config import SearchConfig
from cognitivetree.node import NodeStatus, ThoughtNode
from cognitivetree.policies import ThoughtEvaluator, ThoughtGenerator
from cognitivetree.state import PhaseTransition, SearchPhase, SearchStateMachine
from cognitivetree.tree import ThoughtTree


@unique
class SearchOutcome(Enum):
    """Final disposition of a search run."""

    SUCCEEDED = "succeeded"
    EXHAUSTED = "exhausted"
    FAILED = "failed"


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
        on_event: Optional[Callable[[SearchEvent], None]] = None,
    ) -> None:
        self._config = config
        self._generator = generator
        self._evaluator = evaluator
        self._on_event = on_event
        self._rng = random.Random(config.seed)

    def run(self, task: str) -> SearchResult:
        """Executes the search loop for ``task`` until a terminal phase is reached."""
        tree = ThoughtTree(task)
        machine = SearchStateMachine()
        iteration = 0
        error = ""

        self._advance(machine, SearchPhase.SELECTION, iteration, None, "search started")
        try:
            while iteration < self._config.max_iterations:
                iteration += 1
                node = self._select(tree)
                if node is None:
                    self._advance(
                        machine, SearchPhase.EXHAUSTED, iteration, None, "frontier empty"
                    )
                    break

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
                    if self._select(tree) is None:
                        self._advance(
                            machine, SearchPhase.EXHAUSTED, iteration, None, "frontier empty"
                        )
                        break
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
                solution = self._evaluate(children)

                self._advance(machine, SearchPhase.BACKPROPAGATION, iteration, node.id, "")
                self._backpropagate(children)

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

        # Guards against non-terminal exit when the loop breaks via budget
        # exhaustion while the machine still sits in SELECTION.
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

    def _select(self, tree: ThoughtTree) -> ThoughtNode | None:
        """Descends from the root via UCT to the next node worth expanding.

        Dead interior nodes discovered during descent are pruned and the walk
        restarts from the root; each restart removes at least one node from
        the frontier, which bounds the loop. Returns ``None`` when the tree
        holds no expandable node.
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
        """Materializes generator candidates as child nodes of ``node``."""
        candidates = self._generator.generate(node, self._config.branching_factor)
        children: list[ThoughtNode] = []
        seen: set[str] = set()
        for content in candidates[: self._config.branching_factor]:
            text = content.strip()
            if not text or text in seen:
                continue
            seen.add(text)
            children.append(tree.add_child(node, text))
        return children

    def _evaluate(self, children: list[ThoughtNode]) -> ThoughtNode | None:
        """Scores each fresh child and returns the first accepted solution.

        Terminal verdicts below the acceptance threshold are pruned rather
        than kept live, because a completed line of reasoning cannot be
        extended by further expansion.
        """
        solution: ThoughtNode | None = None
        for child in children:
            verdict = self._evaluator.evaluate(child)
            if verdict.is_terminal and verdict.score >= self._config.accept_threshold:
                status = NodeStatus.TERMINAL
            elif verdict.is_terminal or verdict.score < self._config.prune_threshold:
                status = NodeStatus.PRUNED
            else:
                status = NodeStatus.EVALUATED
            child.apply_evaluation(verdict.score, status, verdict.rationale)
            if status is NodeStatus.TERMINAL and solution is None:
                solution = child
        return solution

    def _backpropagate(self, children: list[ThoughtNode]) -> None:
        """Propagates each child's score through its ancestor chain."""
        for child in children:
            for node in child.path_from_root():
                node.record_visit(child.score)

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
}

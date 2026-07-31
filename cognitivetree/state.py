"""Finite-state machine that sequences the search control loop.

The machine encodes the only legal orderings of search phases as an explicit
transition table. The controller never mutates its own phase directly; every
change flows through :meth:`SearchStateMachine.transition`, which validates the
edge and records it, producing a complete, replayable trace of the run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, unique


@unique
class SearchPhase(Enum):
    """Enumerates every control state the search loop can occupy."""

    IDLE = "idle"
    SELECTION = "selection"
    EXPANSION = "expansion"
    EVALUATION = "evaluation"
    BACKPROPAGATION = "backpropagation"
    BACKTRACKING = "backtracking"
    SUCCEEDED = "succeeded"
    EXHAUSTED = "exhausted"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    BUDGET_EXHAUSTED = "budget_exhausted"


TERMINAL_PHASES: frozenset[SearchPhase] = frozenset(
    {
        SearchPhase.SUCCEEDED,
        SearchPhase.EXHAUSTED,
        SearchPhase.FAILED,
        SearchPhase.TIMED_OUT,
        SearchPhase.BUDGET_EXHAUSTED,
    }
)

# FAILED, TIMED_OUT, and BUDGET_EXHAUSTED share one reachability set: each
# represents an external constraint — a raised exception, a spent wall-clock
# deadline, a consumed resource budget — that can strike while the machine
# occupies any non-terminal phase, since the controller's exception handler,
# deadline check, and stop-condition poll all wrap the entire run. TIMED_OUT
# and BUDGET_EXHAUSTED stay distinct because they call for different
# responses: retry later versus raise the quota.
_TRANSITIONS: dict[SearchPhase, frozenset[SearchPhase]] = {
    SearchPhase.IDLE: frozenset(
        {
            SearchPhase.SELECTION,
            SearchPhase.FAILED,
            SearchPhase.TIMED_OUT,
            SearchPhase.BUDGET_EXHAUSTED,
        }
    ),
    SearchPhase.SELECTION: frozenset(
        {
            SearchPhase.EXPANSION,
            SearchPhase.BACKTRACKING,
            SearchPhase.EXHAUSTED,
            SearchPhase.FAILED,
            SearchPhase.TIMED_OUT,
            SearchPhase.BUDGET_EXHAUSTED,
        }
    ),
    SearchPhase.EXPANSION: frozenset(
        {
            SearchPhase.EVALUATION,
            SearchPhase.BACKTRACKING,
            SearchPhase.FAILED,
            SearchPhase.TIMED_OUT,
            SearchPhase.BUDGET_EXHAUSTED,
        }
    ),
    SearchPhase.EVALUATION: frozenset(
        {
            SearchPhase.BACKPROPAGATION,
            SearchPhase.FAILED,
            SearchPhase.TIMED_OUT,
            SearchPhase.BUDGET_EXHAUSTED,
        }
    ),
    SearchPhase.BACKPROPAGATION: frozenset(
        {
            SearchPhase.SELECTION,
            SearchPhase.SUCCEEDED,
            SearchPhase.EXHAUSTED,
            SearchPhase.FAILED,
            SearchPhase.TIMED_OUT,
            SearchPhase.BUDGET_EXHAUSTED,
        }
    ),
    SearchPhase.BACKTRACKING: frozenset(
        {
            SearchPhase.SELECTION,
            SearchPhase.EXHAUSTED,
            SearchPhase.FAILED,
            SearchPhase.TIMED_OUT,
            SearchPhase.BUDGET_EXHAUSTED,
        }
    ),
    SearchPhase.SUCCEEDED: frozenset(),
    SearchPhase.EXHAUSTED: frozenset(),
    SearchPhase.FAILED: frozenset(),
    SearchPhase.TIMED_OUT: frozenset(),
    SearchPhase.BUDGET_EXHAUSTED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    """Raised when a caller requests a transition the machine does not permit."""


@dataclass(frozen=True, slots=True)
class PhaseTransition:
    """Immutable record of a single accepted phase change."""

    source: SearchPhase
    target: SearchPhase
    note: str
    timestamp: float


class SearchStateMachine:
    """Guards the legal ordering of search phases and records every transition."""

    def __init__(self) -> None:
        self._phase = SearchPhase.IDLE
        self._history: list[PhaseTransition] = []

    @property
    def phase(self) -> SearchPhase:
        return self._phase

    @property
    def history(self) -> tuple[PhaseTransition, ...]:
        return tuple(self._history)

    @property
    def is_terminal(self) -> bool:
        return self._phase in TERMINAL_PHASES

    def can_transition(self, target: SearchPhase) -> bool:
        """Reports whether ``target`` is reachable from the current phase."""
        return target in _TRANSITIONS[self._phase]

    def transition(self, target: SearchPhase, note: str = "") -> PhaseTransition:
        """Moves the machine into ``target`` or raises when the edge is illegal.

        Returns the recorded :class:`PhaseTransition` so callers can forward it
        to event sinks without re-reading machine state.
        """
        if not self.can_transition(target):
            raise InvalidTransitionError(
                f"illegal transition {self._phase.value!r} -> {target.value!r}"
            )
        record = PhaseTransition(
            source=self._phase,
            target=target,
            note=note,
            timestamp=time.monotonic(),
        )
        self._phase = target
        self._history.append(record)
        return record

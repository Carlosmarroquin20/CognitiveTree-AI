"""Validates transition legality and trace recording of the search state machine."""

import pytest

from cognitivetree.state import (
    TERMINAL_PHASES,
    InvalidTransitionError,
    SearchPhase,
    SearchStateMachine,
)


def test_initial_phase_is_idle() -> None:
    machine = SearchStateMachine()
    assert machine.phase is SearchPhase.IDLE
    assert not machine.is_terminal
    assert machine.history == ()


def test_legal_cycle_reaches_success() -> None:
    machine = SearchStateMachine()
    path = [
        SearchPhase.SELECTION,
        SearchPhase.EXPANSION,
        SearchPhase.EVALUATION,
        SearchPhase.BACKPROPAGATION,
        SearchPhase.SUCCEEDED,
    ]
    for phase in path:
        machine.transition(phase)
    assert machine.phase is SearchPhase.SUCCEEDED
    assert machine.is_terminal
    assert [t.target for t in machine.history] == path
    assert machine.history[0].source is SearchPhase.IDLE


def test_backtracking_edge_returns_to_selection() -> None:
    machine = SearchStateMachine()
    machine.transition(SearchPhase.SELECTION)
    machine.transition(SearchPhase.EXPANSION)
    machine.transition(SearchPhase.BACKTRACKING)
    machine.transition(SearchPhase.SELECTION)
    assert machine.phase is SearchPhase.SELECTION


def test_illegal_transition_raises_and_preserves_state() -> None:
    machine = SearchStateMachine()
    with pytest.raises(InvalidTransitionError):
        machine.transition(SearchPhase.EVALUATION)
    assert machine.phase is SearchPhase.IDLE
    assert machine.history == ()


@pytest.mark.parametrize("terminal", sorted(TERMINAL_PHASES, key=lambda p: p.value))
def test_terminal_phases_admit_no_exit(terminal: SearchPhase) -> None:
    machine = SearchStateMachine()
    machine.transition(SearchPhase.SELECTION)
    if terminal is SearchPhase.SUCCEEDED:
        machine.transition(SearchPhase.EXPANSION)
        machine.transition(SearchPhase.EVALUATION)
        machine.transition(SearchPhase.BACKPROPAGATION)
    machine.transition(terminal)
    assert machine.is_terminal
    for phase in SearchPhase:
        assert not machine.can_transition(phase)


def test_transition_notes_are_recorded() -> None:
    machine = SearchStateMachine()
    machine.transition(SearchPhase.SELECTION, note="search started")
    assert machine.history[0].note == "search started"

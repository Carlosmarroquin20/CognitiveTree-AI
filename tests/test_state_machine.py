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


# Shortest legal transition sequence from IDLE that lands the machine in each
# non-terminal phase, so the reachability matrix below drives only real,
# publicly legal transitions rather than reaching into machine internals.
_PATH_TO_PHASE: dict[SearchPhase, tuple[SearchPhase, ...]] = {
    SearchPhase.IDLE: (),
    SearchPhase.SELECTION: (SearchPhase.SELECTION,),
    SearchPhase.EXPANSION: (SearchPhase.SELECTION, SearchPhase.EXPANSION),
    SearchPhase.EVALUATION: (
        SearchPhase.SELECTION,
        SearchPhase.EXPANSION,
        SearchPhase.EVALUATION,
    ),
    SearchPhase.BACKPROPAGATION: (
        SearchPhase.SELECTION,
        SearchPhase.EXPANSION,
        SearchPhase.EVALUATION,
        SearchPhase.BACKPROPAGATION,
    ),
    SearchPhase.BACKTRACKING: (SearchPhase.SELECTION, SearchPhase.BACKTRACKING),
}


def test_path_matrix_covers_every_non_terminal_phase() -> None:
    assert set(_PATH_TO_PHASE) == {p for p in SearchPhase if p not in TERMINAL_PHASES}


@pytest.mark.parametrize(
    "phase", sorted(_PATH_TO_PHASE, key=lambda p: p.value), ids=lambda p: p.value
)
def test_timed_out_mirrors_failed_reachability(phase: SearchPhase) -> None:
    # A wall-clock deadline, like a raised exception, can strike while the
    # machine occupies any non-terminal phase; TIMED_OUT must therefore be
    # reachable from exactly the same phases as FAILED.
    machine = SearchStateMachine()
    for step in _PATH_TO_PHASE[phase]:
        machine.transition(step)
    assert machine.phase is phase
    assert machine.can_transition(SearchPhase.FAILED)
    assert machine.can_transition(SearchPhase.TIMED_OUT)


_EXTERNAL_STOPS = (
    SearchPhase.TIMED_OUT,
    SearchPhase.BUDGET_EXHAUSTED,
    SearchPhase.CANCELLED,
)


@pytest.mark.parametrize(
    "phase", sorted(_PATH_TO_PHASE, key=lambda p: p.value), ids=lambda p: p.value
)
@pytest.mark.parametrize("stop", _EXTERNAL_STOPS, ids=lambda p: p.value)
def test_external_stops_mirror_failed_reachability(
    phase: SearchPhase, stop: SearchPhase
) -> None:
    machine = SearchStateMachine()
    for step in _PATH_TO_PHASE[phase]:
        machine.transition(step)
    assert machine.can_transition(SearchPhase.FAILED)
    assert machine.can_transition(stop)

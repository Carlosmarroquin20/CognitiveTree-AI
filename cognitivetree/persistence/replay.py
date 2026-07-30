"""Replay of archived runs through the live streaming envelope vocabulary.

A replay session emits the same ``phase`` / ``snapshot`` / ``metrics`` /
``result`` envelopes a live search produces, so the streaming UI renders an
archived run with no client-side changes and no awareness that the search
already finished. It exposes the same ``task`` and ``stream()`` surface as
:class:`~cognitivetree.session.ReasoningSession`, which is all the HTTP server
consumes, making the two interchangeable at the server boundary.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cognitivetree.observability.metrics import RunMetrics
from cognitivetree.persistence.archive import RunArchive, load_run
from cognitivetree.search import SearchEvent
from cognitivetree.state import TERMINAL_PHASES, SearchPhase
from cognitivetree.tree import ThoughtTree
from cognitivetree.ui.events import (
    metrics_envelope,
    phase_envelope,
    result_envelope,
    snapshot_envelope,
)

_SNAPSHOT_PHASES = frozenset({SearchPhase.BACKPROPAGATION}) | TERMINAL_PHASES


class ReplaySession:
    """Streams an archived run in the shape of a live one.

    ``speed`` controls pacing: ``None`` (the default) emits everything without
    delay, which is what offline inspection wants, while a positive factor
    reproduces the original inter-transition gaps divided by that factor —
    ``1.0`` replays at the pace the run actually had.
    """

    def __init__(self, archive: RunArchive, speed: float | None = None) -> None:
        if speed is not None and speed <= 0:
            raise ValueError("speed must be positive when provided")
        self._archive = archive
        self._speed = speed

    @classmethod
    def from_path(cls, path: str | Path, speed: float | None = None) -> ReplaySession:
        """Loads the archive at ``path`` and wraps it for replay."""
        return cls(load_run(path), speed=speed)

    @property
    def task(self) -> str:
        return self._archive.task

    @property
    def archive(self) -> RunArchive:
        return self._archive

    def stream(self) -> Iterator[dict[str, Any]]:
        """Yields the archived run as an ordered envelope sequence.

        Snapshots are reconstructed rather than stored per transition: the
        archive holds only the tree's final state, so every snapshot carries
        it. Structural evolution over time is not recoverable from an archive
        — the phase log, which is recorded per transition, is what conveys
        how the run progressed.
        """
        result = self._archive.result
        tree_payload = result.tree.to_dict(include_metadata=True)
        history = result.phase_history
        iteration = 0

        for index, transition in enumerate(history):
            if index > 0:
                self._wait(history[index - 1].timestamp, transition.timestamp)
            iteration = _iteration_for(transition, iteration)
            yield phase_envelope(
                SearchEvent(
                    iteration=iteration,
                    phase=transition.target,
                    node_id=None,
                    detail=transition.note,
                )
            )
            if transition.target in _SNAPSHOT_PHASES:
                yield snapshot_envelope(ThoughtTree.from_dict(tree_payload))

        metrics = self._archive.metrics or RunMetrics.from_result(result).to_dict()
        yield metrics_envelope(metrics)
        yield result_envelope(result)

    def _wait(self, previous: float, current: float) -> None:
        """Sleeps the scaled gap between two recorded transitions."""
        if self._speed is None:
            return
        delay = (current - previous) / self._speed
        if delay > 0:
            time.sleep(delay)


def _iteration_for(transition: Any, current: int) -> int:
    """Infers the iteration counter from the transition sequence.

    Iteration numbers are not recorded per transition, but the controller
    increments them exactly once per selection-driven cycle; counting entries
    into EXPANSION reproduces the same sequence the live stream carried.
    """
    return current + 1 if transition.target is SearchPhase.EXPANSION else current

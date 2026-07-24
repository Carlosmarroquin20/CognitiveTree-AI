"""JSON envelope vocabulary and SSE framing for streamed runs.

Three envelope types cover a run: ``phase`` mirrors every state-machine
transition, ``snapshot`` carries a full serialized tree at backpropagation
and terminal phases, and ``result`` closes the stream with the aggregated
outcome. The envelope ``type`` doubles as the SSE event name so browser
clients subscribe with plain ``EventSource`` listeners.
"""

from __future__ import annotations

import json
import time
from typing import Any

from cognitivetree.search import SearchEvent, SearchResult
from cognitivetree.tree import ThoughtTree


def phase_envelope(event: SearchEvent) -> dict[str, Any]:
    """Wraps a controller event for transport."""
    return {
        "type": "phase",
        "iteration": event.iteration,
        "phase": event.phase.value,
        "node_id": event.node_id,
        "detail": event.detail,
        "ts": time.time(),
    }


def snapshot_envelope(tree: ThoughtTree) -> dict[str, Any]:
    """Wraps a full tree serialization for transport."""
    return {"type": "snapshot", "tree": tree.to_dict(), "ts": time.time()}


def result_envelope(result: SearchResult) -> dict[str, Any]:
    """Wraps the final run outcome for transport."""
    return {
        "type": "result",
        "outcome": result.outcome.value,
        "iterations": result.iterations,
        "node_count": result.node_count,
        "solution": result.solution,
        "error": result.error,
        "best_path": [node.id for node in result.best_path],
        "ts": time.time(),
    }


def format_sse(envelope: dict[str, Any]) -> bytes:
    """Frames an envelope as one Server-Sent Events message.

    The payload is serialized on a single line, so no ``data:`` continuation
    handling is required on either side.
    """
    payload = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return f"event: {envelope['type']}\ndata: {payload}\n\n".encode()

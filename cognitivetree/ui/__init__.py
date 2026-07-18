"""Streaming interface exposing reasoning runs over Server-Sent Events.

The stack is standard-library only: :mod:`events` defines the JSON envelope
vocabulary and SSE framing, :mod:`server` hosts the HTTP endpoint pair
(``/`` for the embedded page, ``/stream`` for the event stream), and
:mod:`serve` provides the command-line entry point.
"""

from cognitivetree.ui.events import (
    format_sse,
    phase_envelope,
    result_envelope,
    snapshot_envelope,
)
from cognitivetree.ui.server import StreamingUiServer

__all__ = [
    "StreamingUiServer",
    "format_sse",
    "phase_envelope",
    "result_envelope",
    "snapshot_envelope",
]

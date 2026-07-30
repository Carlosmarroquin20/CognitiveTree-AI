"""Durable archival and replay of completed reasoning runs.

A run archive is a single self-contained JSON document holding the thought
tree (with its per-node execution records, critiques, and reward breakdowns),
the state-machine transition history, and the metrics summary. Loading one
rehydrates a real :class:`~cognitivetree.search.SearchResult`, so every tool
that operates on a live run — :class:`~cognitivetree.observability.RunMetrics`,
``tree.render()``, ``best_path`` — works unchanged on an archived one.
"""

from cognitivetree.persistence.archive import (
    ARCHIVE_FORMAT,
    ARCHIVE_VERSION,
    ArchiveFormatError,
    RunArchive,
    load_run,
    result_to_archive_document,
    save_run,
)
from cognitivetree.persistence.replay import ReplaySession

__all__ = [
    "ARCHIVE_FORMAT",
    "ARCHIVE_VERSION",
    "ArchiveFormatError",
    "ReplaySession",
    "RunArchive",
    "load_run",
    "result_to_archive_document",
    "save_run",
]

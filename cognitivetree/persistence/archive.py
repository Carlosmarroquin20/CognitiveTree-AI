"""Versioned JSON archival of completed runs, and their rehydration.

The archive round-trips a run through a single document so a search that ended
hours earlier — on a CI runner, behind a timeout, on a machine without a model
— can be reopened and dissected locally. Rehydration deliberately produces a
genuine :class:`~cognitivetree.search.SearchResult` rather than a parallel
read-only type: the existing metrics, rendering, and path-extraction tooling
then applies to archived runs without a second code path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cognitivetree.search import SearchOutcome, SearchResult
from cognitivetree.state import PhaseTransition, SearchPhase
from cognitivetree.tree import ThoughtTree

ARCHIVE_FORMAT = "cognitivetree-run"
ARCHIVE_VERSION = 1


class ArchiveFormatError(ValueError):
    """Raised when a document is not a readable run archive."""


@dataclass(frozen=True, slots=True)
class RunArchive:
    """A loaded run archive.

    Attributes:
        result: The rehydrated run, indistinguishable in type from one the
            controller just produced.
        metrics: The metrics summary captured at save time, or ``None`` when
            the archive was written without one. Retained rather than always
            recomputed because token accounting originates outside the result
            (see :class:`~cognitivetree.observability.AccountingLlmClient`)
            and cannot be derived from the tree afterwards.
        saved_at: UTC ISO-8601 timestamp of when the archive was written.
        version: Archive schema version the document declared.
    """

    result: SearchResult
    metrics: dict[str, Any] | None
    saved_at: str
    version: int

    @property
    def task(self) -> str:
        """Returns the task statement the run was launched with."""
        return self.result.tree.root.content


def result_to_archive_document(
    result: SearchResult, metrics: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Builds the JSON-compatible archive document for ``result``.

    Phase timestamps are recorded as captured, on the monotonic clock: their
    absolute values carry no meaning across processes, but their differences —
    which is all the metrics layer consumes — survive the round trip exactly.
    """
    return {
        "format": ARCHIVE_FORMAT,
        "version": ARCHIVE_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "task": result.tree.root.content,
        "outcome": result.outcome.value,
        "iterations": result.iterations,
        "node_count": result.node_count,
        "error": result.error,
        "best_path": [node.id for node in result.best_path],
        "tree": result.tree.to_dict(include_metadata=True),
        "phase_history": [
            {
                "source": transition.source.value,
                "target": transition.target.value,
                "note": transition.note,
                "timestamp": transition.timestamp,
            }
            for transition in result.phase_history
        ],
        "metrics": metrics,
    }


def save_run(
    result: SearchResult,
    path: str | Path,
    metrics: dict[str, Any] | None = None,
    indent: int | None = 2,
) -> Path:
    """Writes ``result`` to ``path`` as a run archive and returns the path.

    Parent directories are created as needed. ``metrics`` accepts the output
    of :meth:`~cognitivetree.observability.RunMetrics.to_dict`; supplying it
    preserves token accounting, which the result alone cannot reproduce.
    """
    document = result_to_archive_document(result, metrics=metrics)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(document, ensure_ascii=False, indent=indent), encoding="utf-8"
    )
    return destination


def load_run(path: str | Path) -> RunArchive:
    """Reads a run archive and rehydrates it into a :class:`RunArchive`."""
    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArchiveFormatError(f"{source} is not valid JSON: {exc}") from exc
    return archive_from_document(document)


def archive_from_document(document: Any) -> RunArchive:
    """Rehydrates a parsed archive document, validating its shape first."""
    if not isinstance(document, dict):
        raise ArchiveFormatError("archive document must be a JSON object")
    if document.get("format") != ARCHIVE_FORMAT:
        raise ArchiveFormatError(
            f"unrecognized archive format {document.get('format')!r}; "
            f"expected {ARCHIVE_FORMAT!r}"
        )
    version = document.get("version")
    if version != ARCHIVE_VERSION:
        raise ArchiveFormatError(
            f"unsupported archive version {version!r}; this build reads "
            f"version {ARCHIVE_VERSION}"
        )

    try:
        tree = ThoughtTree.from_dict(document["tree"])
        outcome = SearchOutcome(document["outcome"])
        phase_history = tuple(
            PhaseTransition(
                source=SearchPhase(entry["source"]),
                target=SearchPhase(entry["target"]),
                note=entry.get("note", ""),
                timestamp=float(entry["timestamp"]),
            )
            for entry in document["phase_history"]
        )
        best_path = tuple(tree.get(node_id) for node_id in document["best_path"])
        result = SearchResult(
            outcome=outcome,
            best_path=best_path,
            iterations=int(document["iterations"]),
            node_count=int(document["node_count"]),
            phase_history=phase_history,
            tree=tree,
            error=document.get("error", ""),
        )
    except (KeyError, TypeError) as exc:
        raise ArchiveFormatError(f"archive document is missing or malformed: {exc}") from exc
    except ValueError as exc:
        raise ArchiveFormatError(f"archive document holds an invalid value: {exc}") from exc

    return RunArchive(
        result=result,
        metrics=document.get("metrics"),
        saved_at=document.get("saved_at", ""),
        version=version,
    )

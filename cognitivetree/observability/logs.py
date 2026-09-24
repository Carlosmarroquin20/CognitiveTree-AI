"""Log output for operators: human-readable text or one JSON object per line.

Structured fields travel through the standard ``extra`` mapping, so library
code logs with plain :mod:`logging` calls and never depends on this module;
only the entry point that owns the process chooses the output format.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

LOG_FORMATS = ("text", "json")

_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Attributes every LogRecord carries; anything else on a record arrived
# through ``extra`` and is a structured field worth emitting.
_RECORD_ATTRIBUTES = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None))
) | {"message", "asctime", "taskName"}


class JsonLogFormatter(logging.Formatter):
    """Renders each record as a single-line JSON object.

    Every object carries ``ts`` (UTC, ISO 8601), ``level``, ``logger``, and
    ``message``, followed by the record's ``extra`` fields and, for records
    logged with exception information, the formatted traceback under
    ``exception``. Values JSON cannot encode natively, such as paths, are
    rendered with ``str`` rather than dropped.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RECORD_ATTRIBUTES and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO, log_format: str = "text") -> None:
    """Installs a root handler writing ``log_format`` output to stderr.

    Replaces any handlers already on the root logger, so calling it twice
    does not duplicate every line.
    """
    if log_format not in LOG_FORMATS:
        raise ValueError(f"log_format must be one of {', '.join(LOG_FORMATS)}")
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonLogFormatter() if log_format == "json" else logging.Formatter(_TEXT_FORMAT)
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

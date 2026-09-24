"""Validates the JSON log formatter and process-level log configuration."""

import json
import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from cognitivetree.observability.logs import JsonLogFormatter, configure_logging
from cognitivetree.ui.serve import build_parser


def render(record: logging.LogRecord) -> dict[str, object]:
    line = JsonLogFormatter().format(record)
    assert "\n" not in line
    parsed = json.loads(line)
    assert isinstance(parsed, dict)
    return parsed


def make_record(message: str, *args: object, **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        "cognitivetree.session", logging.INFO, __file__, 1, message, args, None
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_core_fields_are_present() -> None:
    payload = render(make_record("run %s finished", "abc"))
    assert payload["message"] == "run abc finished"
    assert payload["level"] == "info"
    assert payload["logger"] == "cognitivetree.session"
    assert str(payload["ts"]).endswith("+00:00")


def test_extra_fields_are_emitted_and_record_internals_are_not() -> None:
    payload = render(make_record("run finished", outcome="succeeded", tokens=430))
    assert payload["outcome"] == "succeeded"
    assert payload["tokens"] == 430
    for internal in ("args", "msg", "levelno", "pathname", "created"):
        assert internal not in payload


def test_values_json_cannot_encode_are_stringified() -> None:
    payload = render(make_record("archived", archive=Path("runs") / "a.json"))
    assert payload["archive"] == str(Path("runs") / "a.json")


def test_multiline_messages_stay_on_one_line() -> None:
    assert render(make_record("first\nsecond"))["message"] == "first\nsecond"


def test_exception_traceback_is_included() -> None:
    try:
        raise ValueError("broken backend")
    except ValueError:
        record = make_record("run failed")
        record.exc_info = sys.exc_info()
    payload = render(record)
    assert "ValueError: broken backend" in str(payload["exception"])


@pytest.fixture
def isolated_root() -> Iterator[logging.Logger]:
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    yield root
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_configure_logging_installs_one_json_handler(isolated_root: logging.Logger) -> None:
    configure_logging(logging.DEBUG, "json")
    configure_logging(logging.DEBUG, "json")

    assert len(isolated_root.handlers) == 1
    assert isinstance(isolated_root.handlers[0].formatter, JsonLogFormatter)
    assert isolated_root.level == logging.DEBUG


def test_configure_logging_rejects_unknown_formats(isolated_root: logging.Logger) -> None:
    with pytest.raises(ValueError, match="log_format"):
        configure_logging(logging.INFO, "xml")


def test_cli_defaults_to_text_and_accepts_json() -> None:
    assert build_parser().parse_args([]).log_format == "text"
    assert build_parser().parse_args(["--log-format", "json"]).log_format == "json"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--log-format", "xml"])

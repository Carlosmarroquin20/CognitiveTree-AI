"""Parsers for structured content inside model completions."""

from __future__ import annotations

import json
import re

CANDIDATE_MARKER = "### CANDIDATE"

_MARKER_LINE = re.compile(r"^###\s*candidate\b.*$", re.IGNORECASE | re.MULTILINE)
_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


def split_candidates(text: str) -> list[str]:
    """Splits a completion into candidate thoughts.

    The primary format is ``### CANDIDATE`` marker lines, which survive
    embedded code fences and numbered content. Completions without markers
    fall back to top-level numbered-list parsing, and finally to treating the
    whole completion as a single candidate. Blank candidates are dropped.
    """
    if _MARKER_LINE.search(text):
        parts = _MARKER_LINE.split(text)
        return _cleaned(parts)
    numbered = _split_numbered(text)
    if numbered:
        return numbered
    stripped = text.strip()
    return [stripped] if stripped else []


def _split_numbered(text: str) -> list[str]:
    """Splits on top-level ``1.`` / ``2)`` markers; empty when none exist."""
    matches = list(_NUMBERED_LINE.finditer(text))
    if len(matches) < 2:
        return []
    blocks: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[start:end])
    return _cleaned(blocks)


def _cleaned(parts: list[str]) -> list[str]:
    return [part.strip() for part in parts if part.strip()]


def extract_json_object(text: str) -> dict | None:
    """Returns the first balanced JSON object found in ``text``, or ``None``.

    The scanner tolerates surrounding prose and code fences, which model
    outputs frequently add around requested JSON.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : position + 1])
                    except json.JSONDecodeError:
                        break
                    return parsed if isinstance(parsed, dict) else None
        start = text.find("{", start + 1)
    return None

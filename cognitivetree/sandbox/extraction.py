"""Extraction of executable payloads from free-form thought content.

Model output is markdown written by a model, not by a parser, so the fences
it produces vary: ``Python`` or ``python3`` instead of ``python``, a space
before the tag, fences indented inside a list, and blocks in other
languages (sample output, shell commands) ahead of the code. Each of these
used to cost a correct candidate its evaluation, either by finding no code
at all or by pairing the wrong fences and executing prose. Blocks are
therefore parsed line by line, with a fence closed only by a matching fence.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass

_OPENING_FENCE = re.compile(r"^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^`]*)$")
_PYTHON_TAG = re.compile(r"^(?:py|python)(?:3(?:\.\d+)?)?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _Block:
    tag: str
    body: str


def extract_python_payload(text: str) -> str | None:
    """Returns the Python code a thought carries, or ``None`` when it has none.

    The first non-blank block tagged as Python (``python``, ``py``,
    ``python3``, and so on, in any case) wins; failing that, the first
    non-blank untagged block. Blocks tagged with another language are
    skipped. The body is dedented, so a fence indented inside a markdown list
    yields runnable code, and a block left unclosed at the end of the text
    runs to the end, as in CommonMark, so a truncated completion is still
    executed and diagnosed rather than ignored.
    """
    untagged: str | None = None
    for block in _fenced_blocks(text):
        if not block.body:
            continue
        if _PYTHON_TAG.match(block.tag):
            return block.body
        if not block.tag and untagged is None:
            untagged = block.body
    return untagged


def _fenced_blocks(text: str) -> list[_Block]:
    """Splits ``text`` into its fenced blocks, pairing each fence with its closer."""
    blocks: list[_Block] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        opening = _OPENING_FENCE.match(lines[index])
        index += 1
        if opening is None:
            continue
        fence = opening.group("fence")
        info = opening.group("info").split()
        body: list[str] = []
        while index < len(lines) and not _closes(lines[index], fence):
            body.append(lines[index])
            index += 1
        index += 1  # Steps past the closing fence, or beyond the end.
        blocks.append(
            _Block(tag=info[0] if info else "", body=textwrap.dedent("\n".join(body)).strip())
        )
    return blocks


def _closes(line: str, fence: str) -> bool:
    """Reports whether ``line`` closes a block opened with ``fence``."""
    stripped = line.strip()
    return len(stripped) >= len(fence) and stripped == fence[0] * len(stripped)

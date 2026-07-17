"""Extraction of executable payloads from free-form thought content."""

from __future__ import annotations

import re

_FENCED_BLOCK = re.compile(
    r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
    re.DOTALL,
)


def extract_python_payload(text: str) -> str | None:
    """Returns the first fenced Python code block found in ``text``.

    Accepts ``python`` / ``py`` language tags as well as untagged fences.
    Returns ``None`` when the text carries no fenced block or the block is
    blank, signalling that the thought holds no executable artifact.
    """
    match = _FENCED_BLOCK.search(text)
    if match is None:
        return None
    payload = match.group(1).strip()
    return payload or None

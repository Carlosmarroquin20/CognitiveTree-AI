"""Validates payload extraction from free-form thought content."""

from cognitivetree.sandbox.extraction import extract_python_payload


def test_extracts_tagged_python_block() -> None:
    text = "Reasoning first.\n```python\nprint('x')\n```\ntrailing prose"
    assert extract_python_payload(text) == "print('x')"


def test_extracts_py_tag_and_untagged_fences() -> None:
    assert extract_python_payload("```py\nvalue = 1\n```") == "value = 1"
    assert extract_python_payload("```\nvalue = 2\n```") == "value = 2"


def test_returns_first_of_multiple_blocks() -> None:
    text = "```python\nfirst = True\n```\n```python\nsecond = True\n```"
    assert extract_python_payload(text) == "first = True"


def test_preserves_internal_structure() -> None:
    code = "def f(x):\n    if x:\n        return 1\n    return 0"
    assert extract_python_payload(f"```python\n{code}\n```") == code


def test_returns_none_without_fence() -> None:
    assert extract_python_payload("plain reasoning, no code") is None


def test_returns_none_for_blank_block() -> None:
    assert extract_python_payload("```python\n   \n```") is None


def test_handles_crlf_line_endings() -> None:
    assert extract_python_payload("```python\r\nvalue = 3\r\n```") == "value = 3"

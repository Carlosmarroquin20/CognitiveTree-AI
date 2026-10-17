"""Validates payload extraction from free-form thought content."""

import pytest

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


FENCE = "`" * 3


def fenced(tag: str, body: str, fence: str = FENCE) -> str:
    return f"{fence}{tag}\n{body}\n{fence}"


@pytest.mark.parametrize(
    "tag", ["Python", "PYTHON", "python3", "python3.12", "py3", " python", "python  "]
)
def test_accepts_the_tag_variants_models_write(tag: str) -> None:
    # Each of these used to yield no payload, so a correct candidate was
    # never executed and could never be accepted.
    assert extract_python_payload(fenced(tag, "value = 1")) == "value = 1"


def test_skips_a_block_in_another_language_before_the_code() -> None:
    # The old pattern paired this text block's closing fence with the next
    # opening one and executed the prose in between.
    text = (
        "Expected output:\n"
        + fenced("text", "1")
        + "\nImplementation:\n"
        + fenced("python", "def f():\n    return 1")
    )
    assert extract_python_payload(text) == "def f():\n    return 1"


def test_tagged_python_wins_over_an_earlier_untagged_block() -> None:
    text = fenced("", "f(1)") + "\n" + fenced("python", "def f(x):\n    return x")
    assert extract_python_payload(text) == "def f(x):\n    return x"


def test_untagged_block_is_the_fallback_when_nothing_is_tagged_python() -> None:
    text = fenced("bash", "pip install x") + "\n" + fenced("", "value = 2")
    assert extract_python_payload(text) == "value = 2"


def test_other_languages_alone_carry_no_payload() -> None:
    text = fenced("bash", "pip install x") + "\n" + fenced("json", '{"a": 1}')
    assert extract_python_payload(text) is None


def test_fence_indented_inside_a_list_is_dedented() -> None:
    # Stripping only the first line left later top-level statements indented,
    # which raised IndentationError on a correct answer.
    text = (
        "1. Define it:\n   " + FENCE + "python\n   def f():\n       return 1\n"
        "   print(f())\n   " + FENCE
    )
    assert extract_python_payload(text) == "def f():\n    return 1\nprint(f())"


def test_shorter_fence_inside_a_longer_one_does_not_close_it() -> None:
    body = 'doc = """\n' + FENCE + '\nexample\n' + FENCE + '\n"""'
    assert extract_python_payload(fenced("python", body, fence="`" * 4)) == body


def test_tilde_fences_are_recognized() -> None:
    assert extract_python_payload(fenced("python", "value = 3", fence="~~~")) == "value = 3"


def test_unclosed_block_runs_to_the_end_of_the_text() -> None:
    # A completion cut off by max_tokens still gets executed and diagnosed.
    assert extract_python_payload(FENCE + "python\ndef f(:\n") == "def f(:"


def test_blank_block_does_not_hide_a_later_one() -> None:
    text = fenced("python", "   ") + "\n" + fenced("python", "value = 4")
    assert extract_python_payload(text) == "value = 4"

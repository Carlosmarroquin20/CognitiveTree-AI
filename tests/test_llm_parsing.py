"""Validates completion parsing: candidate splitting and JSON extraction."""

from cognitivetree.llm.parsing import extract_json_object, split_candidates


def test_marker_split_survives_code_fences() -> None:
    text = (
        "### CANDIDATE\nUse a loop.\n```python\nfor i in range(3):\n    print(i)\n```\n"
        "### CANDIDATE\nUse recursion.\n```python\ndef f(n):\n    return n\n```\n"
    )
    candidates = split_candidates(text)
    assert len(candidates) == 2
    assert candidates[0].startswith("Use a loop.")
    assert "recursion" in candidates[1]


def test_marker_split_is_case_insensitive_and_tolerant() -> None:
    text = "### candidate 1\nfirst\n### Candidate 2\nsecond\n"
    assert split_candidates(text) == ["first", "second"]


def test_numbered_fallback() -> None:
    text = "1. First idea spans\nmultiple lines.\n2) Second idea.\n3. Third idea."
    candidates = split_candidates(text)
    assert len(candidates) == 3
    assert candidates[0] == "First idea spans\nmultiple lines."
    assert candidates[1] == "Second idea."


def test_single_numbered_line_is_not_split() -> None:
    text = "The answer involves 1. something inline only."
    assert split_candidates(text) == [text]


def test_whole_text_fallback_and_empty_input() -> None:
    assert split_candidates("just one thought") == ["just one thought"]
    assert split_candidates("   \n  ") == []


def test_json_extraction_from_clean_object() -> None:
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_json_extraction_from_prose_and_fences() -> None:
    text = 'Here is my diagnosis:\n```json\n{"severity": 0.8, "nested": {"x": [1, 2]}}\n```\nDone.'
    assert extract_json_object(text) == {"severity": 0.8, "nested": {"x": [1, 2]}}


def test_json_extraction_handles_braces_in_strings() -> None:
    text = '{"guidance": "wrap in braces {like this}", "severity": 0.5}'
    parsed = extract_json_object(text)
    assert parsed is not None
    assert parsed["guidance"] == "wrap in braces {like this}"


def test_json_extraction_skips_invalid_then_finds_valid() -> None:
    text = "{broken json} but later {\"ok\": true}"
    assert extract_json_object(text) == {"ok": True}


def test_json_extraction_returns_none_without_object() -> None:
    assert extract_json_object("no json here") is None
    assert extract_json_object("[1, 2, 3]") is None

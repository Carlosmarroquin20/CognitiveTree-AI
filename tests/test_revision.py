"""Validates revision eligibility rules and notes compilation."""

import pytest

from cognitivetree.feedback.revision import (
    REVISION_ATTEMPTS_KEY,
    REVISION_NOTES_KEY,
    BoundedRevisionPolicy,
    compile_revision_notes,
)
from cognitivetree.node import NodeStatus, ThoughtNode
from cognitivetree.policies import CRITIQUE_METADATA_KEY


def critique_dict(guidance: str, failure_class: str = "assertion") -> dict:
    return {
        "failure_class": failure_class,
        "summary": "a check failed",
        "guidance": guidance,
        "severity": 0.6,
    }


def saturated_parent(guidances: list[str | None]) -> ThoughtNode:
    """Builds a parent whose children are all pruned, with optional critiques."""
    parent = ThoughtNode(content="task")
    for index, guidance in enumerate(guidances):
        child = parent.attach_child(f"candidate {index}")
        child.status = NodeStatus.PRUNED
        if guidance is not None:
            child.metadata[CRITIQUE_METADATA_KEY] = critique_dict(guidance)
    return parent


def test_childless_node_is_not_revisable() -> None:
    assert BoundedRevisionPolicy().revise(ThoughtNode(content="task")) is False


def test_live_child_blocks_revision() -> None:
    parent = saturated_parent(["fix A"])
    parent.attach_child("still viable").status = NodeStatus.EVALUATED
    assert BoundedRevisionPolicy().revise(parent) is False


def test_terminal_child_blocks_revision() -> None:
    parent = saturated_parent(["fix A"])
    parent.attach_child("solved").status = NodeStatus.TERMINAL
    assert BoundedRevisionPolicy().revise(parent) is False


def test_grant_compiles_notes_and_counts_attempt() -> None:
    parent = saturated_parent(["fix A", "fix B"])
    assert BoundedRevisionPolicy(max_attempts=1).revise(parent) is True
    assert parent.metadata[REVISION_ATTEMPTS_KEY] == 1
    notes = parent.metadata[REVISION_NOTES_KEY]
    assert "- [assertion] fix A" in notes
    assert "- [assertion] fix B" in notes


def test_budget_is_enforced_across_calls() -> None:
    parent = saturated_parent(["fix A"])
    policy = BoundedRevisionPolicy(max_attempts=1)
    assert policy.revise(parent) is True
    assert policy.revise(parent) is False
    assert parent.metadata[REVISION_ATTEMPTS_KEY] == 1


def test_guidance_requirement_gates_revision() -> None:
    parent = saturated_parent([None, None])
    assert BoundedRevisionPolicy(require_guidance=True).revise(parent) is False
    assert BoundedRevisionPolicy(require_guidance=False).revise(parent) is True
    assert parent.metadata[REVISION_NOTES_KEY] == ""


def test_notes_deduplicate_repeated_guidance() -> None:
    parent = saturated_parent(["fix A", "fix A", "fix B"])
    notes = compile_revision_notes(parent)
    assert notes.count("fix A") == 1
    assert notes.count("fix B") == 1


def test_invalid_budget_is_rejected() -> None:
    with pytest.raises(ValueError):
        BoundedRevisionPolicy(max_attempts=0)

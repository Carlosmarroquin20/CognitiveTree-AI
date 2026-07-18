"""Validates composite reward shaping and its recorded breakdown."""

import pytest

from cognitivetree.feedback.rewards import (
    REWARD_METADATA_KEY,
    RewardShaper,
    RewardWeights,
)
from cognitivetree.node import ThoughtNode
from cognitivetree.policies import Critique, FailureClass


def make_node(depth: int) -> ThoughtNode:
    return ThoughtNode(content="thought", depth=depth)


def make_critique(severity: float) -> Critique:
    return Critique(
        failure_class=FailureClass.ASSERTION,
        summary="check failed",
        guidance="fix it",
        severity=severity,
    )


def test_default_weights_compose_expected_value() -> None:
    node = make_node(depth=1)
    shaped = RewardShaper().shape(node, base_score=0.5, critique=make_critique(0.6))
    expected = 0.7 * 0.5 + 0.2 * 0.4 + 0.1 * (1.0 - 1 / 16)
    assert shaped == pytest.approx(expected)


def test_absent_critique_contributes_full_term() -> None:
    node = make_node(depth=0)
    shaped = RewardShaper().shape(node, base_score=1.0, critique=None)
    assert shaped == pytest.approx(0.7 * 1.0 + 0.2 * 1.0 + 0.1 * 1.0)


def test_depth_term_bottoms_out_at_horizon() -> None:
    weights = RewardWeights(depth_horizon=4)
    shallow = RewardShaper(weights).shape(make_node(1), 0.5, None)
    deep = RewardShaper(weights).shape(make_node(32), 0.5, None)
    assert shallow > deep
    breakdown = make_node(32)
    RewardShaper(weights).shape(breakdown, 0.5, None)
    assert breakdown.metadata[REWARD_METADATA_KEY]["depth_term"] == 0.0


def test_breakdown_is_recorded_in_metadata() -> None:
    node = make_node(depth=2)
    shaped = RewardShaper().shape(node, base_score=0.05, critique=make_critique(0.9))
    record = node.metadata[REWARD_METADATA_KEY]
    assert record["base_score"] == pytest.approx(0.05)
    assert record["critique_term"] == pytest.approx(0.1)
    assert record["shaped"] == pytest.approx(shaped, abs=1e-6)


def test_shaped_value_stays_within_unit_interval() -> None:
    node = make_node(depth=0)
    assert 0.0 <= RewardShaper().shape(node, 0.0, make_critique(1.0)) <= 1.0
    assert 0.0 <= RewardShaper().shape(node, 1.0, None) <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"evaluation": -0.1},
        {"evaluation": 0.0, "critique": 0.0, "depth": 0.0},
        {"depth_horizon": 0},
    ],
)
def test_invalid_weights_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RewardWeights(**kwargs)

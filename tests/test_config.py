"""Validates SearchConfig field constraints, including the wall-clock budget."""

import pytest

from cognitivetree.config import SearchConfig


def test_defaults_are_unbounded_in_wall_time() -> None:
    assert SearchConfig().max_wall_seconds is None


def test_accepts_a_positive_wall_clock_budget() -> None:
    assert SearchConfig(max_wall_seconds=2.5).max_wall_seconds == 2.5


@pytest.mark.parametrize("value", [0.0, -1.0, -0.001])
def test_rejects_non_positive_wall_clock_budget(value: float) -> None:
    with pytest.raises(ValueError, match="max_wall_seconds"):
        SearchConfig(max_wall_seconds=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_iterations": 0},
        {"max_depth": 0},
        {"branching_factor": 0},
        {"exploration_weight": -0.1},
        {"prune_threshold": 0.9, "accept_threshold": 0.5},
    ],
)
def test_rejects_invalid_core_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)


def test_config_is_immutable() -> None:
    config = SearchConfig()
    with pytest.raises(AttributeError):
        config.max_wall_seconds = 10.0  # type: ignore[misc]

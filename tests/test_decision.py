"""Decision rules and targeting policies."""

from __future__ import annotations

import numpy as np
import pytest

from cip.decision import recommend, targeting_policy
from cip.errors import EstimationError


def test_a_large_credible_effect_ships() -> None:
    result = recommend(5.0, (2.0, 8.0), baseline=100.0)
    assert result.verdict == "ship"
    assert result.credible and result.material


def test_a_real_but_tiny_effect_does_not_ship() -> None:
    """The gate significance cannot supply: real is not the same as worth doing."""
    result = recommend(0.2, (0.1, 0.3), baseline=100.0, min_relative_effect=0.01)
    assert result.verdict == "credible but immaterial"
    assert result.credible and not result.material


def test_an_interval_spanning_zero_is_inconclusive() -> None:
    result = recommend(1.0, (-3.0, 5.0), baseline=100.0)
    assert result.verdict == "inconclusive"
    assert "power" in result.reason


def test_a_credibly_negative_effect_is_refused() -> None:
    result = recommend(-5.0, (-9.0, -1.0), baseline=100.0)
    assert result.verdict == "do not ship"


def test_a_posterior_probability_can_drive_the_decision() -> None:
    result = recommend(
        5.0, (-1.0, 11.0), baseline=100.0, probability_of_benefit=0.97, min_probability=0.95
    )
    assert result.credible
    assert result.verdict == "ship"


def test_a_weak_posterior_does_not_ship() -> None:
    result = recommend(
        5.0, (-1.0, 11.0), baseline=100.0, probability_of_benefit=0.80, min_probability=0.95
    )
    assert result.verdict == "inconclusive"


def test_the_thresholds_are_reported_with_the_verdict() -> None:
    """Disagreeing with a recommendation should mean disagreeing with a number."""
    result = recommend(5.0, (2.0, 8.0), baseline=100.0, min_relative_effect=0.02)
    assert result.min_relative_effect == 0.02
    assert "2.00%" in result.reason


def test_reversed_interval_bounds_are_rejected() -> None:
    with pytest.raises(EstimationError, match="reversed"):
        recommend(1.0, (5.0, 2.0), baseline=10.0)


def test_targeting_treats_only_profitable_units() -> None:
    policy = targeting_policy(
        np.array([1.0, 0.5, 0.1, -0.2]), cost_per_treatment=0.4, value_per_outcome=1.0
    )
    assert policy["treat"].tolist() == [True, True, False, False]


def test_everyone_is_treated_when_treatment_is_free() -> None:
    policy = targeting_policy(np.array([0.5, 0.2, 0.01]), cost_per_treatment=0.0)
    assert policy["treat"].all()


def test_a_budget_cap_keeps_the_highest_uplift_units() -> None:
    policy = targeting_policy(
        np.array([0.1, 0.9, 0.5, 0.3]), cost_per_treatment=0.0, budget_share=0.5
    )
    assert policy["treat"].tolist() == [False, True, True, False]


def test_an_impossible_budget_share_is_rejected() -> None:
    with pytest.raises(EstimationError, match="budget_share"):
        targeting_policy(np.array([0.1, 0.2]), budget_share=1.5)

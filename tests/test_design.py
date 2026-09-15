"""Sample size, power and balance checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cip.data.simulator import SimulationResult, estimation_frame
from cip.design import (
    check_balance,
    minimum_detectable_effect,
    power_for_sample_size,
    sample_size_for_means,
    sample_size_for_proportions,
    standardised_difference,
)
from cip.errors import DesignError


def test_sizing_and_power_are_inverses() -> None:
    """A size computed for 80% power must deliver 80% power, or one of them is wrong."""
    sized = sample_size_for_means(100.0, 40.0, 0.05, power=0.80)
    achieved = power_for_sample_size(sized.effect_size, sized.per_arm)
    assert achieved.power == pytest.approx(0.80, abs=0.01)


def test_a_smaller_effect_needs_a_larger_sample() -> None:
    big = sample_size_for_means(100.0, 40.0, 0.20)
    small = sample_size_for_means(100.0, 40.0, 0.02)
    assert small.per_arm > big.per_arm * 10


def test_more_power_costs_more_units() -> None:
    low = sample_size_for_means(100.0, 40.0, 0.05, power=0.60)
    high = sample_size_for_means(100.0, 40.0, 0.05, power=0.95)
    assert high.per_arm > low.per_arm


def test_a_rarer_conversion_rate_needs_more_units() -> None:
    common = sample_size_for_proportions(0.30, 0.10)
    rare = sample_size_for_proportions(0.01, 0.10)
    assert rare.per_arm > common.per_arm


def test_a_lift_that_leaves_the_unit_interval_is_rejected() -> None:
    with pytest.raises(DesignError, match="leaves the unit interval"):
        sample_size_for_proportions(0.9, 0.5)


def test_minimum_detectable_effect_shrinks_with_sample_size() -> None:
    assert minimum_detectable_effect(10_000) < minimum_detectable_effect(100)


def test_an_underpowered_test_is_visibly_underpowered() -> None:
    """The number that stops a null result being read as evidence of no effect."""
    assert power_for_sample_size(0.05, 100).power < 0.15


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"alpha": 0.0}, "alpha"),
        ({"power": 1.0}, "power"),
    ],
)
def test_impossible_design_parameters_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(DesignError, match=message):
        sample_size_for_means(100.0, 40.0, 0.05, **kwargs)


def test_standardised_difference_is_symmetric_under_relabelling() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 3.0, 4.0, 5.0])
    assert standardised_difference(a, b) == pytest.approx(-standardised_difference(b, a))


def test_identical_arms_are_perfectly_balanced() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert standardised_difference(values, values) == pytest.approx(0.0)


def test_a_randomised_design_passes_its_own_balance_check(
    randomised: SimulationResult,
) -> None:
    frame = estimation_frame(randomised)
    report = check_balance(frame, randomised.treatment, list(randomised.covariates))
    assert report.passed, report.reason()


def test_a_confounded_design_fails_the_balance_check(confounded: SimulationResult) -> None:
    """This is the check earning its keep: assignment depended on covariates."""
    frame = estimation_frame(confounded)
    report = check_balance(frame, confounded.treatment, list(confounded.covariates))
    assert not report.passed
    assert report.imbalanced
    assert "imbalanced" in report.reason()


def test_a_broken_bucketing_is_caught_even_when_covariates_balance() -> None:
    """A 70/30 split when 50/50 was intended, with covariates drawn identically."""
    rng = np.random.default_rng(0)
    n = 4000
    treatment = (rng.uniform(size=n) < 0.7).astype(int)
    frame = pd.DataFrame({"t": treatment, "x": rng.normal(size=n)})
    report = check_balance(frame, "t", ["x"], expected_share=0.5)
    assert not report.passed
    assert report.share_pvalue < 0.01


def test_missing_covariates_are_named(frame: pd.DataFrame) -> None:
    with pytest.raises(DesignError, match="missing covariates"):
        check_balance(frame, "treatment", ["not_a_column"])


def test_a_non_binary_treatment_is_rejected() -> None:
    frame = pd.DataFrame({"t": [0, 1, 2, 1], "x": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(DesignError, match="binary"):
        check_balance(frame, "t", ["x"])

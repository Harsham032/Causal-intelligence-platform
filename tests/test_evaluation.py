"""Scoring estimators and testing sensitivity to hidden confounding."""

from __future__ import annotations

import numpy as np
import pytest

from cip.errors import EvaluationError
from cip.evaluation import (
    coverage_of_intervals,
    estimator_bias,
    rosenbaum_bounds,
    unmeasured_confounding,
)


def test_an_unbiased_estimator_reports_no_bias() -> None:
    rng = np.random.default_rng(0)
    report = estimator_bias(rng.normal(loc=2.0, scale=0.1, size=500), truth=2.0)
    assert abs(report.bias) < 0.02
    assert report.rmse > 0


def test_a_biased_estimator_is_caught() -> None:
    report = estimator_bias(np.full(100, 3.0), truth=2.0)
    assert report.bias == pytest.approx(1.0)
    assert report.relative_bias == pytest.approx(0.5)


def test_correct_intervals_cover_at_their_nominal_rate() -> None:
    """A 95% interval that covers 95% of the time; the property being asserted."""
    rng = np.random.default_rng(1)
    truth, n = 2.0, 3000
    estimates = rng.normal(loc=truth, scale=1.0, size=n)
    lowers, uppers = estimates - 1.96, estimates + 1.96
    report = coverage_of_intervals(lowers, uppers, truth)
    assert report.passed, report.detail


def test_intervals_that_are_too_narrow_are_caught() -> None:
    """Overconfidence is invisible in the point estimate; this is how it surfaces."""
    rng = np.random.default_rng(2)
    truth, n = 2.0, 3000
    estimates = rng.normal(loc=truth, scale=1.0, size=n)
    report = coverage_of_intervals(estimates - 0.5, estimates + 0.5, truth)
    assert not report.passed
    assert "too narrow" in report.detail


def test_reversed_intervals_are_rejected() -> None:
    with pytest.raises(EvaluationError, match="reversed"):
        coverage_of_intervals([1.0], [0.0], truth=0.5)


def test_a_strong_result_survives_hidden_bias() -> None:
    rng = np.random.default_rng(3)
    treated = rng.normal(loc=5.0, scale=1.0, size=200)
    control = rng.normal(loc=0.0, scale=1.0, size=200)
    report = rosenbaum_bounds(treated, control)
    assert report.robust


def test_a_marginal_result_breaks_easily() -> None:
    rng = np.random.default_rng(4)
    treated = rng.normal(loc=0.12, scale=1.0, size=60)
    control = rng.normal(loc=0.0, scale=1.0, size=60)
    report = rosenbaum_bounds(treated, control)
    assert not report.robust


def test_tied_pairs_cannot_be_tested() -> None:
    with pytest.raises(EvaluationError, match="tied"):
        rosenbaum_bounds(np.ones(10), np.ones(10))


def test_confounding_strength_scales_with_the_estimate() -> None:
    weak = unmeasured_confounding(estimate=1.0, standard_error=0.5)
    strong = unmeasured_confounding(estimate=10.0, standard_error=0.5)
    assert strong.breaking_point > weak.breaking_point
    assert strong.robust and not weak.robust


def test_a_zero_standard_error_is_rejected() -> None:
    with pytest.raises(EvaluationError, match="positive"):
        unmeasured_confounding(estimate=1.0, standard_error=0.0)

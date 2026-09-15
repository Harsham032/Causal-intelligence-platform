"""Two-arm tests and multiple-testing control."""

from __future__ import annotations

import numpy as np
import pytest

from cip.data.simulator import SimulationResult, estimation_frame
from cip.errors import EstimationError
from cip.frequentist import (
    adjust_pvalues,
    bootstrap_difference,
    difference_in_means,
    difference_in_proportions,
)


def _arms(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    frame = estimation_frame(result)
    treated = frame.loc[frame[result.treatment] == 1, result.outcome].to_numpy()
    control = frame.loc[frame[result.treatment] == 0, result.outcome].to_numpy()
    return treated, control


def test_a_randomised_experiment_recovers_its_true_effect(randomised: SimulationResult) -> None:
    treated, control = _arms(randomised)
    result = difference_in_means(treated, control)
    assert result.absolute_effect == pytest.approx(randomised.true_ate, abs=0.2)
    assert result.ci_lower <= randomised.true_ate <= result.ci_upper


def test_the_bootstrap_agrees_with_the_analytic_interval(randomised: SimulationResult) -> None:
    """Two routes to the same answer; a disagreement means one of them is wrong."""
    treated, control = _arms(randomised)
    analytic = difference_in_means(treated, control)
    boot = bootstrap_difference(treated, control, resamples=600, seed=3)
    assert boot.absolute_effect == pytest.approx(analytic.absolute_effect, abs=1e-9)
    assert boot.ci_lower == pytest.approx(analytic.ci_lower, rel=0.25)
    assert boot.ci_upper == pytest.approx(analytic.ci_upper, rel=0.25)


def test_a_null_effect_is_not_declared_significant() -> None:
    rng = np.random.default_rng(0)
    result = difference_in_means(rng.normal(size=2000), rng.normal(size=2000))
    assert not result.significant
    assert result.pvalue > 0.05


def test_welch_handles_unequal_variances() -> None:
    """Student's test would overstate confidence here; Welch should not."""
    rng = np.random.default_rng(1)
    treated = rng.normal(loc=0.0, scale=10.0, size=100)
    control = rng.normal(loc=0.0, scale=1.0, size=2000)
    result = difference_in_means(treated, control)
    assert not result.significant


def test_proportions_require_binary_arms() -> None:
    with pytest.raises(EstimationError, match="binary"):
        difference_in_proportions(np.array([0.5, 1.0]), np.array([0.0, 1.0]))


def test_a_proportion_interval_stays_interpretable_near_zero() -> None:
    """The normal approximation can produce impossible bounds at low rates."""
    rng = np.random.default_rng(2)
    treated = (rng.uniform(size=5000) < 0.004).astype(float)
    control = (rng.uniform(size=5000) < 0.002).astype(float)
    result = difference_in_proportions(treated, control)
    assert -1.0 <= result.ci_lower <= result.ci_upper <= 1.0


def test_a_large_binary_effect_is_detected(binary_trial: SimulationResult) -> None:
    treated, control = _arms(binary_trial)
    result = difference_in_proportions(treated, control)
    assert result.significant
    assert result.absolute_effect > 0


def test_tiny_arms_are_rejected() -> None:
    with pytest.raises(EstimationError, match="at least two observations"):
        difference_in_means([1.0], [2.0, 3.0])


def test_bonferroni_is_never_less_conservative_than_benjamini_hochberg() -> None:
    rng = np.random.default_rng(5)
    p = rng.uniform(size=50)
    bonferroni = adjust_pvalues(p, method="bonferroni")
    bh = adjust_pvalues(p, method="benjamini-hochberg")
    assert all(b >= h for b, h in zip(bonferroni.adjusted, bh.adjusted, strict=True))


def test_benjamini_hochberg_is_monotone() -> None:
    """Without the monotonicity pass the procedure is simply invalid."""
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.5])
    adjusted = np.array(adjust_pvalues(p, method="benjamini-hochberg").adjusted)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)


def test_the_uncorrected_error_rate_is_the_problem_being_solved() -> None:
    """Twenty null metrics at alpha 0.05 produce a false win most of the time."""
    rng = np.random.default_rng(7)
    trials = 400
    uncorrected = sum(
        bool(adjust_pvalues(rng.uniform(size=20), method="none").discoveries) for _ in range(trials)
    )
    assert uncorrected / trials > 0.5


def test_corrections_control_the_error_rate() -> None:
    rng = np.random.default_rng(8)
    trials = 400
    for method in ("bonferroni", "benjamini-hochberg"):
        false_positives = sum(
            bool(adjust_pvalues(rng.uniform(size=20), method=method).discoveries)
            for _ in range(trials)
        )
        assert false_positives / trials < 0.12, method


def test_misaligned_names_are_rejected() -> None:
    with pytest.raises(EstimationError, match="align"):
        adjust_pvalues([0.1, 0.2], names=["only_one"])


def test_out_of_range_pvalues_are_rejected() -> None:
    with pytest.raises(EstimationError, match=r"\[0, 1\]"):
        adjust_pvalues([0.5, 1.5])

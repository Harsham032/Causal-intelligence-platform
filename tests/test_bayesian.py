"""Bayesian two-arm analysis."""

from __future__ import annotations

import numpy as np
import pytest

from cip.bayesian import beta_binomial_test, normal_test
from cip.data.simulator import SimulationResult, estimation_frame
from cip.errors import EstimationError
from cip.frequentist import difference_in_proportions


def _arms(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    frame = estimation_frame(result)
    return (
        frame.loc[frame[result.treatment] == 1, result.outcome].to_numpy(),
        frame.loc[frame[result.treatment] == 0, result.outcome].to_numpy(),
    )


def test_the_conjugate_posterior_recovers_a_known_effect(
    binary_trial: SimulationResult,
) -> None:
    treated, control = _arms(binary_trial)
    result = beta_binomial_test(treated, control)
    assert result.ci_lower <= binary_trial.true_ate <= result.ci_upper


def test_conjugate_and_frequentist_agree_on_a_large_sample(
    binary_trial: SimulationResult,
) -> None:
    """With a flat prior and thousands of rows, the two should be hard to tell apart."""
    treated, control = _arms(binary_trial)
    bayes = beta_binomial_test(treated, control)
    freq = difference_in_proportions(treated, control)
    assert bayes.effect_mean == pytest.approx(freq.absolute_effect, abs=0.01)
    assert bayes.ci_lower == pytest.approx(freq.ci_lower, abs=0.02)


def test_a_clear_winner_gets_a_high_probability_of_benefit() -> None:
    rng = np.random.default_rng(0)
    treated = (rng.uniform(size=4000) < 0.30).astype(float)
    control = (rng.uniform(size=4000) < 0.20).astype(float)
    result = beta_binomial_test(treated, control)
    assert result.probability_of_benefit > 0.999
    assert result.expected_loss < 1e-3


def test_no_difference_gives_an_uncommitted_posterior() -> None:
    rng = np.random.default_rng(1)
    treated = (rng.uniform(size=3000) < 0.20).astype(float)
    control = (rng.uniform(size=3000) < 0.20).astype(float)
    result = beta_binomial_test(treated, control)
    assert 0.2 < result.probability_of_benefit < 0.8


def test_the_rope_catches_a_real_but_negligible_effect() -> None:
    """A tiny true difference should land mostly inside the equivalence region."""
    rng = np.random.default_rng(2)
    treated = (rng.uniform(size=60000) < 0.2002).astype(float)
    control = (rng.uniform(size=60000) < 0.2000).astype(float)
    result = beta_binomial_test(treated, control, rope_relative=0.05)
    assert result.probability_practically_equivalent > 0.5


def test_a_non_binary_arm_is_rejected() -> None:
    with pytest.raises(EstimationError, match="binary"):
        beta_binomial_test(np.array([0.5, 1.0]), np.array([0.0, 1.0]))


def test_an_invalid_prior_is_rejected() -> None:
    with pytest.raises(EstimationError, match="positive"):
        beta_binomial_test(np.array([0.0, 1.0]), np.array([1.0, 0.0]), prior_alpha=0.0)


def test_an_unknown_likelihood_is_rejected() -> None:
    with pytest.raises(EstimationError, match="unknown likelihood"):
        normal_test(
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0, 3.0]),
            likelihood="cauchy",  # type: ignore[arg-type]
        )


@pytest.mark.mcmc
@pytest.mark.slow
def test_the_normal_likelihood_matches_the_difference_in_means(
    randomised: SimulationResult,
) -> None:
    """The estimand check: a Normal model should land on the mean shift."""
    treated, control = _arms(randomised)
    result = normal_test(
        treated, control, likelihood="normal", draws=500, tune=300, chains=2, seed=5
    )
    observed = float(treated.mean() - control.mean())
    assert result.effect_mean == pytest.approx(observed, abs=0.15)
    assert result.diagnostics.get("r_hat", 1.0) < 1.05


@pytest.mark.mcmc
@pytest.mark.slow
def test_convergence_diagnostics_are_reported(randomised: SimulationResult) -> None:
    """A posterior from unconverged chains is numbers, not inference."""
    treated, control = _arms(randomised)
    result = normal_test(treated, control, draws=400, tune=300, chains=2, seed=6)
    assert "r_hat" in result.diagnostics
    assert "ess_bulk" in result.diagnostics
    assert result.diagnostics["ess_bulk"] > 100

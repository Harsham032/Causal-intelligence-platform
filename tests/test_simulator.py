"""The simulator, which everything else is validated against.

If the generator is wrong, every measurement built on it is wrong in the same
direction and nothing downstream will notice. These tests check the properties
the rest of the suite assumes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cip.data.simulator import (
    SimulationConfig,
    SimulationResult,
    estimation_frame,
    simulate_experiment,
)
from cip.errors import DataError


def test_a_randomised_design_recovers_its_own_effect(randomised: SimulationResult) -> None:
    """With no confounding a difference in means is unbiased, by construction."""
    summary = randomised.describe()
    assert summary["observed_difference_in_means"] == pytest.approx(summary["true_ate"], abs=0.15)


def test_confounding_biases_the_naive_comparison() -> None:
    """The dial must actually do something, monotonically."""
    biases = []
    for strength in (0.0, 1.0, 2.0, 4.0):
        result = simulate_experiment(SimulationConfig(n_units=8000, confounding=strength, seed=5))
        summary = result.describe()
        biases.append(summary["observed_difference_in_means"] - summary["true_ate"])
    assert abs(biases[0]) < 0.1, "a randomised design should not be biased"
    assert biases == sorted(biases), f"bias should grow with confounding, got {biases}"
    assert biases[-1] > 2.0


def test_potential_outcomes_are_consistent_with_the_observed_one() -> None:
    """The observed outcome must be the potential outcome of the assigned arm."""
    result = simulate_experiment(SimulationConfig(n_units=1000, seed=3))
    data = result.data
    treated = data["treatment"] == 1
    assert np.allclose(data.loc[treated, "outcome"], data.loc[treated, "y1"])
    assert np.allclose(data.loc[~treated, "outcome"], data.loc[~treated, "y0"])


def test_the_true_effect_is_the_expected_gap_between_potential_outcomes() -> None:
    """true_cate is E[Y(1) - Y(0) | X], not the realised difference for a row.

    The two potential outcomes carry independent idiosyncratic noise, so a
    single unit's realised difference is the effect plus that noise - which is
    exactly why an individual effect is not observable even in simulation. The
    conditional expectation is what an uplift model can be asked to rank on, so
    that is what is recorded, and it is the *average* realised gap that must
    match it.
    """
    result = simulate_experiment(SimulationConfig(n_units=4000, effect="constant", seed=4))
    data = result.data
    realised = (data["y1"] - data["y0"]).to_numpy()
    assert realised.mean() == pytest.approx(result.true_ate, abs=0.1)
    assert result.true_ate == pytest.approx(result.config.average_effect, abs=1e-9)


def test_a_binary_outcome_records_the_realised_effect_exactly() -> None:
    """On the probability scale the effect is p1 - p0, which is not noisy."""
    result = simulate_experiment(
        SimulationConfig(n_units=2000, outcome="binary", baseline=0.3, average_effect=0.1, seed=4)
    )
    assert result.data["true_cate"].mean() == pytest.approx(result.true_ate, abs=1e-12)


def test_estimators_never_see_the_answer(randomised: SimulationResult) -> None:
    """An estimator handed the potential outcomes would score perfectly and prove nothing."""
    visible = set(estimation_frame(randomised).columns)
    for hidden in ("y0", "y1", "true_cate", "propensity_true"):
        assert hidden not in visible


def test_a_constant_effect_has_no_heterogeneity() -> None:
    result = simulate_experiment(SimulationConfig(n_units=2000, effect="constant", seed=6))
    assert result.data["true_cate"].std(ddof=1) == pytest.approx(0.0, abs=1e-12)


def test_a_null_effect_is_exactly_zero() -> None:
    result = simulate_experiment(SimulationConfig(n_units=2000, effect="none", seed=7))
    assert result.true_ate == 0.0
    assert result.true_att == 0.0


def test_a_binary_outcome_stays_binary(binary_trial: SimulationResult) -> None:
    values = set(np.unique(binary_trial.data["outcome"]).tolist())
    assert values <= {0.0, 1.0}


def test_binary_effects_cannot_leave_the_probability_scale() -> None:
    """A large effect on a high baseline must clip rather than produce impossible rates."""
    result = simulate_experiment(
        SimulationConfig(n_units=2000, outcome="binary", baseline=0.9, average_effect=0.5, seed=8)
    )
    assert result.data["true_cate"].max() <= 1.0
    assert result.data["true_cate"].min() >= -1.0


def test_the_same_seed_gives_the_same_sample() -> None:
    a = simulate_experiment(SimulationConfig(n_units=500, seed=99))
    b = simulate_experiment(SimulationConfig(n_units=500, seed=99))
    assert np.allclose(a.data["outcome"], b.data["outcome"])
    assert a.true_ate == b.true_ate


def test_different_seeds_give_different_samples() -> None:
    a = simulate_experiment(SimulationConfig(n_units=500, seed=1))
    b = simulate_experiment(SimulationConfig(n_units=500, seed=2))
    assert not np.allclose(a.data["outcome"], b.data["outcome"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_units": 5}, "at least 10 units"),
        ({"treatment_share": 0.0}, "strictly between 0 and 1"),
        ({"n_informative": 50, "n_covariates": 10}, "cannot exceed"),
        ({"confounding": -1.0}, "must not be negative"),
        ({"outcome": "binary", "baseline": 5.0}, "must be a probability"),
    ],
)
def test_impossible_configurations_are_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(DataError, match=message):
        SimulationConfig(**kwargs)

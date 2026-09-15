"""Uplift learners and the curves that score them."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from cip.data.simulator import (
    SimulationConfig,
    SimulationResult,
    estimation_frame,
    simulate_experiment,
)
from cip.errors import EstimationError, EvaluationError
from cip.uplift import fit_uplift, qini_curve, uplift_by_decile


@pytest.fixture(scope="module")
def heterogeneous() -> SimulationResult:
    return simulate_experiment(
        SimulationConfig(
            n_units=4000,
            n_covariates=6,
            n_informative=3,
            effect="heterogeneous",
            average_effect=1.0,
            effect_scale=1.0,
            seed=21,
        )
    )


def test_a_model_beats_random_ranking(heterogeneous: SimulationResult) -> None:
    frame = estimation_frame(heterogeneous)
    model = fit_uplift(
        frame,
        outcome=heterogeneous.outcome,
        treatment=heterogeneous.treatment,
        covariates=list(heterogeneous.covariates),
        learner="s-learner",
        n_estimators=60,
        max_depth=3,
        n_folds=3,
    )
    fitted = qini_curve(
        frame[heterogeneous.outcome], frame[heterogeneous.treatment], model.predicted_uplift
    )
    rng = np.random.default_rng(0)
    random = qini_curve(
        frame[heterogeneous.outcome], frame[heterogeneous.treatment], rng.normal(size=len(frame))
    )
    assert fitted.qini_coefficient > random.qini_coefficient


def test_predictions_correlate_with_effects_the_model_never_saw(
    heterogeneous: SimulationResult,
) -> None:
    """The measurement only a simulation can support."""
    frame = estimation_frame(heterogeneous)
    model = fit_uplift(
        frame,
        outcome=heterogeneous.outcome,
        treatment=heterogeneous.treatment,
        covariates=list(heterogeneous.covariates),
        learner="s-learner",
        n_estimators=60,
        max_depth=3,
        n_folds=3,
    )
    correlation = stats.spearmanr(model.predicted_uplift, heterogeneous.true_cate).statistic
    assert correlation > 0.5


def test_the_oracle_ranking_is_the_ceiling(heterogeneous: SimulationResult) -> None:
    """No model can beat ranking by the true effect; if one does, the metric is wrong."""
    frame = estimation_frame(heterogeneous)
    oracle = qini_curve(
        frame[heterogeneous.outcome], frame[heterogeneous.treatment], heterogeneous.true_cate
    )
    model = fit_uplift(
        frame,
        outcome=heterogeneous.outcome,
        treatment=heterogeneous.treatment,
        covariates=list(heterogeneous.covariates),
        learner="s-learner",
        n_estimators=60,
        max_depth=3,
        n_folds=3,
    )
    fitted = qini_curve(
        frame[heterogeneous.outcome], frame[heterogeneous.treatment], model.predicted_uplift
    )
    assert fitted.qini_coefficient <= oracle.qini_coefficient + 1e-9


def test_an_inverted_ranking_scores_below_random(heterogeneous: SimulationResult) -> None:
    """A confidently wrong model spends budget on the people it repels."""
    frame = estimation_frame(heterogeneous)
    inverted = qini_curve(
        frame[heterogeneous.outcome], frame[heterogeneous.treatment], -heterogeneous.true_cate
    )
    assert inverted.qini_coefficient < 0.0


def test_the_decile_table_is_monotone_for_a_good_model(
    heterogeneous: SimulationResult,
) -> None:
    frame = estimation_frame(heterogeneous)
    table = uplift_by_decile(
        frame[heterogeneous.outcome],
        frame[heterogeneous.treatment],
        heterogeneous.true_cate,
        n_bins=4,
    )
    observed = table["observed_uplift"].to_numpy()
    assert observed[0] > observed[-1], f"top bin should beat bottom bin, got {observed}"


def test_a_null_effect_produces_no_meaningful_ranking() -> None:
    result = simulate_experiment(SimulationConfig(n_units=3000, effect="none", seed=22))
    frame = estimation_frame(result)
    rng = np.random.default_rng(1)
    curve = qini_curve(frame[result.outcome], frame[result.treatment], rng.normal(size=len(frame)))
    assert abs(curve.qini_coefficient) < 0.25


@pytest.mark.parametrize("learner", ["t-learner", "s-learner", "dr-learner"])
def test_every_learner_runs_and_predicts_one_value_per_unit(
    heterogeneous: SimulationResult, learner: str
) -> None:
    frame = estimation_frame(heterogeneous)
    model = fit_uplift(
        frame,
        outcome=heterogeneous.outcome,
        treatment=heterogeneous.treatment,
        covariates=list(heterogeneous.covariates),
        learner=learner,  # type: ignore[arg-type]
        n_estimators=40,
        max_depth=3,
        n_folds=2,
    )
    assert model.predicted_uplift.shape == (len(frame),)
    assert np.isfinite(model.predicted_uplift).all()


def test_an_unknown_learner_is_rejected(heterogeneous: SimulationResult) -> None:
    frame = estimation_frame(heterogeneous)
    with pytest.raises(EstimationError, match="unknown learner"):
        fit_uplift(
            frame,
            outcome=heterogeneous.outcome,
            treatment=heterogeneous.treatment,
            covariates=list(heterogeneous.covariates),
            learner="magic",  # type: ignore[arg-type]
            n_folds=2,
        )


def test_misaligned_inputs_are_rejected() -> None:
    with pytest.raises(EvaluationError, match="align"):
        qini_curve([1.0, 0.0], [1, 0], [0.5])


def test_a_single_arm_cannot_be_scored() -> None:
    with pytest.raises(EvaluationError, match="both arms"):
        qini_curve([1.0, 0.0, 1.0], [1, 1, 1], [0.5, 0.2, 0.1])

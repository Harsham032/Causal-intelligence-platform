"""Double machine learning and causal forests.

Regression adjustment breaks when the relationship between covariates and
outcome is not the shape the model assumed. Machine learning fixes the shape
problem and introduces a worse one: a flexible model fitted on the same rows it
is evaluated on overfits, and that overfitting lands directly in the treatment
effect as bias.

Double machine learning solves it with two ideas that only work together.
**Orthogonalisation** removes the covariates' influence from both the treatment
and the outcome, and regresses one residual on the other, so first-stage errors
cancel to first order instead of propagating. **Cross-fitting** predicts every
unit from models that never saw it, so the residuals carry no overfitting at
all. Drop either and the bias comes back.

Causal forests answer the neighbouring question - who benefits most - by
splitting on treatment-effect heterogeneity rather than on outcome level, and
return an interval for each unit's effect rather than a point guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ..errors import EstimationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class CausalEstimate:
    """An average effect, with per-unit effects when the estimator provides them."""

    estimand: str
    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    n_units: int
    n_treated: int
    method: str
    cate: np.ndarray | None = field(default=None, repr=False)

    @property
    def heterogeneity(self) -> float:
        """Spread of the per-unit effects; zero when the effect is constant."""
        if self.cate is None:
            return float("nan")
        return float(np.std(self.cate, ddof=1))

    def to_dict(self) -> dict[str, float | str]:
        return {
            "estimand": self.estimand,
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "pvalue": self.pvalue,
            "n_units": float(self.n_units),
            "n_treated": float(self.n_treated),
            "method": self.method,
            "heterogeneity": self.heterogeneity,
        }


def _prepare(
    frame: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for column in (outcome, treatment):
        if column not in frame.columns:
            raise EstimationError(f"no column {column!r}")
    missing = [c for c in covariates if c not in frame.columns]
    if missing:
        raise EstimationError(f"missing covariates: {', '.join(missing)}")

    data = frame[[outcome, treatment, *covariates]].dropna()
    if data.empty:
        raise EstimationError("no complete rows after dropping missing values")

    y = data[outcome].to_numpy(dtype=np.float64)
    t = data[treatment].to_numpy(dtype=np.float64)
    X = data[list(covariates)].to_numpy(dtype=np.float64)
    if set(np.unique(t).tolist()) - {0.0, 1.0}:
        raise EstimationError("treatment must be binary 0/1")
    if t.sum() == 0 or t.sum() == t.size:
        raise EstimationError("both arms must contain units")
    return y, t, X


def double_machine_learning(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: list[str] | tuple[str, ...],
    n_folds: int = 5,
    n_estimators: int = 500,
    min_samples_leaf: int = 20,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> CausalEstimate:
    """Cross-fitted partially-linear DML estimate of the average effect.

    ``n_folds`` is the cross-fitting split. Two is enough for the theory; five
    is the usual choice because it wastes less data per fold. Setting it to one
    would mean no cross-fitting at all, which is the failure this method exists
    to avoid, so it is rejected.
    """
    if n_folds < 2:
        raise EstimationError("cross-fitting needs at least 2 folds")

    y, t, X = _prepare(frame, outcome, treatment, covariates)

    try:
        from econml.dml import LinearDML
    except ImportError as exc:  # pragma: no cover - a hard dependency
        raise EstimationError("econml is not installed; run `make install`") from exc

    estimator = LinearDML(
        model_y=GradientBoostingRegressor(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=seed
        ),
        model_t=GradientBoostingClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, random_state=seed
        ),
        discrete_treatment=True,
        cv=n_folds,
        random_state=seed,
    )
    estimator.fit(y, t, X=X)

    effect = float(np.mean(estimator.effect(X)))
    inference = estimator.effect_inference(X)
    population = inference.population_summary()
    se = float(np.asarray(population.stderr_mean).reshape(-1)[0])
    lower, upper = (
        float(np.asarray(b).reshape(-1)[0])
        for b in population.conf_int_mean(alpha=1.0 - confidence)
    )
    pvalue = float(np.asarray(population.pvalue()).reshape(-1)[0])

    result = CausalEstimate(
        estimand="ATE",
        estimate=effect,
        standard_error=se,
        ci_lower=lower,
        ci_upper=upper,
        pvalue=pvalue,
        n_units=int(y.size),
        n_treated=int(t.sum()),
        method=f"linear DML, {n_folds}-fold cross-fitting, gradient-boosted nuisances",
        cate=np.asarray(estimator.effect(X), dtype=np.float64),
    )
    logger.info(
        "dml_estimated", estimate=round(effect, 4), standard_error=round(se, 4), folds=n_folds
    )
    return result


def causal_forest(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: list[str] | tuple[str, ...],
    n_folds: int = 5,
    n_estimators: int = 500,
    min_samples_leaf: int = 20,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> CausalEstimate:
    """Causal forest: an average effect plus a per-unit effect for every row.

    The per-unit effects are the useful output. An average effect says whether
    to ship; the distribution says to whom, and a forest that finds no spread
    has told you something worth knowing - that targeting will not help.
    """
    y, t, X = _prepare(frame, outcome, treatment, covariates)

    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:  # pragma: no cover - a hard dependency
        raise EstimationError("econml is not installed; run `make install`") from exc

    estimator = CausalForestDML(
        model_y=GradientBoostingRegressor(
            n_estimators=200, min_samples_leaf=min_samples_leaf, random_state=seed
        ),
        model_t=GradientBoostingClassifier(
            n_estimators=200, min_samples_leaf=min_samples_leaf, random_state=seed
        ),
        discrete_treatment=True,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        cv=n_folds,
        random_state=seed,
    )
    estimator.fit(y, t, X=X)

    cate = np.asarray(estimator.effect(X), dtype=np.float64).reshape(-1)
    inference = estimator.effect_inference(X)
    population = inference.population_summary()
    effect = float(np.asarray(population.mean_point).reshape(-1)[0])
    se = float(np.asarray(population.stderr_mean).reshape(-1)[0])
    lower, upper = (
        float(np.asarray(b).reshape(-1)[0])
        for b in population.conf_int_mean(alpha=1.0 - confidence)
    )
    pvalue = float(np.asarray(population.pvalue()).reshape(-1)[0])

    result = CausalEstimate(
        estimand="ATE",
        estimate=effect,
        standard_error=se,
        ci_lower=lower,
        ci_upper=upper,
        pvalue=pvalue,
        n_units=int(y.size),
        n_treated=int(t.sum()),
        method=f"causal forest, {n_estimators} trees, {n_folds}-fold cross-fitting",
        cate=cate,
    )
    logger.info(
        "causal_forest_estimated",
        estimate=round(effect, 4),
        heterogeneity=round(result.heterogeneity, 4),
    )
    return result

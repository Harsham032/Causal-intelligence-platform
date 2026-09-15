"""Uplift learners.

An uplift model predicts a *difference* that is never observed. Every unit is
seen under one arm only, so there is no label to regress on - which is why a
response model trained on the treated group is not an uplift model, and why
substituting one is the most common mistake in the field. A response model
finds who converts; an uplift model finds who converts *because of* the
treatment, and the two rankings can be almost unrelated.

Three learners, in increasing sophistication:

*T-learner* fits one outcome model per arm and subtracts. Simple and
transparent; its weakness is that both models spend their capacity fitting the
baseline, which is usually far larger than the effect, so small effects are
swamped by independent errors from two models.

*S-learner* fits a single model with treatment as a feature. Shares the
baseline fit, but a flexible model can ignore the treatment feature entirely
and return a near-zero effect everywhere.

*DR-learner* (doubly robust) builds a pseudo-outcome that is unbiased if either
the outcome model or the propensity model is right, then regresses it on the
covariates. More moving parts, and the only one of the three that stays
consistent when one nuisance model is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

from ..errors import EstimationError
from ..logging_utils import get_logger

logger = get_logger(__name__)

Learner = Literal["t-learner", "s-learner", "dr-learner"]


@dataclass
class UpliftModel:
    """A fitted uplift model and the predictions it made on its training frame."""

    learner: Learner
    predicted_uplift: np.ndarray
    covariates: tuple[str, ...]
    n_units: int
    n_treated: int
    binary_outcome: bool
    model: Any = None

    @property
    def heterogeneity(self) -> float:
        return float(np.std(self.predicted_uplift, ddof=1))

    def to_dict(self) -> dict[str, float | str]:
        return {
            "learner": self.learner,
            "n_units": float(self.n_units),
            "n_treated": float(self.n_treated),
            "mean_predicted_uplift": float(self.predicted_uplift.mean()),
            "heterogeneity": self.heterogeneity,
        }


def _split(
    frame: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list[str] | tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    for column in (outcome, treatment):
        if column not in frame.columns:
            raise EstimationError(f"no column {column!r}")
    missing = [c for c in covariates if c not in frame.columns]
    if missing:
        raise EstimationError(f"missing covariates: {', '.join(missing)}")

    data = frame[[outcome, treatment, *covariates]].dropna()
    if data.empty:
        raise EstimationError("no complete rows")
    y = data[outcome].to_numpy(dtype=np.float64)
    t = data[treatment].to_numpy(dtype=np.int64)
    X = data[list(covariates)].to_numpy(dtype=np.float64)
    if set(np.unique(t).tolist()) - {0, 1}:
        raise EstimationError("treatment must be binary 0/1")
    if t.sum() == 0 or t.sum() == t.size:
        raise EstimationError("both arms must contain units")
    binary = bool(np.isin(y, (0.0, 1.0)).all())
    return y, t, X, binary


def _outcome_model(binary: bool, *, n_estimators: int, max_depth: int, lr: float, seed: int) -> Any:
    kwargs = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": lr,
        "random_state": seed,
    }
    return GradientBoostingClassifier(**kwargs) if binary else GradientBoostingRegressor(**kwargs)


def _predict(model: Any, X: np.ndarray, binary: bool) -> np.ndarray:
    if binary:
        return np.asarray(model.predict_proba(X))[:, 1]
    return np.asarray(model.predict(X), dtype=np.float64)


def fit_uplift(
    frame: pd.DataFrame,
    *,
    outcome: str,
    treatment: str,
    covariates: list[str] | tuple[str, ...],
    learner: Learner = "t-learner",
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    n_folds: int = 5,
    seed: int = 20260101,
) -> UpliftModel:
    """Fit an uplift model and predict each unit's treatment effect.

    Predictions are cross-fitted: every unit's uplift comes from models that did
    not see it. Without that, a flexible learner reproduces the noise in its own
    training rows and the resulting Qini curve is a measure of memorisation
    rather than of targeting.
    """
    y, t, X, binary = _split(frame, outcome, treatment, covariates)
    if n_folds < 2:
        raise EstimationError("cross-fitting needs at least 2 folds")

    from sklearn.model_selection import StratifiedKFold

    predictions = np.zeros(y.size, dtype=np.float64)
    # Stratify on the treatment-outcome pair so every fold keeps both arms and,
    # for a binary outcome, both classes within each arm.
    strata = t * 2 + (y > np.median(y)).astype(int) if not binary else t * 2 + y.astype(int)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_idx, test_idx in splitter.split(X, strata):
        X_tr, y_tr, t_tr = X[train_idx], y[train_idx], t[train_idx]
        X_te = X[test_idx]
        if t_tr.sum() == 0 or t_tr.sum() == t_tr.size:
            raise EstimationError("a cross-fitting fold lost one treatment arm")

        if learner == "t-learner":
            m1 = _outcome_model(
                binary, n_estimators=n_estimators, max_depth=max_depth, lr=learning_rate, seed=seed
            )
            m0 = _outcome_model(
                binary, n_estimators=n_estimators, max_depth=max_depth, lr=learning_rate, seed=seed
            )
            m1.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
            m0.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
            predictions[test_idx] = _predict(m1, X_te, binary) - _predict(m0, X_te, binary)

        elif learner == "s-learner":
            model = _outcome_model(
                binary, n_estimators=n_estimators, max_depth=max_depth, lr=learning_rate, seed=seed
            )
            model.fit(np.column_stack([X_tr, t_tr]), y_tr)
            ones = np.column_stack([X_te, np.ones(X_te.shape[0])])
            zeros = np.column_stack([X_te, np.zeros(X_te.shape[0])])
            predictions[test_idx] = _predict(model, ones, binary) - _predict(model, zeros, binary)

        elif learner == "dr-learner":
            predictions[test_idx] = _doubly_robust(
                X_tr,
                y_tr,
                t_tr,
                X_te,
                binary,
                n_estimators=n_estimators,
                max_depth=max_depth,
                lr=learning_rate,
                seed=seed,
            )
        else:
            raise EstimationError(f"unknown learner {learner!r}")

    model = UpliftModel(
        learner=learner,
        predicted_uplift=predictions,
        covariates=tuple(covariates),
        n_units=int(y.size),
        n_treated=int(t.sum()),
        binary_outcome=binary,
    )
    logger.info(
        "uplift_fitted",
        learner=learner,
        mean_uplift=round(float(predictions.mean()), 4),
        heterogeneity=round(model.heterogeneity, 4),
    )
    return model


def _doubly_robust(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    t_tr: np.ndarray,
    X_te: np.ndarray,
    binary: bool,
    *,
    n_estimators: int,
    max_depth: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    """Pseudo-outcome regression, unbiased if either nuisance model is right."""
    from sklearn.linear_model import LogisticRegression

    propensity = LogisticRegression(max_iter=2000, random_state=seed).fit(X_tr, t_tr)
    e = np.clip(np.asarray(propensity.predict_proba(X_tr))[:, 1], 0.02, 0.98)

    m1 = _outcome_model(binary, n_estimators=n_estimators, max_depth=max_depth, lr=lr, seed=seed)
    m0 = _outcome_model(binary, n_estimators=n_estimators, max_depth=max_depth, lr=lr, seed=seed)
    m1.fit(X_tr[t_tr == 1], y_tr[t_tr == 1])
    m0.fit(X_tr[t_tr == 0], y_tr[t_tr == 0])
    mu1, mu0 = _predict(m1, X_tr, binary), _predict(m0, X_tr, binary)

    # The augmented inverse-probability-weighted pseudo-outcome. Its conditional
    # expectation is the treatment effect, so regressing it on X estimates CATE.
    pseudo = (t_tr - e) / (e * (1.0 - e)) * (y_tr - np.where(t_tr == 1, mu1, mu0)) + mu1 - mu0
    final = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr, random_state=seed
    ).fit(X_tr, pseudo)
    return np.asarray(final.predict(X_te), dtype=np.float64)

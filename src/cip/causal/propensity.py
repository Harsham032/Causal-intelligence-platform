"""Propensity scores and the overlap they depend on.

A propensity score is the probability a unit received treatment given its
covariates. Conditioning on it makes treated and control units comparable -
*provided* two assumptions hold, and the second one is checkable:

**Unconfoundedness**: every covariate that drives both treatment and outcome is
in the model. This is not testable from the data. Nothing here pretends
otherwise; the sensitivity analysis in ``cip.evaluation`` exists because of it.

**Overlap**: for every covariate profile, both arms are possible. This *is*
checkable, and it fails routinely. If no control unit looks anything like a
given treated unit, no method can say what would have happened to that unit
without treatment - it can only extrapolate, and extrapolation dressed as an
estimate is the failure mode this module refuses to commit silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..errors import AssumptionError, EstimationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PropensityResult:
    """Fitted propensity scores and how well they separate the arms."""

    scores: np.ndarray
    model: object
    auc: float
    covariates: tuple[str, ...]
    n_treated: int
    n_control: int

    def to_dict(self) -> dict[str, float]:
        return {
            "auc": self.auc,
            "n_treated": float(self.n_treated),
            "n_control": float(self.n_control),
            "min_score": float(self.scores.min()),
            "max_score": float(self.scores.max()),
        }


@dataclass
class OverlapReport:
    """Whether the two arms occupy the same covariate space at all."""

    treated_min: float
    treated_max: float
    control_min: float
    control_max: float
    common_lower: float
    common_upper: float
    treated_outside: int
    control_outside: int
    n_trimmed: int
    share_trimmed: float
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "common_lower": self.common_lower,
            "common_upper": self.common_upper,
            "treated_outside": float(self.treated_outside),
            "control_outside": float(self.control_outside),
            "n_trimmed": float(self.n_trimmed),
            "share_trimmed": self.share_trimmed,
            "passed": self.passed,
            "detail": self.detail,
        }


def estimate_propensity(
    frame: pd.DataFrame,
    treatment: str,
    covariates: list[str] | tuple[str, ...],
    *,
    model: str = "logistic",
    seed: int = 20260101,
) -> PropensityResult:
    """Fit the probability of treatment given covariates.

    A high AUC is **not** a success here, which is the opposite of the intuition
    people bring from prediction. It means treatment is close to deterministic
    given the covariates, and therefore that some units had almost no chance of
    landing in one arm - an overlap failure. The AUC is reported so that this is
    visible rather than buried.
    """
    missing = [c for c in covariates if c not in frame.columns]
    if missing:
        raise EstimationError(f"missing covariates: {', '.join(missing)}")
    if treatment not in frame.columns:
        raise EstimationError(f"no treatment column {treatment!r}")

    X = frame[list(covariates)].to_numpy(dtype=np.float64)
    y = frame[treatment].to_numpy(dtype=np.int64)
    if set(np.unique(y).tolist()) - {0, 1}:
        raise EstimationError("treatment must be binary 0/1")
    n_treated, n_control = int(y.sum()), int((1 - y).sum())
    if n_treated == 0 or n_control == 0:
        raise EstimationError("both arms must contain units")

    if model == "logistic":
        estimator: object = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
    elif model == "gradient-boosting":
        estimator = GradientBoostingClassifier(random_state=seed)
    else:
        raise EstimationError(f"unknown propensity model {model!r}")

    estimator.fit(X, y)  # type: ignore[attr-defined]
    scores = np.asarray(estimator.predict_proba(X))[:, 1]  # type: ignore[attr-defined]

    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(y, scores))
    logger.info(
        "propensity_fitted",
        model=model,
        auc=round(auc, 4),
        n_treated=n_treated,
        n_control=n_control,
    )
    return PropensityResult(
        scores=scores,
        model=estimator,
        auc=auc,
        covariates=tuple(covariates),
        n_treated=n_treated,
        n_control=n_control,
    )


def check_overlap(
    scores: np.ndarray,
    treatment: np.ndarray,
    *,
    trim: tuple[float, float] = (0.01, 0.99),
    max_trimmed_share: float = 0.5,
    strict: bool = True,
) -> OverlapReport:
    """Test whether the arms share a covariate region, and refuse if they do not.

    Units whose propensity falls outside the region both arms occupy have no
    counterfactual in the data. Keeping them means extrapolating; the estimate
    would still be a number, and it would still be reported to three decimals.

    Raises :class:`AssumptionError` when trimming would discard more than
    ``max_trimmed_share`` of the sample, because at that point the estimand has
    quietly become 'the effect among the units that happened to overlap', which
    is not what anyone asked for.

    ``strict=False`` reports the failure instead of raising, for callers that
    intend to trim deliberately and relabel the estimand. That is a legitimate
    analysis - it is what the LaLonde literature does - but it must be a choice
    someone made, not a default that hides the problem.
    """
    p = np.asarray(scores, dtype=np.float64)
    t = np.asarray(treatment, dtype=np.int64)
    if p.shape != t.shape:
        raise EstimationError("scores and treatment must align")

    treated, control = p[t == 1], p[t == 0]
    if treated.size == 0 or control.size == 0:
        raise EstimationError("both arms must contain units")

    # The region both arms actually occupy, intersected with the configured trim.
    common_lower = max(treated.min(), control.min(), trim[0])
    common_upper = min(treated.max(), control.max(), trim[1])

    inside = (p >= common_lower) & (p <= common_upper)
    n_trimmed = int((~inside).sum())
    share = n_trimmed / p.size

    treated_outside = int(((t == 1) & ~inside).sum())
    control_outside = int(((t == 0) & ~inside).sum())

    if common_lower >= common_upper and strict:
        raise AssumptionError(
            "the arms share no common propensity region: treated scores span "
            f"[{treated.min():.4f}, {treated.max():.4f}] and controls "
            f"[{control.min():.4f}, {control.max():.4f}]. No estimator can "
            "compare units that have no counterpart in the other arm."
        )
    if share > max_trimmed_share and strict:
        raise AssumptionError(
            f"enforcing overlap would discard {share:.1%} of the sample "
            f"({n_trimmed} of {p.size}). The remaining units are not the "
            "population the question was about; report the overlap failure "
            "rather than an effect estimated on what is left."
        )

    detail = (
        f"common support [{common_lower:.4f}, {common_upper:.4f}]; "
        f"{n_trimmed} of {p.size} units ({share:.2%}) outside it "
        f"({treated_outside} treated, {control_outside} control)"
    )
    report = OverlapReport(
        treated_min=float(treated.min()),
        treated_max=float(treated.max()),
        control_min=float(control.min()),
        control_max=float(control.max()),
        common_lower=float(common_lower),
        common_upper=float(common_upper),
        treated_outside=treated_outside,
        control_outside=control_outside,
        n_trimmed=n_trimmed,
        share_trimmed=float(share),
        passed=share <= 0.10,
        detail=detail,
    )
    logger.info("overlap_checked", share_trimmed=round(share, 4), passed=report.passed)
    return report


def trim_to_overlap(
    scores: np.ndarray,
    treatment: np.ndarray,
    *,
    trim: tuple[float, float] = (0.01, 0.99),
) -> tuple[np.ndarray, OverlapReport]:
    """Restrict a sample to the region both arms occupy.

    Returns a boolean mask and the overlap report describing what was dropped.

    This changes the estimand, and the change is not cosmetic. After trimming,
    an estimate is the effect **among units that could plausibly have received
    either arm** - not the effect on the original population. On the LaLonde
    benchmark that is the difference between a question about disadvantaged
    trainees and a question about the US labour force, and no amount of
    statistical machinery bridges it.

    Callers are expected to relabel what they report accordingly, which is why
    this is a separate, explicit step rather than something the estimators do
    for themselves.
    """
    report = check_overlap(scores, treatment, trim=trim, strict=False)
    p = np.asarray(scores, dtype=np.float64)
    mask = (p >= report.common_lower) & (p <= report.common_upper)
    logger.info(
        "trimmed_to_overlap",
        kept=int(mask.sum()),
        dropped=int((~mask).sum()),
        share_dropped=round(report.share_trimmed, 4),
    )
    return mask, report

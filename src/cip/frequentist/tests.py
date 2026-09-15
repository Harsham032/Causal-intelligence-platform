"""Two-arm comparisons.

A p-value answers one narrow question: how surprising is this data if the
treatment did nothing? It says nothing about how large the effect is, and an
experiment that reports only significance has thrown away the number the
decision actually depends on. Every result here therefore carries the effect,
its interval and the arm sizes, and the p-value is the least prominent field.

Welch's test is the default for means rather than Student's. Equal variances
across arms is an assumption nobody checks and the treatment often violates -
a change that helps some users and not others moves the variance as well as the
mean - and Welch costs nothing when variances happen to be equal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ..errors import EstimationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class TestResult:
    """A two-arm comparison, reported effect first."""

    metric: str
    treated_mean: float
    control_mean: float
    absolute_effect: float
    relative_effect: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    n_treated: int
    n_control: int
    method: str
    confidence: float = 0.95

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero, at the stated confidence."""
        return self.ci_lower > 0.0 or self.ci_upper < 0.0

    def to_dict(self) -> dict[str, float | str | int | bool]:
        return {
            "metric": self.metric,
            "treated_mean": self.treated_mean,
            "control_mean": self.control_mean,
            "absolute_effect": self.absolute_effect,
            "relative_effect": self.relative_effect,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "pvalue": self.pvalue,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "method": self.method,
            "significant": self.significant,
        }


def _arms(treated: ArrayLike, control: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(treated, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    if t.size < 2 or c.size < 2:
        raise EstimationError("each arm needs at least two observations")
    if not np.isfinite(t).all() or not np.isfinite(c).all():
        raise EstimationError("arms contain non-finite values")
    return t, c


def _relative(effect: float, control_mean: float) -> float:
    return float(effect / control_mean) if control_mean != 0 else float("nan")


def difference_in_means(
    treated: ArrayLike,
    control: ArrayLike,
    *,
    metric: str = "outcome",
    confidence: float = 0.95,
) -> TestResult:
    """Welch's unequal-variance comparison of two arm means."""
    t, c = _arms(treated, control)
    effect = float(t.mean() - c.mean())
    se = float(np.sqrt(t.var(ddof=1) / t.size + c.var(ddof=1) / c.size))
    if se == 0.0:
        raise EstimationError("both arms are constant; no interval is defined")

    # Welch-Satterthwaite degrees of freedom.
    vt, vc = t.var(ddof=1) / t.size, c.var(ddof=1) / c.size
    df = (vt + vc) ** 2 / (vt**2 / (t.size - 1) + vc**2 / (c.size - 1))
    critical = float(stats.t.ppf(0.5 + confidence / 2.0, df))
    pvalue = float(2.0 * stats.t.sf(abs(effect / se), df))

    return TestResult(
        metric=metric,
        treated_mean=float(t.mean()),
        control_mean=float(c.mean()),
        absolute_effect=effect,
        relative_effect=_relative(effect, float(c.mean())),
        ci_lower=effect - critical * se,
        ci_upper=effect + critical * se,
        pvalue=pvalue,
        n_treated=int(t.size),
        n_control=int(c.size),
        method="welch t-test",
        confidence=confidence,
    )


def difference_in_proportions(
    treated: ArrayLike,
    control: ArrayLike,
    *,
    metric: str = "conversion",
    confidence: float = 0.95,
) -> TestResult:
    """Compare two conversion rates.

    The interval uses the Newcombe hybrid-score method rather than the normal
    approximation. Near zero - where conversion rates usually are - the normal
    interval can extend below zero, which is not a possible value for a rate
    difference bounded by the rates themselves.
    """
    t, c = _arms(treated, control)
    for name, arm in (("treated", t), ("control", c)):
        if not np.isin(arm, (0.0, 1.0)).all():
            raise EstimationError(f"the {name} arm must be binary 0/1 for a proportion test")

    nt, nc = int(t.size), int(c.size)
    st, sc = float(t.sum()), float(c.sum())
    pt, pc = st / nt, sc / nc
    effect = pt - pc

    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    lo_t, hi_t = _wilson(st, nt, z)
    lo_c, hi_c = _wilson(sc, nc, z)
    # Newcombe: combine each arm's score interval rather than pooling variances.
    ci_lower = effect - np.sqrt((pt - lo_t) ** 2 + (hi_c - pc) ** 2)
    ci_upper = effect + np.sqrt((hi_t - pt) ** 2 + (pc - lo_c) ** 2)

    pooled = (st + sc) / (nt + nc)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / nt + 1 / nc))
    pvalue = float(2.0 * stats.norm.sf(abs(effect / se_pooled))) if se_pooled > 0 else 1.0

    return TestResult(
        metric=metric,
        treated_mean=pt,
        control_mean=pc,
        absolute_effect=float(effect),
        relative_effect=_relative(float(effect), pc),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        pvalue=pvalue,
        n_treated=nt,
        n_control=nc,
        method="newcombe hybrid score",
        confidence=confidence,
    )


def _wilson(successes: float, n: int, z: float) -> tuple[float, float]:
    """Wilson score interval for one proportion."""
    p = successes / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
    return float(centre - half), float(centre + half)


def bootstrap_difference(
    treated: ArrayLike,
    control: ArrayLike,
    *,
    metric: str = "outcome",
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> TestResult:
    """Percentile bootstrap of the difference in means.

    Worth reaching for when the outcome is heavily skewed - revenue per user
    usually is - because the t-interval's accuracy then depends on a normality
    that the data plainly does not have.
    """
    t, c = _arms(treated, control)
    if resamples < 100:
        raise EstimationError("a bootstrap needs at least 100 resamples to be meaningful")

    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        draws[i] = (
            rng.choice(t, size=t.size, replace=True).mean()
            - rng.choice(c, size=c.size, replace=True).mean()
        )

    effect = float(t.mean() - c.mean())
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(draws, [alpha, 1.0 - alpha])
    # Two-sided bootstrap p-value: how much of the distribution sits past zero.
    tail = float(min((draws <= 0).mean(), (draws >= 0).mean()))
    return TestResult(
        metric=metric,
        treated_mean=float(t.mean()),
        control_mean=float(c.mean()),
        absolute_effect=effect,
        relative_effect=_relative(effect, float(c.mean())),
        ci_lower=float(lower),
        ci_upper=float(upper),
        pvalue=min(1.0, 2.0 * tail),
        n_treated=int(t.size),
        n_control=int(c.size),
        method=f"percentile bootstrap ({resamples} resamples)",
        confidence=confidence,
    )

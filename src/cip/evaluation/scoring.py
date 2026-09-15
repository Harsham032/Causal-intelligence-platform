"""Scoring an estimator against a known answer.

Only possible where a truth exists: a simulation that wrote down both potential
outcomes, or a benchmark where a randomised experiment supplies the answer an
observational method should have found.

Two properties are worth measuring, and they are not the same thing.

**Bias** is how far the estimate lands from the truth on average. An estimator
can be badly biased and produce narrow, confident intervals - the confidence is
then a statement about precision, not accuracy.

**Coverage** is how often the interval actually contains the truth. A nominal
95% interval that covers 70% of the time is not conservative, it is wrong, and
nothing about the point estimate reveals it. Coverage needs repetition, which
is why it can only be measured in simulation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..errors import EvaluationError


@dataclass
class BiasReport:
    """How far a set of estimates sits from a known truth."""

    truth: float
    estimates: tuple[float, ...]
    bias: float
    absolute_bias: float
    relative_bias: float
    rmse: float
    n_runs: int

    def to_dict(self) -> dict[str, float]:
        return {
            "truth": self.truth,
            "mean_estimate": float(np.mean(self.estimates)),
            "bias": self.bias,
            "absolute_bias": self.absolute_bias,
            "relative_bias": self.relative_bias,
            "rmse": self.rmse,
            "n_runs": float(self.n_runs),
        }


@dataclass
class CoverageReport:
    """How often a nominal interval contains the truth."""

    nominal: float
    empirical: float
    n_runs: int
    mean_width: float
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "nominal": self.nominal,
            "empirical": self.empirical,
            "n_runs": float(self.n_runs),
            "mean_width": self.mean_width,
            "passed": self.passed,
            "detail": self.detail,
        }


def estimator_bias(estimates: ArrayLike, truth: float) -> BiasReport:
    """Bias and root mean squared error of repeated estimates."""
    e = np.asarray(estimates, dtype=np.float64).reshape(-1)
    if e.size == 0:
        raise EvaluationError("no estimates to score")
    bias = float(e.mean() - truth)
    return BiasReport(
        truth=float(truth),
        estimates=tuple(float(x) for x in e),
        bias=bias,
        absolute_bias=abs(bias),
        relative_bias=float(bias / truth) if truth != 0 else float("nan"),
        rmse=float(np.sqrt(((e - truth) ** 2).mean())),
        n_runs=int(e.size),
    )


def coverage_of_intervals(
    lowers: ArrayLike,
    uppers: ArrayLike,
    truth: float,
    *,
    nominal: float = 0.95,
    tolerance: float = 0.03,
) -> CoverageReport:
    """Measure how often intervals contain the truth, against their claim.

    ``tolerance`` is the margin allowed before coverage is called wrong. With a
    few hundred runs the empirical rate is itself noisy, so an exact match is
    not expected; a systematic shortfall is what matters.
    """
    lo = np.asarray(lowers, dtype=np.float64).reshape(-1)
    hi = np.asarray(uppers, dtype=np.float64).reshape(-1)
    if lo.shape != hi.shape:
        raise EvaluationError("interval bounds must align")
    if lo.size == 0:
        raise EvaluationError("no intervals to score")
    if np.any(lo > hi):
        raise EvaluationError("some intervals have reversed bounds")

    covered = (lo <= truth) & (truth <= hi)
    empirical = float(covered.mean())
    shortfall = nominal - empirical
    passed = abs(shortfall) <= tolerance

    if passed:
        detail = f"{empirical:.3f} against a nominal {nominal:.2f}, within {tolerance:.2f}"
    elif shortfall > 0:
        detail = (
            f"{empirical:.3f} against a nominal {nominal:.2f}: the intervals are too narrow, "
            "so reported precision overstates what the estimator knows"
        )
    else:
        detail = (
            f"{empirical:.3f} against a nominal {nominal:.2f}: the intervals are wider than "
            "necessary, which costs power but does not mislead"
        )

    return CoverageReport(
        nominal=float(nominal),
        empirical=empirical,
        n_runs=int(lo.size),
        mean_width=float((hi - lo).mean()),
        passed=passed,
        detail=detail,
    )


def bias_table(results: dict[str, tuple[float, float, float]], truth: float) -> pd.DataFrame:
    """Tabulate ``{method: (estimate, ci_lower, ci_upper)}`` against a truth."""
    rows = []
    for method, (estimate, lower, upper) in results.items():
        rows.append(
            {
                "method": method,
                "estimate": estimate,
                "truth": truth,
                "bias": estimate - truth,
                "relative_bias": (estimate - truth) / truth if truth != 0 else float("nan"),
                "ci_lower": lower,
                "ci_upper": upper,
                "covers_truth": bool(lower <= truth <= upper),
            }
        )
    return pd.DataFrame(rows)

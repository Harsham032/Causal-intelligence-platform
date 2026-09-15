"""Randomisation and covariate balance checks.

Randomisation is a claim about the assignment mechanism, and claims can be
wrong: a bug in the bucketing, a filter applied after assignment, a redirect
that fails for one arm. These checks test the claim against the data before any
effect is reported, because an imbalanced experiment is an observational study
wearing an experiment's clothing.

The statistic is the standardised mean difference, not a t-test p-value. With a
large sample a trivial imbalance is "significant", and with a small one a
serious imbalance is not - so the p-value answers a question nobody asked. The
standardised difference measures the size of the imbalance in standard
deviations, and the conventional action threshold of 0.10 does not move with n.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ..errors import DesignError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CovariateBalance:
    """How far apart the arms are on one covariate."""

    covariate: str
    treated_mean: float
    control_mean: float
    standardised_difference: float
    balanced: bool
    n_missing: int = 0

    @property
    def computable(self) -> bool:
        """False when missing data leaves nothing to compare."""
        return bool(np.isfinite(self.standardised_difference))

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "covariate": self.covariate,
            "treated_mean": self.treated_mean,
            "control_mean": self.control_mean,
            "standardised_difference": self.standardised_difference,
            "balanced": self.balanced,
            "n_missing": float(self.n_missing),
            "computable": self.computable,
        }


@dataclass
class BalanceReport:
    """Balance across every covariate, plus the assignment-share check."""

    covariates: list[CovariateBalance]
    n_treated: int
    n_control: int
    expected_share: float
    share_pvalue: float
    threshold: float

    @property
    def imbalanced(self) -> list[CovariateBalance]:
        return [c for c in self.covariates if c.computable and not c.balanced]

    @property
    def not_computable(self) -> list[CovariateBalance]:
        """Covariates whose balance could not be measured, usually missing data."""
        return [c for c in self.covariates if not c.computable]

    @property
    def passed(self) -> bool:
        return not self.imbalanced and not self.not_computable and self.share_pvalue >= 0.01

    @property
    def worst(self) -> CovariateBalance | None:
        """The largest measurable imbalance; ignores covariates that could not be computed."""
        measurable = [c for c in self.covariates if c.computable]
        if not measurable:
            return None
        return max(measurable, key=lambda c: abs(c.standardised_difference))

    def reason(self) -> str:
        if self.passed:
            return "arms are balanced on every covariate and on assignment share"
        problems = []
        if self.share_pvalue < 0.01:
            problems.append(
                f"assignment share {self.n_treated / (self.n_treated + self.n_control):.4f} "
                f"differs from the expected {self.expected_share:.4f} (p={self.share_pvalue:.3g})"
            )
        if self.imbalanced:
            names = ", ".join(
                f"{c.covariate} ({c.standardised_difference:+.3f})" for c in self.imbalanced
            )
            problems.append(f"imbalanced beyond {self.threshold:.2f}: {names}")
        if self.not_computable:
            names = ", ".join(f"{c.covariate} ({c.n_missing} missing)" for c in self.not_computable)
            problems.append(f"balance could not be measured: {names}")
        return "; ".join(problems)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.covariates])


def standardised_difference(treated: np.ndarray, control: np.ndarray) -> float:
    """Difference in means, in pooled standard deviations.

    Uses the pooled standard deviation of the two arms rather than either arm's
    own, so the statistic does not change when the arms are relabelled.
    """
    t = np.asarray(treated, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    if t.size == 0 or c.size == 0:
        raise DesignError("both arms must contain observations")
    pooled = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2.0)
    if pooled == 0.0:
        # Constant in both arms: either identical (balanced) or a hard split.
        return 0.0 if t.mean() == c.mean() else float("inf")
    return float((t.mean() - c.mean()) / pooled)


def check_balance(
    frame: pd.DataFrame,
    treatment: str,
    covariates: list[str] | tuple[str, ...],
    *,
    threshold: float = 0.10,
    expected_share: float = 0.5,
) -> BalanceReport:
    """Test the randomisation claim against the observed assignment."""
    if treatment not in frame.columns:
        raise DesignError(f"no treatment column {treatment!r}")
    missing = [c for c in covariates if c not in frame.columns]
    if missing:
        raise DesignError(f"missing covariates: {', '.join(missing)}")

    arm = frame[treatment].to_numpy()
    unique = set(np.unique(arm).tolist())
    if not unique <= {0, 1}:
        raise DesignError(f"treatment must be binary 0/1; found {sorted(unique)}")
    treated_mask = arm == 1
    n_treated, n_control = int(treated_mask.sum()), int((~treated_mask).sum())
    if n_treated == 0 or n_control == 0:
        raise DesignError("both arms must contain units")

    results: list[CovariateBalance] = []
    for name in covariates:
        values = frame[name].to_numpy(dtype=np.float64)
        observed = np.isfinite(values)
        n_missing = int((~observed).sum())

        treated_values = values[treated_mask & observed]
        control_values = values[~treated_mask & observed]
        # Missing data is reported as such rather than silently producing a NaN
        # that then reads as "imbalanced". Those are different findings: one says
        # the arms differ, the other says nobody can tell.
        if treated_values.size < 2 or control_values.size < 2:
            results.append(
                CovariateBalance(
                    covariate=name,
                    treated_mean=float("nan"),
                    control_mean=float("nan"),
                    standardised_difference=float("nan"),
                    balanced=False,
                    n_missing=n_missing,
                )
            )
            continue

        smd = standardised_difference(treated_values, control_values)
        results.append(
            CovariateBalance(
                covariate=name,
                treated_mean=float(treated_values.mean()),
                control_mean=float(control_values.mean()),
                standardised_difference=smd,
                balanced=bool(np.isfinite(smd)) and abs(smd) <= threshold,
                n_missing=n_missing,
            )
        )

    # A binomial test on the split itself catches a bucketing bug that leaves
    # every covariate balanced but the arms the wrong size.
    share_pvalue = float(stats.binomtest(n_treated, n_treated + n_control, expected_share).pvalue)

    report = BalanceReport(
        covariates=results,
        n_treated=n_treated,
        n_control=n_control,
        expected_share=expected_share,
        share_pvalue=share_pvalue,
        threshold=threshold,
    )
    logger.info(
        "balance_checked",
        covariates=len(results),
        imbalanced=len(report.imbalanced),
        not_computable=len(report.not_computable),
        share_pvalue=round(share_pvalue, 4),
        passed=report.passed,
    )
    return report

"""Sensitivity to unmeasured confounding.

Every observational estimate assumes that nothing outside the covariate list
drives both treatment and outcome. The assumption is not testable - if the
confounder were measured it would not be unmeasured - so the honest question is
not "is it true?" but "how wrong would it have to be to overturn this?"

An estimate that survives only if no hidden factor exists is fragile. One that
survives unless a hidden factor is stronger than every measured covariate is
robust. Both can have the same point estimate and the same interval, and
nothing else in a standard write-up distinguishes them.

Two complementary views:

*Rosenbaum bounds* ask how much the odds of treatment would have to differ
between two otherwise-identical units before the result loses significance.

*Confounding strength* asks how strongly a hidden variable would have to relate
to both treatment and outcome to explain the effect away entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ..errors import EvaluationError


@dataclass
class SensitivityReport:
    """How much hidden confounding an estimate can absorb."""

    estimate: float
    breaking_point: float
    interpretation: str
    robust: bool
    method: str
    grid: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "estimate": self.estimate,
            "breaking_point": self.breaking_point,
            "robust": self.robust,
            "method": self.method,
            "interpretation": self.interpretation,
        }


def rosenbaum_bounds(
    treated_outcomes: ArrayLike,
    control_outcomes: ArrayLike,
    *,
    gammas: tuple[float, ...] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
    alpha: float = 0.05,
) -> SensitivityReport:
    """How much hidden bias a matched-pair result can absorb before it breaks.

    Gamma is the factor by which two units identical on measured covariates
    could differ in their odds of treatment. Gamma 1 is the no-hidden-bias case.
    The reported breaking point is the smallest gamma at which the result stops
    being significant.

    Expects matched pairs: ``treated_outcomes[i]`` and ``control_outcomes[i]``
    are the two halves of one pair.
    """
    t = np.asarray(treated_outcomes, dtype=np.float64).reshape(-1)
    c = np.asarray(control_outcomes, dtype=np.float64).reshape(-1)
    if t.shape != c.shape:
        raise EvaluationError("matched pairs must align")
    if t.size < 5:
        raise EvaluationError("too few pairs for a sensitivity analysis")

    differences = t - c
    nonzero = differences[differences != 0]
    if nonzero.size == 0:
        raise EvaluationError("every pair is tied; there is nothing to test")

    n = nonzero.size
    positive = int((nonzero > 0).sum())

    breaking_point = float("inf")
    grid: list[tuple[float, float]] = []
    for gamma in gammas:
        # Under hidden bias of size gamma, the probability a pair's difference
        # is positive lies in [1/(1+gamma), gamma/(1+gamma)]. The worst case for
        # the result is the upper bound, so the test uses that.
        p_upper = gamma / (1.0 + gamma)
        pvalue = float(stats.binomtest(positive, n, p_upper, alternative="greater").pvalue)
        grid.append((float(gamma), pvalue))
        if (
            pvalue > alpha
            and not np.isfinite(breaking_point)
            or pvalue > alpha
            and gamma < breaking_point
        ):
            breaking_point = float(gamma)

    robust = breaking_point >= 1.5
    if not np.isfinite(breaking_point):
        interpretation = (
            f"significant at every gamma tested up to {max(gammas):.2f}: a hidden factor would "
            f"have to more than {max(gammas):.1f}x the odds of treatment to overturn this"
        )
        robust = True
    elif breaking_point <= 1.1:
        interpretation = (
            f"breaks at gamma {breaking_point:.2f}: a hidden factor shifting the odds of "
            "treatment by a tenth would be enough to explain the result away"
        )
    else:
        interpretation = (
            f"breaks at gamma {breaking_point:.2f}: it survives hidden bias up to that "
            "strength but not beyond"
        )

    return SensitivityReport(
        estimate=float(differences.mean()),
        breaking_point=breaking_point,
        interpretation=interpretation,
        robust=robust,
        method="rosenbaum bounds (sign test on matched pairs)",
        grid=tuple(grid),
    )


def unmeasured_confounding(
    estimate: float,
    standard_error: float,
    *,
    confidence: float = 0.95,
) -> SensitivityReport:
    """How strong a hidden confounder must be to reduce the effect to zero.

    Reported as the number of standard errors of bias required. A result whose
    estimate sits two standard errors from zero needs a confounder worth two
    standard errors to erase; one sitting ten away needs five times as much,
    and is correspondingly harder to dismiss.
    """
    if standard_error <= 0:
        raise EvaluationError("standard_error must be positive")

    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    to_zero = abs(estimate) / standard_error
    to_insignificance = max(0.0, to_zero - z)

    robust = to_insignificance >= 2.0
    interpretation = (
        f"a hidden confounder would have to supply {to_insignificance:.2f} standard errors "
        f"of bias to render this insignificant, and {to_zero:.2f} to erase it entirely"
    )
    return SensitivityReport(
        estimate=float(estimate),
        breaking_point=float(to_insignificance),
        interpretation=interpretation,
        robust=robust,
        method="standard-error bias equivalence",
    )

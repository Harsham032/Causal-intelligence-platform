"""Sample size and power.

The calculation that decides whether an experiment is worth running at all. An
underpowered test does not return "no effect"; it returns noise, and reading
that noise as evidence of no effect is the most common way an experimentation
programme talks itself out of a real improvement.

Everything here is two-arm and fixed-horizon. Sequential testing changes the
error rates and is deliberately not pretended at: peeking at a fixed-horizon
test inflates the false positive rate well beyond its nominal alpha, and the
honest fix is a design built for it, not a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from ..errors import DesignError


@dataclass(frozen=True)
class SampleSizeResult:
    """How many units each arm needs, and what was assumed to get there."""

    per_arm: int
    total: int
    effect_size: float
    alpha: float
    power: float
    two_sided: bool
    detail: str

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "per_arm": float(self.per_arm),
            "total": float(self.total),
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "power": self.power,
            "two_sided": self.two_sided,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PowerResult:
    """The probability of detecting a given effect at a given sample size."""

    power: float
    per_arm: int
    effect_size: float
    alpha: float
    two_sided: bool

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "power": self.power,
            "per_arm": float(self.per_arm),
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "two_sided": self.two_sided,
        }


def _critical_z(alpha: float, two_sided: bool) -> float:
    return float(stats.norm.ppf(1.0 - (alpha / 2.0 if two_sided else alpha)))


def _validate(alpha: float, power: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise DesignError("alpha must lie strictly between 0 and 1")
    if not 0.0 < power < 1.0:
        raise DesignError("power must lie strictly between 0 and 1")


def sample_size_for_means(
    baseline_mean: float,
    baseline_sd: float,
    relative_mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> SampleSizeResult:
    """Units per arm to detect a relative change in a continuous mean.

    The minimum detectable effect is expressed relative to the baseline because
    an absolute one is uninterpretable until you know the outcome's scale: a
    one-dollar lift is enormous on a one-dollar basket and invisible on a
    thousand-dollar one.
    """
    _validate(alpha, power)
    if baseline_sd <= 0:
        raise DesignError("baseline_sd must be positive")
    if relative_mde <= 0:
        raise DesignError("relative_mde must be positive")
    if baseline_mean == 0:
        raise DesignError("a relative effect is undefined when the baseline mean is zero")

    absolute_mde = abs(baseline_mean) * relative_mde
    effect_size = absolute_mde / baseline_sd  # Cohen's d
    z_alpha = _critical_z(alpha, two_sided)
    z_beta = float(stats.norm.ppf(power))
    per_arm = math.ceil(2.0 * ((z_alpha + z_beta) / effect_size) ** 2)
    return SampleSizeResult(
        per_arm=per_arm,
        total=2 * per_arm,
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        two_sided=two_sided,
        detail=(
            f"detect a {relative_mde:.1%} change on a baseline mean of {baseline_mean:,.4g} "
            f"(sd {baseline_sd:,.4g}), i.e. an absolute shift of {absolute_mde:,.4g}"
        ),
    )


def sample_size_for_proportions(
    baseline_rate: float,
    relative_mde: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> SampleSizeResult:
    """Units per arm to detect a relative change in a conversion rate.

    Uses the arcsine transform, whose variance does not depend on the rate. The
    pooled-variance formula is more familiar but understates the requirement for
    rates near zero - which is where most conversion experiments live.
    """
    _validate(alpha, power)
    if not 0.0 < baseline_rate < 1.0:
        raise DesignError("baseline_rate must lie strictly between 0 and 1")
    if relative_mde <= 0:
        raise DesignError("relative_mde must be positive")

    treated_rate = baseline_rate * (1.0 + relative_mde)
    if not 0.0 < treated_rate < 1.0:
        raise DesignError(
            f"a {relative_mde:.1%} lift on {baseline_rate:.3f} leaves the unit interval"
        )

    effect_size = abs(
        2.0 * math.asin(math.sqrt(treated_rate)) - 2.0 * math.asin(math.sqrt(baseline_rate))
    )
    z_alpha = _critical_z(alpha, two_sided)
    z_beta = float(stats.norm.ppf(power))
    per_arm = math.ceil(2.0 * ((z_alpha + z_beta) / effect_size) ** 2)
    return SampleSizeResult(
        per_arm=per_arm,
        total=2 * per_arm,
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        two_sided=two_sided,
        detail=(
            f"detect a move from {baseline_rate:.4f} to {treated_rate:.4f} "
            f"({relative_mde:.1%} relative)"
        ),
    )


def power_for_sample_size(
    effect_size: float,
    per_arm: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> PowerResult:
    """The power a two-arm test actually has at a given size.

    Run this on an experiment that has already concluded. A null result at 30%
    power is not evidence of no effect; it is evidence that the experiment could
    not have found one.
    """
    if per_arm < 2:
        raise DesignError("per_arm must be at least 2")
    if effect_size <= 0:
        raise DesignError("effect_size must be positive")
    if not 0.0 < alpha < 1.0:
        raise DesignError("alpha must lie strictly between 0 and 1")

    z_alpha = _critical_z(alpha, two_sided)
    lambda_ = effect_size * math.sqrt(per_arm / 2.0)
    power = float(stats.norm.cdf(lambda_ - z_alpha))
    if two_sided:
        power += float(stats.norm.cdf(-lambda_ - z_alpha))
    return PowerResult(
        power=min(power, 1.0),
        per_arm=per_arm,
        effect_size=effect_size,
        alpha=alpha,
        two_sided=two_sided,
    )


def minimum_detectable_effect(
    per_arm: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """The smallest standardised effect a given sample size can detect.

    The question to ask *before* running: if the true effect were smaller than
    this, the experiment would probably miss it. If that is a size worth acting
    on, the experiment is not big enough.
    """
    _validate(alpha, power)
    if per_arm < 2:
        raise DesignError("per_arm must be at least 2")
    z_alpha = _critical_z(alpha, two_sided)
    z_beta = float(stats.norm.ppf(power))
    return float((z_alpha + z_beta) * math.sqrt(2.0 / per_arm))

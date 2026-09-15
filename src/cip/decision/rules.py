"""Decision rules.

An estimate is not a decision. Shipping a change costs engineering time,
support load and the risk of being wrong, and none of that appears in a
confidence interval. This module makes the missing step explicit: a
recommendation names the threshold it was judged against, so that disagreeing
with it means disagreeing with a stated number rather than with a vibe.

Two gates, and both must pass:

**Credibility** - is the effect distinguishable from nothing? Either the
interval excludes zero, or the posterior probability of benefit clears the
configured bar.

**Materiality** - is it large enough to be worth the cost? A real 0.01%
improvement is still a real improvement and still not worth a migration. This
is the gate that statistical significance cannot supply, because significance
is a statement about evidence and materiality is a statement about value.

A result that is credible but immaterial gets its own verdict rather than being
lumped in with a failure. The two call for different follow-ups: one says the
idea does not work, the other says it works and does not matter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..errors import EstimationError

Verdict = Literal["ship", "do not ship", "credible but immaterial", "inconclusive"]


@dataclass(frozen=True)
class Recommendation:
    """A verdict, the numbers behind it, and the thresholds it was judged against."""

    verdict: Verdict
    effect: float
    relative_effect: float
    probability_of_benefit: float | None
    interval: tuple[float, float]
    credible: bool
    material: bool
    min_probability: float
    min_relative_effect: float
    reason: str

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "verdict": self.verdict,
            "effect": self.effect,
            "relative_effect": self.relative_effect,
            "probability_of_benefit": self.probability_of_benefit,
            "ci_lower": self.interval[0],
            "ci_upper": self.interval[1],
            "credible": self.credible,
            "material": self.material,
            "min_probability": self.min_probability,
            "min_relative_effect": self.min_relative_effect,
            "reason": self.reason,
        }


def recommend(
    effect: float,
    interval: tuple[float, float],
    baseline: float,
    *,
    probability_of_benefit: float | None = None,
    min_probability: float = 0.95,
    min_relative_effect: float = 0.01,
) -> Recommendation:
    """Judge one estimate against a credibility bar and a materiality bar.

    ``baseline`` is the control mean, used to express the effect in relative
    terms. Materiality is judged on the relative effect because a threshold in
    raw units means nothing without knowing the scale.
    """
    lower, upper = interval
    if lower > upper:
        raise EstimationError("interval bounds are reversed")
    if not 0.0 < min_probability < 1.0:
        raise EstimationError("min_probability must lie strictly between 0 and 1")

    relative = float(effect / baseline) if baseline != 0 else float("nan")

    if probability_of_benefit is not None:
        credible = probability_of_benefit >= min_probability
        evidence = (
            f"P(benefit) = {probability_of_benefit:.4f} against a bar of {min_probability:.2f}"
        )
    else:
        credible = lower > 0.0
        evidence = (
            f"the {100 * (1 - 0.05):.0f}% interval [{lower:,.4g}, {upper:,.4g}] excludes zero"
            if lower > 0
            else f"the interval [{lower:,.4g}, {upper:,.4g}] contains zero"
        )

    material = bool(np.isfinite(relative) and abs(relative) >= min_relative_effect)

    harmful = upper < 0.0 or (
        probability_of_benefit is not None and probability_of_benefit <= 1.0 - min_probability
    )

    if harmful:
        verdict: Verdict = "do not ship"
        reason = f"the effect is credibly negative: {evidence}"
    elif credible and material:
        verdict = "ship"
        reason = (
            f"{evidence}, and the effect of {relative:+.2%} clears the "
            f"{min_relative_effect:.2%} materiality bar"
        )
    elif credible and not material:
        verdict = "credible but immaterial"
        reason = (
            f"{evidence}, but the effect of {relative:+.2%} is below the "
            f"{min_relative_effect:.2%} materiality bar - it is real and not worth the change"
        )
    else:
        verdict = "inconclusive"
        reason = (
            f"{evidence}; the experiment cannot separate this effect from none. "
            "Check the power the design actually had before reading it as no effect"
        )

    return Recommendation(
        verdict=verdict,
        effect=float(effect),
        relative_effect=relative,
        probability_of_benefit=probability_of_benefit,
        interval=(float(lower), float(upper)),
        credible=credible,
        material=material,
        min_probability=min_probability,
        min_relative_effect=min_relative_effect,
        reason=reason,
    )


def targeting_policy(
    predicted_uplift: ArrayLike,
    *,
    cost_per_treatment: float = 0.0,
    value_per_outcome: float = 1.0,
    budget_share: float | None = None,
) -> pd.DataFrame:
    """Who to treat, given what treating costs and what an outcome is worth.

    Treating everyone is optimal only when treatment is free. Once it costs
    something, a unit is worth treating when its expected uplift times the value
    of an outcome exceeds the cost - which is a per-unit decision, and the whole
    reason to model uplift rather than average effect.

    ``budget_share`` caps the treated fraction when capacity binds before
    profitability does; the highest-uplift units are kept.
    """
    u = np.asarray(predicted_uplift, dtype=np.float64)
    if u.size == 0:
        raise EstimationError("no predictions to build a policy from")
    if budget_share is not None and not 0.0 < budget_share <= 1.0:
        raise EstimationError("budget_share must lie in (0, 1]")

    expected_value = u * value_per_outcome - cost_per_treatment
    profitable = expected_value > 0.0

    if budget_share is not None:
        limit = int(np.floor(budget_share * u.size))
        keep = np.zeros(u.size, dtype=bool)
        if limit > 0:
            keep[np.argsort(-u, kind="stable")[:limit]] = True
        treat = profitable & keep
    else:
        treat = profitable

    return pd.DataFrame(
        {
            "predicted_uplift": u,
            "expected_value": expected_value,
            "treat": treat,
        }
    )

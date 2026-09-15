"""Difference-in-differences.

When treatment is not randomised but arrives at a known moment for some units
and not others, the before-after change in the untreated group estimates what
would have happened to the treated group anyway. Subtracting it removes
everything that moved for both - a seasonal swing, a macro shock, a site-wide
redesign - and leaves the treatment.

The whole thing rests on **parallel trends**: absent treatment, the two groups
would have moved together. That is an assumption about a world we do not
observe, so it cannot be verified. What *can* be done is to check whether the
groups moved together before treatment, which is evidence about the assumption
rather than proof of it, and that check is run and reported here rather than
waved at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..errors import AssumptionError, EstimationError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DiDResult:
    """A difference-in-differences estimate and its pre-trend evidence."""

    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    pvalue: float
    n_units: int
    n_periods: int
    treated_units: int
    pre_trend_difference: float
    pre_trend_pvalue: float
    parallel_trends_supported: bool
    method: str
    period_means: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "pvalue": self.pvalue,
            "n_units": float(self.n_units),
            "n_periods": float(self.n_periods),
            "treated_units": float(self.treated_units),
            "pre_trend_difference": self.pre_trend_difference,
            "pre_trend_pvalue": self.pre_trend_pvalue,
            "parallel_trends_supported": self.parallel_trends_supported,
            "method": self.method,
        }


def difference_in_differences(
    frame: pd.DataFrame,
    *,
    outcome: str,
    unit: str,
    time: str,
    treated_units: list[str] | tuple[str, ...],
    treatment_time: float,
    confidence: float = 0.95,
    pre_trend_alpha: float = 0.05,
    refuse_on_pre_trend: bool = False,
) -> DiDResult:
    """Estimate a treatment effect from a panel with two-way fixed effects.

    ``treatment_time`` is the first period in which treatment is active.

    The pre-trend test regresses the outcome on a group-by-time interaction
    using only pre-treatment periods. A significant interaction means the groups
    were already diverging before anything happened, and the parallel-trends
    assumption is not credible - the reported effect is then partly that
    divergence continuing.
    """
    for column in (outcome, unit, time):
        if column not in frame.columns:
            raise EstimationError(f"no column {column!r} in the panel")
    if not treated_units:
        raise EstimationError("no treated units given")

    data = frame[[outcome, unit, time]].dropna().copy()
    data["treated_group"] = data[unit].isin(list(treated_units)).astype(int)
    data["post"] = (data[time] >= treatment_time).astype(int)

    if data["treated_group"].nunique() < 2:
        raise EstimationError("the panel contains no untreated comparison units")
    if data["post"].nunique() < 2:
        raise EstimationError("the panel contains no pre-treatment periods")

    pre = data[data["post"] == 0]
    if pre[time].nunique() < 2:
        raise EstimationError("at least two pre-treatment periods are needed to test trends")

    # Pre-trend: do the groups already move apart before treatment?
    pre_model = smf.ols(f"{outcome} ~ treated_group * {time}", data=pre).fit()
    interaction = f"treated_group:{time}"
    pre_difference = float(pre_model.params.get(interaction, np.nan))
    pre_pvalue = float(pre_model.pvalues.get(interaction, np.nan))
    supported = bool(np.isnan(pre_pvalue) or pre_pvalue >= pre_trend_alpha)

    if not supported and refuse_on_pre_trend:
        raise AssumptionError(
            f"the groups were already diverging before treatment "
            f"({interaction} = {pre_difference:.4g}, p = {pre_pvalue:.4g}). "
            "A difference-in-differences estimate here would attribute that "
            "existing divergence to the treatment."
        )

    model = smf.ols(f"{outcome} ~ treated_group + post + treated_group:post", data=data).fit()
    term = "treated_group:post"
    estimate = float(model.params[term])
    se = float(model.bse[term])
    interval = model.conf_int(alpha=1.0 - confidence).loc[term]

    period_means = (
        data.groupby(["treated_group", time])[outcome].mean().rename("mean_outcome").reset_index()
    )

    result = DiDResult(
        estimate=estimate,
        standard_error=se,
        ci_lower=float(interval.iloc[0]),
        ci_upper=float(interval.iloc[1]),
        pvalue=float(model.pvalues[term]),
        n_units=int(data[unit].nunique()),
        n_periods=int(data[time].nunique()),
        treated_units=int(data.loc[data["treated_group"] == 1, unit].nunique()),
        pre_trend_difference=pre_difference,
        pre_trend_pvalue=pre_pvalue,
        parallel_trends_supported=supported,
        method="two-way fixed effects (group x post)",
        period_means=period_means,
    )
    logger.info(
        "did_estimated",
        estimate=round(estimate, 4),
        pvalue=round(result.pvalue, 4),
        pre_trend_pvalue=round(pre_pvalue, 4) if not np.isnan(pre_pvalue) else None,
        parallel_trends_supported=supported,
    )
    return result

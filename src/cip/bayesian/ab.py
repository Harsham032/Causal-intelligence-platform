"""Bayesian two-arm comparison.

The frequentist output answers "how surprising is this data under no effect?".
The question a decision actually needs is "how likely is it that the treatment
is better, and by enough to be worth shipping?" - and that is a statement about
the effect, which only a posterior can make.

Three quantities are reported, in the order they matter:

* the **posterior distribution of the effect**, summarised by its mean and a
  credible interval, which is the range the effect is in with the stated
  probability - what people wrongly believe a confidence interval means;
* the **probability that treatment beats control**, which is directly
  actionable and has no frequentist equivalent;
* the **probability that the effect is practically negligible**, computed
  against a region of practical equivalence. This is the part most Bayesian A/B
  writeups omit, and it is what stops a programme shipping a real but pointless
  0.01% improvement at the cost of a migration.

Two engines, chosen by outcome type. A binary outcome uses the Beta-Binomial
conjugate pair, which has an exact closed-form posterior - no sampler, no
convergence to check, and no reason to use MCMC. A continuous outcome uses
PyMC with Student-t likelihoods, whose heavier tails stop a handful of large
values dominating the posterior, which matters for revenue-shaped data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from ..errors import EstimationError
from ..logging_utils import get_logger

logger = get_logger(__name__)

Likelihood = Literal["student-t", "normal"]


@dataclass
class BayesianResult:
    """A posterior summary of the treatment effect."""

    metric: str
    treated_mean: float
    control_mean: float
    effect_mean: float
    effect_median: float
    ci_lower: float
    ci_upper: float
    probability_of_benefit: float
    probability_practically_equivalent: float
    expected_loss: float
    n_treated: int
    n_control: int
    method: str
    credibility: float = 0.95
    rope: tuple[float, float] = (0.0, 0.0)
    diagnostics: dict[str, float] = field(default_factory=dict)
    draws: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "treated_mean": self.treated_mean,
            "control_mean": self.control_mean,
            "effect_mean": self.effect_mean,
            "effect_median": self.effect_median,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "probability_of_benefit": self.probability_of_benefit,
            "probability_practically_equivalent": self.probability_practically_equivalent,
            "expected_loss": self.expected_loss,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "method": self.method,
            "rope_lower": self.rope[0],
            "rope_upper": self.rope[1],
            **{f"diagnostic_{k}": v for k, v in self.diagnostics.items()},
        }


def _summarise(
    effect_draws: np.ndarray,
    *,
    metric: str,
    treated_mean: float,
    control_mean: float,
    n_treated: int,
    n_control: int,
    method: str,
    credibility: float,
    rope: tuple[float, float],
    diagnostics: dict[str, float] | None = None,
) -> BayesianResult:
    """Turn posterior draws of the effect into the quantities a decision needs."""
    if effect_draws.size == 0:
        raise EstimationError("no posterior draws to summarise")
    alpha = (1.0 - credibility) / 2.0
    lower, upper = np.quantile(effect_draws, [alpha, 1.0 - alpha])

    # Expected loss: if we ship and the treatment is in fact worse, how much do
    # we lose on average? Zero-truncated, because being right costs nothing.
    expected_loss = float(np.maximum(-effect_draws, 0.0).mean())

    in_rope = (
        float(((effect_draws >= rope[0]) & (effect_draws <= rope[1])).mean())
        if rope[1] > rope[0]
        else 0.0
    )

    return BayesianResult(
        metric=metric,
        treated_mean=treated_mean,
        control_mean=control_mean,
        effect_mean=float(effect_draws.mean()),
        effect_median=float(np.median(effect_draws)),
        ci_lower=float(lower),
        ci_upper=float(upper),
        probability_of_benefit=float((effect_draws > 0).mean()),
        probability_practically_equivalent=in_rope,
        expected_loss=expected_loss,
        n_treated=n_treated,
        n_control=n_control,
        method=method,
        credibility=credibility,
        rope=rope,
        diagnostics=diagnostics or {},
        draws=effect_draws,
    )


def beta_binomial_test(
    treated: ArrayLike,
    control: ArrayLike,
    *,
    metric: str = "conversion",
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    draws: int = 200_000,
    credibility: float = 0.95,
    rope_relative: float = 0.0,
    seed: int = 20260101,
) -> BayesianResult:
    """Exact conjugate analysis of two conversion rates.

    The Beta prior is conjugate to the Binomial likelihood, so the posterior is
    another Beta and needs no sampler. Sampling from it is only to get the
    *difference* of two Betas, which has no convenient closed form; those draws
    are exact posterior draws, not an MCMC approximation, so there is nothing to
    diagnose for convergence.

    The default Beta(1, 1) is uniform on [0, 1] - deliberately uninformative, so
    the data does the work and the result can be compared against the
    frequentist estimate without arguing about the prior.
    """
    t = np.asarray(treated, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    for name, arm in (("treated", t), ("control", c)):
        if arm.size == 0:
            raise EstimationError(f"the {name} arm is empty")
        if not np.isin(arm, (0.0, 1.0)).all():
            raise EstimationError(f"the {name} arm must be binary 0/1")
    if prior_alpha <= 0 or prior_beta <= 0:
        raise EstimationError("prior parameters must be positive")

    rng = np.random.default_rng(seed)
    post_t = rng.beta(prior_alpha + t.sum(), prior_beta + t.size - t.sum(), size=draws)
    post_c = rng.beta(prior_alpha + c.sum(), prior_beta + c.size - c.sum(), size=draws)
    effect = post_t - post_c

    control_rate = float(c.mean())
    rope_half = abs(control_rate) * rope_relative
    result = _summarise(
        effect,
        metric=metric,
        treated_mean=float(t.mean()),
        control_mean=control_rate,
        n_treated=int(t.size),
        n_control=int(c.size),
        method=f"beta-binomial conjugate (prior Beta({prior_alpha:g}, {prior_beta:g}))",
        credibility=credibility,
        rope=(-rope_half, rope_half),
    )
    logger.info(
        "bayesian_conjugate",
        metric=metric,
        effect=round(result.effect_mean, 6),
        probability_of_benefit=round(result.probability_of_benefit, 4),
    )
    return result


def normal_test(
    treated: ArrayLike,
    control: ArrayLike,
    *,
    metric: str = "outcome",
    likelihood: Likelihood = "student-t",
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    credibility: float = 0.95,
    rope_relative: float = 0.0,
    seed: int = 20260101,
) -> BayesianResult:
    """Posterior for the difference between two arms of a continuous outcome.

    The likelihood is a choice about the estimand, not a technicality, so it is
    a parameter rather than a default buried in the code.

    ``student-t`` estimates a location shift robust to extreme values, with the
    degrees-of-freedom parameter learned from the data. Use it when a handful of
    outliers should not decide the answer.

    ``normal`` estimates the shift in the **mean**, which is what a difference
    in means estimates and what a total-value question needs: if the decision is
    about revenue or earnings in aggregate, the large values are the point and
    down-weighting them answers a different question.

    The gap between them is not a rounding error. On the NSW earnings data - many
    zeros, a long right tail - the Student-t posterior centres near $1,277 and
    the Normal near the $1,794 the difference in means reports. Both are correct
    about their own estimand.
    """
    t = np.asarray(treated, dtype=np.float64)
    c = np.asarray(control, dtype=np.float64)
    if t.size < 2 or c.size < 2:
        raise EstimationError("each arm needs at least two observations")
    if likelihood not in ("student-t", "normal"):
        raise EstimationError(f"unknown likelihood {likelihood!r}")

    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - a hard dependency
        raise EstimationError("pymc is not installed; run `make install`") from exc

    pooled_sd = float(np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2.0)) or 1.0
    centre = float(np.concatenate([t, c]).mean())

    with pm.Model():
        # Priors scaled to the data so the model is usable on any outcome scale
        # without hand-tuning; wide enough to be dominated by the likelihood.
        mu_c = pm.Normal("mu_control", mu=centre, sigma=10.0 * pooled_sd)
        effect = pm.Normal("effect", mu=0.0, sigma=5.0 * pooled_sd)
        sigma_t = pm.HalfNormal("sigma_treated", sigma=5.0 * pooled_sd)
        sigma_c = pm.HalfNormal("sigma_control", sigma=5.0 * pooled_sd)
        nu = pm.Exponential("nu", lam=1.0 / 30.0) + 1.0

        if likelihood == "student-t":
            pm.StudentT("obs_control", nu=nu, mu=mu_c, sigma=sigma_c, observed=c)
            pm.StudentT("obs_treated", nu=nu, mu=mu_c + effect, sigma=sigma_t, observed=t)
        else:
            pm.Normal("obs_control", mu=mu_c, sigma=sigma_c, observed=c)
            pm.Normal("obs_treated", mu=mu_c + effect, sigma=sigma_t, observed=t)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=True,
        )

    posterior = idata.posterior["effect"].to_numpy().reshape(-1)
    diagnostics = _convergence(idata, "effect")

    control_mean = float(c.mean())
    rope_half = abs(control_mean) * rope_relative
    result = _summarise(
        posterior,
        metric=metric,
        treated_mean=float(t.mean()),
        control_mean=control_mean,
        n_treated=int(t.size),
        n_control=int(c.size),
        method=f"{likelihood} model, {chains} chains x {draws} draws",
        credibility=credibility,
        rope=(-rope_half, rope_half),
        diagnostics=diagnostics,
    )
    logger.info(
        "bayesian_mcmc",
        metric=metric,
        effect=round(result.effect_mean, 4),
        probability_of_benefit=round(result.probability_of_benefit, 4),
        **{k: round(v, 4) for k, v in diagnostics.items()},
    )
    return result


def _convergence(idata: Any, variable: str) -> dict[str, float]:
    """R-hat and effective sample size.

    Reported rather than hidden: a posterior from chains that did not converge
    is a set of numbers, not an inference, and the only way a reader can tell is
    if these are on the page. R-hat above 1.01 or an effective sample size below
    roughly 400 means the summary above it should not be trusted.
    """
    try:
        import arviz as az

        summary = az.summary(idata, var_names=[variable], kind="diagnostics")
        return {
            "r_hat": float(summary["r_hat"].iloc[0]),
            "ess_bulk": float(summary["ess_bulk"].iloc[0]),
            "ess_tail": float(summary["ess_tail"].iloc[0]),
        }
    except Exception as exc:  # pragma: no cover - diagnostics must not fail a run
        logger.warning("convergence_diagnostics_unavailable", error=str(exc))
        return {}


def sample_posterior(result: BayesianResult) -> np.ndarray:
    """The posterior draws behind a result, for plotting."""
    if result.draws is None:
        raise EstimationError("this result carries no posterior draws")
    return result.draws

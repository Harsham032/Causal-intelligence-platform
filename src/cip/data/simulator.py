"""Simulated experiments with known treatment effects.

Real data cannot tell you whether an estimator is right. Even the LaLonde
benchmark only pins down one number - the average effect - so it can score an
ATE estimate and nothing else. Uplift models rank units by their *individual*
effect, and no real dataset ever reveals an individual effect: each unit is
observed under one arm only, which is the fundamental problem of causal
inference.

A simulator sidesteps it by generating both potential outcomes. Y(0) and Y(1)
are both written down, the observed outcome is whichever arm the unit was
assigned to, and the difference is that unit's true effect. That makes it
possible to measure what otherwise can only be asserted: an estimator's bias,
whether its confidence intervals cover at their stated rate, and whether an
uplift model's ranking matches the real one.

The cost is the usual one, stated here rather than buried: a simulator contains
only the structure it was given. Its numbers measure the *estimator*, not the
world. Section 11 of ``docs/results.md`` says what that does and does not
license.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..errors import DataError
from ..logging_utils import get_logger

logger = get_logger(__name__)

Outcome = Literal["continuous", "binary"]
Effect = Literal["constant", "heterogeneous", "none"]


@dataclass
class SimulationConfig:
    """Parameters of a simulated experiment.

    ``confounding`` is the one that changes the nature of the problem. At zero,
    assignment is a coin flip and a difference in means is unbiased - a
    randomised experiment. Above zero, assignment depends on covariates that
    also drive the outcome, which is an observational study, and a difference in
    means is then wrong by a knowable amount.
    """

    n_units: int = 20_000
    n_covariates: int = 10
    # How many covariates actually drive the outcome. The rest are noise, which
    # is realistic and punishes estimators that cannot ignore them.
    n_informative: int = 5
    treatment_share: float = 0.5
    confounding: float = 0.0
    outcome: Outcome = "continuous"
    effect: Effect = "heterogeneous"
    # Average effect, on the outcome's own scale for a continuous outcome and
    # on the probability scale for a binary one.
    average_effect: float = 1.0
    effect_scale: float = 1.0
    noise: float = 1.0
    baseline: float = 5.0
    seed: int = 20260101

    def __post_init__(self) -> None:
        if self.n_units < 10:
            raise DataError("a simulation needs at least 10 units")
        if not 0.0 < self.treatment_share < 1.0:
            raise DataError("treatment_share must lie strictly between 0 and 1")
        if self.n_informative > self.n_covariates:
            raise DataError("n_informative cannot exceed n_covariates")
        if self.confounding < 0.0:
            raise DataError("confounding must not be negative")
        if self.outcome == "binary" and not 0.0 <= self.baseline <= 1.0:
            raise DataError("for a binary outcome, baseline must be a probability")


@dataclass
class SimulationResult:
    """The generated sample, plus the truths a real dataset would hide."""

    data: pd.DataFrame
    covariates: tuple[str, ...]
    treatment: str
    outcome: str
    true_ate: float
    true_att: float
    config: SimulationConfig

    @property
    def true_cate(self) -> np.ndarray:
        """Each unit's own treatment effect."""
        return self.data["true_cate"].to_numpy()

    def describe(self) -> dict[str, float]:
        treated = self.data[self.treatment] == 1
        return {
            "units": float(len(self.data)),
            "treated": float(treated.sum()),
            "treatment_share": float(treated.mean()),
            "true_ate": self.true_ate,
            "true_att": self.true_att,
            "cate_std": float(self.data["true_cate"].std(ddof=1)),
            "observed_difference_in_means": float(
                self.data.loc[treated, self.outcome].mean()
                - self.data.loc[~treated, self.outcome].mean()
            ),
        }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def simulate_experiment(config: SimulationConfig) -> SimulationResult:
    """Generate a sample in which both potential outcomes are known."""
    rng = np.random.default_rng(config.seed)
    n, p = config.n_units, config.n_covariates
    names = tuple(f"x{i}" for i in range(p))

    X = rng.normal(size=(n, p))
    informative = np.zeros(p)
    informative[: config.n_informative] = rng.uniform(0.5, 1.5, size=config.n_informative)

    # Assignment. With no confounding this is a coin flip at the configured
    # share; with confounding it tilts towards units whose covariates also
    # raise the outcome, which is exactly what makes naive comparisons wrong.
    if config.confounding == 0.0:
        propensity = np.full(n, config.treatment_share)
    else:
        logit_share = np.log(config.treatment_share / (1.0 - config.treatment_share))
        propensity = _sigmoid(logit_share + config.confounding * (X @ informative) / np.sqrt(p))
    treatment = rng.binomial(1, propensity)

    # Each unit's own treatment effect.
    if config.effect == "none":
        cate = np.zeros(n)
    elif config.effect == "constant":
        cate = np.full(n, config.average_effect)
    else:
        # Effect varies with the first two informative covariates, and the
        # interaction means a linear model cannot recover the ranking.
        modifier = X[:, 0] + 0.5 * X[:, 1] * X[:, 0]
        modifier = (modifier - modifier.mean()) / (modifier.std(ddof=1) or 1.0)
        cate = config.average_effect + config.effect_scale * modifier

    baseline_index = config.baseline + X @ informative

    if config.outcome == "binary":
        # On the probability scale, so an effect cannot push a unit outside
        # [0, 1] - which would silently change the average effect.
        p0 = np.clip(_sigmoid(X @ informative), 0.01, 0.99)
        p1 = np.clip(p0 + cate, 0.001, 0.999)
        cate = p1 - p0
        y0 = rng.binomial(1, p0).astype(float)
        y1 = rng.binomial(1, p1).astype(float)
    else:
        noise0 = rng.normal(scale=config.noise, size=n)
        noise1 = rng.normal(scale=config.noise, size=n)
        y0 = baseline_index + noise0
        y1 = baseline_index + cate + noise1

    observed = np.where(treatment == 1, y1, y0)

    frame = pd.DataFrame(X, columns=list(names))
    frame["treatment"] = treatment.astype(int)
    frame["outcome"] = observed
    frame["propensity_true"] = propensity
    frame["true_cate"] = cate
    # Potential outcomes are kept so evaluation can use them, and are never
    # passed to an estimator; `estimation_frame` is what models are given.
    frame["y0"] = y0
    frame["y1"] = y1

    treated_mask = treatment == 1
    if treated_mask.sum() == 0 or treated_mask.sum() == n:
        raise DataError("simulation produced a single treatment arm; adjust treatment_share")

    result = SimulationResult(
        data=frame,
        covariates=names,
        treatment="treatment",
        outcome="outcome",
        true_ate=float(cate.mean()),
        true_att=float(cate[treated_mask].mean()),
        config=config,
    )
    logger.info("simulation_complete", **{k: round(v, 4) for k, v in result.describe().items()})
    return result


def estimation_frame(result: SimulationResult) -> pd.DataFrame:
    """The columns an estimator is allowed to see.

    Potential outcomes, the true effect and the true propensity are dropped. An
    estimator handed any of them would score perfectly and prove nothing.
    """
    columns = [*result.covariates, result.treatment, result.outcome]
    return result.data[columns].copy()

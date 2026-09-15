"""Uplift evaluation: Qini, AUUC and decile tables.

Scoring an uplift model is harder than scoring a classifier, because the thing
it predicts is never observed for any single unit. The trick every metric here
relies on is to stop asking about units and start asking about *groups*: take
the units a model ranks highest, and compare the treated and control outcomes
inside that group. Randomisation makes that comparison valid at every cut-off,
so a curve can be traced without ever knowing an individual effect.

**Qini curve**: for the top ``k`` units by predicted uplift, the incremental
outcomes gained versus targeting nobody, with the control arm scaled to the
treated arm's size. A model that ranks well pulls the curve above the diagonal
that random targeting would trace.

**Qini coefficient**: the area between the model's curve and that diagonal,
normalised so 1.0 is perfect ranking. Zero means no better than random, and
negative means the ranking is actively inverted - worth acting on, because a
confidently wrong uplift model spends budget on the people it repels.

**AUUC**: the area under the uplift curve. Reported alongside Qini because the
two disagree when the arms are unbalanced, and a single number that quietly
depends on the split is not a metric anyone should trust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ..errors import EvaluationError


@dataclass
class QiniResult:
    """A Qini curve and the summary numbers taken from it."""

    fractions: np.ndarray
    model_gain: np.ndarray
    random_gain: np.ndarray
    qini_coefficient: float
    auuc: float
    n_units: int
    n_treated: int

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fraction_targeted": self.fractions,
                "model_gain": self.model_gain,
                "random_gain": self.random_gain,
            }
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "qini_coefficient": self.qini_coefficient,
            "auuc": self.auuc,
            "n_units": float(self.n_units),
            "n_treated": float(self.n_treated),
        }


def _validate(outcome: np.ndarray, treatment: np.ndarray, uplift: np.ndarray) -> None:
    if not (outcome.shape == treatment.shape == uplift.shape):
        raise EvaluationError("outcome, treatment and uplift must align")
    if outcome.size == 0:
        raise EvaluationError("no units to evaluate")
    if set(np.unique(treatment).tolist()) - {0, 1}:
        raise EvaluationError("treatment must be binary 0/1")
    if treatment.sum() == 0 or treatment.sum() == treatment.size:
        raise EvaluationError("both arms must contain units")


def qini_curve(
    outcome: ArrayLike,
    treatment: ArrayLike,
    uplift: ArrayLike,
) -> QiniResult:
    """Trace the incremental gain from targeting the top-ranked units.

    At each cut-off the treated outcomes in the targeted set are compared with
    the control outcomes, scaled by the ratio of arm sizes so the two are
    comparable. Ties in the predicted uplift are broken by the original order,
    which keeps the curve deterministic across runs.
    """
    y = np.asarray(outcome, dtype=np.float64)
    t = np.asarray(treatment, dtype=np.int64)
    u = np.asarray(uplift, dtype=np.float64)
    _validate(y, t, u)

    order = np.argsort(-u, kind="stable")
    y, t = y[order], t[order]

    treated_cumulative = np.cumsum(y * t)
    control_cumulative = np.cumsum(y * (1 - t))
    n_treated_cumulative = np.cumsum(t)
    n_control_cumulative = np.cumsum(1 - t)

    # Scale the control arm up to the treated arm's size at each cut-off; with
    # no controls yet the incremental gain is just the treated outcomes.
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled_control = np.where(
            n_control_cumulative > 0,
            control_cumulative * n_treated_cumulative / n_control_cumulative,
            0.0,
        )
    model_gain = treated_cumulative - scaled_control

    n = y.size
    fractions = np.arange(1, n + 1, dtype=np.float64) / n
    # Random targeting reaches the overall incremental total in proportion to
    # how much of the population it covers: a straight line to the same endpoint.
    random_gain = model_gain[-1] * fractions

    # Trapezoidal areas over the targeted fraction.
    area_model = float(np.trapezoid(model_gain, fractions))
    area_random = float(np.trapezoid(random_gain, fractions))

    # Perfect ranking: every unit ordered by its realised contribution.
    perfect_order = np.argsort(-(y * t - y * (1 - t)), kind="stable")
    perfect_gain = np.cumsum((y * t - y * (1 - t))[perfect_order])
    area_perfect = float(np.trapezoid(perfect_gain, fractions))

    denominator = area_perfect - area_random
    coefficient = float((area_model - area_random) / denominator) if denominator != 0 else 0.0

    return QiniResult(
        fractions=fractions,
        model_gain=model_gain,
        random_gain=random_gain,
        qini_coefficient=coefficient,
        auuc=area_model,
        n_units=int(n),
        n_treated=int(t.sum()),
    )


def qini_coefficient(outcome: ArrayLike, treatment: ArrayLike, uplift: ArrayLike) -> float:
    """The Qini coefficient alone, when the curve is not needed."""
    return qini_curve(outcome, treatment, uplift).qini_coefficient


def auuc(outcome: ArrayLike, treatment: ArrayLike, uplift: ArrayLike) -> float:
    """Area under the uplift curve."""
    return qini_curve(outcome, treatment, uplift).auuc


def uplift_by_decile(
    outcome: ArrayLike,
    treatment: ArrayLike,
    uplift: ArrayLike,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Observed uplift within each predicted-uplift bin.

    The most useful diagnostic in the module, and the one that catches a model
    the aggregate metrics flatter. If the observed effect does not fall as the
    predicted effect falls, the ranking is not real - whatever the Qini
    coefficient says. A monotone table is the claim; Qini is only its summary.
    """
    y = np.asarray(outcome, dtype=np.float64)
    t = np.asarray(treatment, dtype=np.int64)
    u = np.asarray(uplift, dtype=np.float64)
    _validate(y, t, u)
    if n_bins < 2:
        raise EvaluationError("n_bins must be at least 2")

    order = np.argsort(-u, kind="stable")
    y, t, u = y[order], t[order], u[order]
    bins = np.array_split(np.arange(y.size), n_bins)

    rows = []
    for index, idx in enumerate(bins, start=1):
        if idx.size == 0:
            continue
        bin_t, bin_y = t[idx], y[idx]
        n_t, n_c = int(bin_t.sum()), int((1 - bin_t).sum())
        treated_mean = float(bin_y[bin_t == 1].mean()) if n_t else float("nan")
        control_mean = float(bin_y[bin_t == 0].mean()) if n_c else float("nan")
        rows.append(
            {
                "bin": index,
                "n_units": int(idx.size),
                "n_treated": n_t,
                "n_control": n_c,
                "predicted_uplift": float(u[idx].mean()),
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "observed_uplift": treated_mean - control_mean,
            }
        )
    return pd.DataFrame(rows)

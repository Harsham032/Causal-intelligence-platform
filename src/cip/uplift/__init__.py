"""Uplift modelling: who responds to treatment, not whether anyone does."""

from .curves import (
    QiniResult,
    auuc,
    qini_coefficient,
    qini_curve,
    uplift_by_decile,
)
from .learners import UpliftModel, fit_uplift

__all__ = [
    "QiniResult",
    "UpliftModel",
    "auuc",
    "fit_uplift",
    "qini_coefficient",
    "qini_curve",
    "uplift_by_decile",
]

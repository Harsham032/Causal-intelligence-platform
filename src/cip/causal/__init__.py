"""Causal estimators for observational data."""

from .did import DiDResult, difference_in_differences
from .learners import CausalEstimate, causal_forest, double_machine_learning
from .propensity import (
    OverlapReport,
    PropensityResult,
    check_overlap,
    estimate_propensity,
    trim_to_overlap,
)
from .weighting import WeightedResult, inverse_probability_weighting, matched_estimate

__all__ = [
    "CausalEstimate",
    "DiDResult",
    "OverlapReport",
    "PropensityResult",
    "WeightedResult",
    "causal_forest",
    "check_overlap",
    "difference_in_differences",
    "double_machine_learning",
    "estimate_propensity",
    "inverse_probability_weighting",
    "matched_estimate",
    "trim_to_overlap",
]

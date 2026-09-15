"""Frequentist analysis of experiments."""

from .multiplicity import MultipleTestResult, adjust_pvalues
from .tests import (
    TestResult,
    bootstrap_difference,
    difference_in_means,
    difference_in_proportions,
)

__all__ = [
    "MultipleTestResult",
    "TestResult",
    "adjust_pvalues",
    "bootstrap_difference",
    "difference_in_means",
    "difference_in_proportions",
]

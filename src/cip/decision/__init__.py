"""Turning an estimate into a recommendation."""

from .rules import Recommendation, recommend, targeting_policy

__all__ = ["Recommendation", "recommend", "targeting_policy"]

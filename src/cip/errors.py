"""Exception hierarchy.

Every failure raised by library code derives from :class:`CipError`, so callers
can tell an expected domain failure from a genuine bug.
"""

from __future__ import annotations


class CipError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(CipError):
    """Configuration is missing, malformed or internally inconsistent."""


class DataError(CipError):
    """A dataset could not be loaded, or does not have the expected shape."""


class DesignError(CipError):
    """An experiment design is not satisfiable as specified."""


class EstimationError(CipError):
    """An estimator could not be fitted, or was given inconsistent inputs."""


class AssumptionError(CipError):
    """An identifying assumption is violated badly enough to refuse an estimate.

    Causal estimates are only as good as the assumptions behind them. Where a
    violation is detectable - no overlap in propensity scores, a treatment that
    never varies, a pre-trend that is plainly not parallel - it is better to
    refuse than to return a number that looks like an effect.
    """


class EvaluationError(CipError):
    """An evaluation was given inconsistent inputs."""

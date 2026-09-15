"""Multiple-testing corrections.

An experiment that reports twenty metrics at alpha 0.05 will, on average,
produce one "significant" result even when the treatment does nothing at all.
Reading that one as a win is not a subtle statistical error; it is the single
most reliable way for an experimentation programme to ship changes that do
nothing.

Two corrections, for two different questions:

*Bonferroni* controls the family-wise error rate - the probability of **any**
false positive. Appropriate when one false positive is expensive: a guardrail
metric that would trigger a rollback, a regulatory claim.

*Benjamini-Hochberg* controls the false discovery rate - the expected share of
**claimed** discoveries that are false. Appropriate when screening many metrics
to decide what to look at next, where tolerating some false leads is cheaper
than missing real ones. It is uniformly more powerful than Bonferroni.

Neither is a substitute for naming the primary metric before the experiment
starts, which is the only thing that fully removes the problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from ..errors import EstimationError

Method = Literal["none", "bonferroni", "benjamini-hochberg"]


@dataclass(frozen=True)
class MultipleTestResult:
    """Adjusted p-values, and what survived."""

    names: tuple[str, ...]
    raw: tuple[float, ...]
    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]
    method: Method
    alpha: float

    @property
    def discoveries(self) -> list[str]:
        return [n for n, r in zip(self.names, self.rejected, strict=True) if r]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "tested": len(self.names),
            "discoveries": self.discoveries,
            "results": [
                {"metric": n, "pvalue": p, "adjusted_pvalue": a, "rejected": r}
                for n, p, a, r in zip(
                    self.names, self.raw, self.adjusted, self.rejected, strict=True
                )
            ],
        }


def adjust_pvalues(
    pvalues: ArrayLike,
    *,
    names: list[str] | tuple[str, ...] | None = None,
    method: Method = "benjamini-hochberg",
    alpha: float = 0.05,
) -> MultipleTestResult:
    """Adjust a family of p-values and report which survive."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise EstimationError("pvalues must be a non-empty one-dimensional sequence")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise EstimationError("p-values must lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise EstimationError("alpha must lie strictly between 0 and 1")

    labels = tuple(names) if names is not None else tuple(f"metric_{i}" for i in range(p.size))
    if len(labels) != p.size:
        raise EstimationError("names must align with pvalues")

    n = p.size
    if method == "none":
        adjusted = p.copy()
    elif method == "bonferroni":
        adjusted = np.minimum(p * n, 1.0)
    elif method == "benjamini-hochberg":
        order = np.argsort(p)
        ranked = p[order]
        # Step-up: scale each p-value by n/rank, then enforce monotonicity from
        # the largest down so a small p-value is never adjusted above a larger
        # one - without that pass the procedure is not valid.
        scaled = ranked * n / np.arange(1, n + 1)
        scaled = np.minimum.accumulate(scaled[::-1])[::-1]
        adjusted = np.empty_like(scaled)
        adjusted[order] = np.minimum(scaled, 1.0)
    else:
        raise EstimationError(f"unknown correction method {method!r}")

    return MultipleTestResult(
        names=labels,
        raw=tuple(float(x) for x in p),
        adjusted=tuple(float(x) for x in adjusted),
        rejected=tuple(bool(x <= alpha) for x in adjusted),
        method=method,
        alpha=alpha,
    )

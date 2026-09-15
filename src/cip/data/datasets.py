"""Real datasets, loaded offline.

Every dataset here ships inside the ``causaldata`` package, so a full run needs
no network access and no credentials. That is a deliberate constraint: an
analysis that cannot be reproduced from a clean checkout is not reproducible.

The headline one is the LaLonde/NSW pairing. It is the canonical validation
exercise in causal inference and the only one on this list that provides a
*measurable* ground truth:

* ``nsw`` is a randomised experiment, so the difference in means between arms
  is an unbiased estimate of the average treatment effect.
* ``cps`` is a survey population that never received the treatment. Pairing the
  experiment's treated units with these controls fabricates an observational
  study whose true effect is already known from the experiment.

An estimator's bias is then not a matter of opinion. It is the gap between what
it returns on the observational sample and what the experiment measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..errors import DataError
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DatasetSpec:
    """What a dataset is, and what it can support."""

    name: str
    loader: str
    treatment: str
    outcome: str
    covariates: tuple[str, ...]
    randomised: bool
    description: str
    supports: tuple[str, ...] = field(default_factory=tuple)
    unit: str = ""
    time: str = ""


# Covariates follow Dehejia and Wahba: demographics plus two years of earnings
# history, which is what makes the observational comparison defensible at all.
LALONDE_COVARIATES = ("age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75")

DATASETS: dict[str, DatasetSpec] = {
    "nsw": DatasetSpec(
        name="nsw",
        loader="nsw_mixtape",
        treatment="treat",
        outcome="re78",
        covariates=LALONDE_COVARIATES,
        randomised=True,
        description=(
            "National Supported Work demonstration, Dehejia-Wahba sample. A randomised "
            "job-training experiment; the outcome is 1978 earnings in US dollars."
        ),
        supports=("experiment", "frequentist", "bayesian", "benchmark"),
    ),
    "cps": DatasetSpec(
        name="cps",
        loader="cps_mixtape",
        treatment="treat",
        outcome="re78",
        covariates=LALONDE_COVARIATES,
        randomised=False,
        description=(
            "Current Population Survey comparison group. Never treated; used as the "
            "observational control pool for the LaLonde benchmark."
        ),
        supports=("observational",),
    ),
    "nhefs": DatasetSpec(
        name="nhefs",
        loader="nhefs_complete",
        treatment="qsmk",
        outcome="wt82_71",
        covariates=(
            "age",
            "sex",
            "race",
            "education",
            "smokeintensity",
            "smokeyrs",
            "exercise",
            "active",
            "wt71",
        ),
        randomised=False,
        description=(
            "NHEFS smoking-cessation follow-up. The textbook inverse-probability-weighting "
            "example: does quitting smoking change weight, adjusting for who quits?"
        ),
        supports=("observational", "propensity", "ipw", "dml"),
    ),
    "thornton": DatasetSpec(
        name="thornton",
        loader="thornton_hiv",
        treatment="any",
        outcome="got",
        covariates=("age", "distvct", "hiv2004"),
        randomised=True,
        description=(
            "Randomised incentives for learning HIV test results in Malawi. A large "
            "randomised trial with a binary outcome, useful for proportion tests."
        ),
        supports=("experiment", "frequentist", "bayesian"),
    ),
    "organ_donations": DatasetSpec(
        name="organ_donations",
        loader="organ_donations",
        treatment="",
        outcome="Rate",
        covariates=(),
        randomised=False,
        description=(
            "US state organ-donation rates by quarter. California switched to active "
            "choice in Q3 2011; the other states did not, which is a difference-in-"
            "differences design."
        ),
        supports=("did",),
        unit="State",
        time="Quarter",
    ),
    "castle": DatasetSpec(
        name="castle",
        loader="castle",
        treatment="post",
        outcome="l_homicide",
        covariates=(),
        randomised=False,
        description=(
            "Castle-doctrine law adoption by US state and year. A staggered "
            "difference-in-differences panel."
        ),
        supports=("did",),
        unit="sid",
        time="year",
    ),
}


def available_datasets() -> list[DatasetSpec]:
    """Every dataset this platform knows how to load, in a stable order."""
    return [DATASETS[name] for name in sorted(DATASETS)]


def _load_raw(spec: DatasetSpec) -> pd.DataFrame:
    try:
        import causaldata
    except ImportError as exc:  # pragma: no cover - a hard dependency
        raise DataError(
            "causaldata is not installed; run `make install` or `pip install causaldata`"
        ) from exc

    loader: Any = getattr(causaldata, spec.loader, None)
    if loader is None:
        raise DataError(f"causaldata has no dataset named {spec.loader!r}")
    frame = loader.load_pandas().data
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise DataError(f"{spec.name} loaded empty")
    return frame


def load_dataset(name: str) -> tuple[pd.DataFrame, DatasetSpec]:
    """Load one dataset by name, with the specification describing it."""
    spec = DATASETS.get(name)
    if spec is None:
        known = ", ".join(sorted(DATASETS))
        raise DataError(f"unknown dataset {name!r}; available: {known}")
    frame = _load_raw(spec)
    logger.info("dataset_loaded", dataset=spec.name, rows=len(frame), columns=len(frame.columns))
    return frame, spec


def load_lalonde_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, DatasetSpec]:
    """The experimental sample and its observational counterpart.

    Returns ``(experimental, observational, spec)``.

    The observational frame takes the *treated* units from the experiment and
    replaces the experiment's own controls with survey respondents. The
    treatment effect is therefore unchanged by construction - the same people
    received the same programme - while the comparison group is now
    incomparable in exactly the way real observational data is. Anything an
    estimator reports on it can be scored against the experimental answer.
    """
    experimental, spec = load_dataset("nsw")
    controls, _ = load_dataset("cps")

    treated = experimental[experimental[spec.treatment] == 1]
    if treated.empty:
        raise DataError("the experimental sample contains no treated units")

    columns = [spec.treatment, spec.outcome, *spec.covariates]
    missing = [c for c in columns if c not in controls.columns]
    if missing:
        raise DataError(f"the comparison group is missing columns: {', '.join(missing)}")

    observational = pd.concat(
        [treated[columns], controls.assign(**{spec.treatment: 0})[columns]],
        ignore_index=True,
    )
    logger.info(
        "benchmark_built",
        experimental_rows=len(experimental),
        experimental_treated=int(treated.shape[0]),
        observational_rows=len(observational),
        observational_controls=int((observational[spec.treatment] == 0).sum()),
    )
    return experimental[columns].reset_index(drop=True), observational, spec

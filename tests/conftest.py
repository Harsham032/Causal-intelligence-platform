"""Shared fixtures.

Every fixture builds from the seeded simulator or from a dataset that ships
inside causaldata, so no test needs a download and results are identical on any
machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cip.config import AnalysisConfig  # noqa: E402
from cip.data.simulator import (  # noqa: E402
    SimulationConfig,
    SimulationResult,
    estimation_frame,
    simulate_experiment,
)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def config() -> AnalysisConfig:
    return AnalysisConfig.from_yaml(REPO_ROOT / "configs" / "fast.yaml")


@pytest.fixture(scope="session")
def randomised() -> SimulationResult:
    """A randomised experiment: no confounding, so naive comparison is unbiased."""
    return simulate_experiment(
        SimulationConfig(n_units=4000, n_covariates=6, n_informative=3, seed=11)
    )


@pytest.fixture(scope="session")
def confounded() -> SimulationResult:
    """An observational study: assignment depends on covariates that drive the outcome."""
    return simulate_experiment(
        SimulationConfig(n_units=4000, n_covariates=6, n_informative=3, confounding=2.0, seed=12)
    )


@pytest.fixture(scope="session")
def binary_trial() -> SimulationResult:
    return simulate_experiment(
        SimulationConfig(
            n_units=6000,
            n_covariates=5,
            n_informative=3,
            outcome="binary",
            baseline=0.2,
            average_effect=0.05,
            effect_scale=0.03,
            seed=13,
        )
    )


@pytest.fixture
def frame(randomised: SimulationResult) -> pd.DataFrame:
    return estimation_frame(randomised)

"""Dataset loaders, the benchmark pairing, and the warehouse."""

from __future__ import annotations

import pandas as pd
import pytest

from cip.data import Warehouse, available_datasets, load_dataset, load_lalonde_benchmark
from cip.errors import DataError


def test_every_declared_dataset_loads() -> None:
    """A spec that names a dataset the package cannot load is a broken promise."""
    for spec in available_datasets():
        frame, loaded = load_dataset(spec.name)
        assert not frame.empty
        assert loaded.name == spec.name


def test_declared_columns_actually_exist() -> None:
    for spec in available_datasets():
        frame, _ = load_dataset(spec.name)
        for column in (*spec.covariates, spec.outcome):
            if column:
                assert column in frame.columns, f"{spec.name} is missing {column}"


def test_an_unknown_dataset_lists_the_available_ones() -> None:
    with pytest.raises(DataError, match="available:"):
        load_dataset("not_a_dataset")


def test_the_benchmark_keeps_the_experimental_treated_units() -> None:
    """The observational sample must contain the same treated people, or the truth moves."""
    experimental, observational, spec = load_lalonde_benchmark()
    treated_experimental = experimental[experimental[spec.treatment] == 1]
    treated_observational = observational[observational[spec.treatment] == 1]
    assert len(treated_experimental) == len(treated_observational)
    assert treated_experimental[spec.outcome].sum() == pytest.approx(
        treated_observational[spec.outcome].sum()
    )


def test_the_benchmark_replaces_only_the_controls() -> None:
    experimental, observational, spec = load_lalonde_benchmark()
    n_control_experimental = int((experimental[spec.treatment] == 0).sum())
    n_control_observational = int((observational[spec.treatment] == 0).sum())
    assert n_control_observational > n_control_experimental * 10


def test_the_experimental_effect_matches_the_published_benchmark() -> None:
    """Dehejia and Wahba report roughly $1,794; a different number means a different sample."""
    experimental, _, spec = load_lalonde_benchmark()
    treated = experimental.loc[experimental[spec.treatment] == 1, spec.outcome]
    control = experimental.loc[experimental[spec.treatment] == 0, spec.outcome]
    assert treated.mean() - control.mean() == pytest.approx(1794.0, abs=25.0)


def test_the_naive_observational_comparison_is_badly_wrong() -> None:
    """The bias this platform exists to measure; if it vanishes, the pairing broke."""
    _, observational, spec = load_lalonde_benchmark()
    treated = observational.loc[observational[spec.treatment] == 1, spec.outcome]
    control = observational.loc[observational[spec.treatment] == 0, spec.outcome]
    assert treated.mean() - control.mean() < -5000


def test_the_warehouse_round_trips_a_result() -> None:
    with Warehouse(":memory:") as warehouse:
        rows = pd.DataFrame(
            [
                {
                    "run_name": "test",
                    "created_at": pd.Timestamp("2026-01-01"),
                    "dataset": "sim",
                    "estimand": "ATE",
                    "method": "ipw",
                    "estimate": 1.5,
                }
            ]
        )
        assert warehouse.write("estimates", rows) == 1
        stored = warehouse.read("estimates", run_name="test")
        assert len(stored) == 1
        assert stored["estimate"].iloc[0] == pytest.approx(1.5)


def test_an_unknown_table_is_rejected() -> None:
    with Warehouse(":memory:") as warehouse, pytest.raises(DataError, match="unknown table"):
        warehouse.write("not_a_table", pd.DataFrame([{"a": 1}]))


def test_writing_nothing_is_not_an_error() -> None:
    with Warehouse(":memory:") as warehouse:
        assert warehouse.write("estimates", pd.DataFrame()) == 0

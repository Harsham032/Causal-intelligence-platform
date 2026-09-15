"""Configuration loading, validation and overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from cip.config import AnalysisConfig, load_settings
from cip.errors import ConfigurationError


def test_both_shipped_configurations_load(repo_root: Path) -> None:
    for name in ("default.yaml", "fast.yaml"):
        config = AnalysisConfig.from_yaml(repo_root / "configs" / name)
        assert config.run.seed > 0
        assert 0.0 < config.design.alpha < 1.0


def test_the_fast_configuration_is_actually_faster(repo_root: Path) -> None:
    """CI runs the fast config; if it is not smaller it is not serving its purpose."""
    default = AnalysisConfig.from_yaml(repo_root / "configs" / "default.yaml")
    fast = AnalysisConfig.from_yaml(repo_root / "configs" / "fast.yaml")
    assert fast.frequentist.bootstrap_resamples < default.frequentist.bootstrap_resamples
    assert fast.bayesian.draws < default.bayesian.draws
    assert fast.bayesian.chains <= default.bayesian.chains


def test_a_missing_file_says_so() -> None:
    with pytest.raises(ConfigurationError, match="no configuration file"):
        AnalysisConfig.from_yaml("configs/does-not-exist.yaml")


def test_overrides_are_parsed_as_yaml_not_strings(repo_root: Path) -> None:
    config = AnalysisConfig.from_yaml(repo_root / "configs" / "fast.yaml")
    updated = config.with_overrides({"design.alpha": "0.01", "causal.n_folds": "10"})
    assert updated.design.alpha == 0.01
    assert updated.causal.n_folds == 10
    assert isinstance(updated.causal.n_folds, int)


def test_an_unknown_override_key_is_rejected(config: AnalysisConfig) -> None:
    """Silently ignoring a typo means a run that does not do what its command said."""
    with pytest.raises(ConfigurationError, match="unknown configuration key"):
        config.with_overrides({"design.alfa": "0.01"})


def test_an_invalid_value_is_rejected(config: AnalysisConfig) -> None:
    with pytest.raises(ConfigurationError):
        config.with_overrides({"design.alpha": "1.5"})


def test_trim_bounds_must_be_ordered(config: AnalysisConfig) -> None:
    with pytest.raises(ConfigurationError):
        config.with_overrides({"causal.trim_propensity": "[0.9, 0.1]"})


def test_settings_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIP_WAREHOUSE_PATH", "data/processed/custom.duckdb")
    assert load_settings().warehouse_path == "data/processed/custom.duckdb"

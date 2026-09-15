"""Configuration.

Two kinds of setting, deliberately kept apart.

*Analysis configuration* - what a run does - lives in YAML and is overridable
per run, so an experiment is reproducible from a file plus a seed.

*Deployment settings* - where things connect - come from ``CIP_*`` environment
variables, so a container needs no file edits. Credentials are never read from
YAML, which is a file people commit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .errors import ConfigurationError

MultipleTesting = Literal["none", "bonferroni", "benjamini-hochberg"]
PropensityModel = Literal["logistic", "gradient-boosting"]


class RunConfig(BaseModel):
    name: str = "default"
    seed: int = 20260101
    output_dir: str = "reports"


class DesignConfig(BaseModel):
    """Parameters for sample-size and power calculations."""

    alpha: float = 0.05
    power: float = 0.80
    two_sided: bool = True
    relative_mde: float = 0.10
    balance_threshold: float = 0.10

    @field_validator("alpha", "power")
    @classmethod
    def _in_unit_interval(cls, value: float) -> float:
        if not 0.0 < value < 1.0:
            raise ValueError("must lie strictly between 0 and 1")
        return value

    @field_validator("relative_mde")
    @classmethod
    def _positive_mde(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("the minimum detectable effect must be positive")
        return value


class FrequentistConfig(BaseModel):
    multiple_testing: MultipleTesting = "benjamini-hochberg"
    bootstrap_resamples: int = 2000


class BayesianConfig(BaseModel):
    draws: int = 2000
    tune: int = 1000
    chains: int = 4
    target_accept: float = 0.9
    rope_relative: float = 0.01

    @field_validator("draws", "tune", "chains")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be positive")
        return value


class CausalConfig(BaseModel):
    propensity_model: PropensityModel = "logistic"
    trim_propensity: tuple[float, float] = (0.01, 0.99)
    n_folds: int = 5
    n_estimators: int = 500
    min_samples_leaf: int = 20

    @model_validator(mode="after")
    def _ordered_trim(self) -> CausalConfig:
        low, high = self.trim_propensity
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("trim_propensity must be an increasing pair inside [0, 1]")
        return self


class UpliftConfig(BaseModel):
    n_bins: int = 20
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.1


class DecisionConfig(BaseModel):
    min_probability_of_benefit: float = 0.95
    min_relative_effect: float = 0.01


class AnalysisConfig(BaseModel):
    """Everything a run needs, loaded from YAML."""

    run: RunConfig = Field(default_factory=RunConfig)
    design: DesignConfig = Field(default_factory=DesignConfig)
    frequentist: FrequentistConfig = Field(default_factory=FrequentistConfig)
    bayesian: BayesianConfig = Field(default_factory=BayesianConfig)
    causal: CausalConfig = Field(default_factory=CausalConfig)
    uplift: UpliftConfig = Field(default_factory=UpliftConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AnalysisConfig:
        file = Path(path)
        if not file.is_file():
            raise ConfigurationError(f"no configuration file at {file}")
        try:
            payload = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"{file} is not valid YAML: {exc}") from exc
        try:
            return cls.model_validate(payload)
        except Exception as exc:
            raise ConfigurationError(f"{file} is not a valid configuration: {exc}") from exc

    def with_overrides(self, overrides: dict[str, str]) -> AnalysisConfig:
        """Apply ``section.key=value`` overrides from the command line.

        Values are parsed as YAML so that numbers, booleans and lists arrive as
        the types the schema expects rather than as strings.
        """
        if not overrides:
            return self
        payload = self.model_dump()
        for dotted, raw in overrides.items():
            parts = dotted.split(".")
            if len(parts) < 2:
                raise ConfigurationError(f"override '{dotted}' must be section.key")
            cursor: Any = payload
            for part in parts[:-1]:
                if part not in cursor:
                    raise ConfigurationError(f"unknown configuration section '{part}'")
                cursor = cursor[part]
            if parts[-1] not in cursor:
                raise ConfigurationError(f"unknown configuration key '{dotted}'")
            try:
                cursor[parts[-1]] = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ConfigurationError(f"override '{dotted}={raw}' is not valid: {exc}") from exc
        try:
            return AnalysisConfig.model_validate(payload)
        except Exception as exc:
            raise ConfigurationError(f"overrides produced an invalid configuration: {exc}") from exc


class Settings(BaseSettings):
    """Deployment settings sourced from the environment."""

    model_config = SettingsConfigDict(
        env_prefix="CIP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: str = "development"
    log_level: str = "INFO"
    data_dir: str = "data"
    warehouse_path: str = "data/processed/cip.duckdb"
    postgres_url: str | None = None
    criteo_url: str = (
        "https://criteo-bucket.s3.eu-central-1.amazonaws.com/criteo-research-uplift-v2.1.csv.gz"
    )
    dashboard_port: int = 8501

    @property
    def is_production(self) -> bool:
        return self.env.lower() in {"prod", "production"}


def load_settings() -> Settings:
    """Read deployment settings from the environment."""
    return Settings()

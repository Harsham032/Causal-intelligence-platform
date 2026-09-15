#!/usr/bin/env python
"""Analyse a randomised experiment end to end.

Design check, frequentist estimate, Bayesian posterior, and a recommendation -
in that order, because each step can invalidate the ones after it. An
experiment whose arms are imbalanced is not a randomised experiment, and there
is no point reading its p-value.

Usage::

    python scripts/run_experiment.py --dataset nsw
    python scripts/run_experiment.py --dataset thornton --config configs/fast.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cip.bayesian import beta_binomial_test, normal_test
from cip.config import AnalysisConfig, load_settings
from cip.data import Warehouse, load_dataset
from cip.decision import recommend
from cip.design import check_balance, minimum_detectable_effect, power_for_sample_size
from cip.frequentist import difference_in_means, difference_in_proportions
from cip.logging_utils import configure_logging, get_logger

logger = get_logger("experiment")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="nsw")
    parser.add_argument(
        "--no-bayesian", action="store_true", help="skip the posterior (MCMC is slow)"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-warehouse", action="store_true")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    overrides = dict(item.split("=", 1) for item in args.set)
    config = AnalysisConfig.from_yaml(args.config).with_overrides(overrides)
    settings = load_settings()

    frame, spec = load_dataset(args.dataset)
    frame = frame.dropna(subset=[spec.treatment, spec.outcome])
    treated = frame.loc[frame[spec.treatment] == 1, spec.outcome].to_numpy(dtype=float)
    control = frame.loc[frame[spec.treatment] == 0, spec.outcome].to_numpy(dtype=float)
    binary = bool(np.isin(frame[spec.outcome].to_numpy(), (0.0, 1.0)).all())

    print(f"\n{spec.name}: {spec.description}")
    print(
        f"randomised: {spec.randomised} | {len(treated)} treated, {len(control)} control | "
        f"outcome {'binary' if binary else 'continuous'}\n"
    )

    # 1. Design. An imbalanced experiment is an observational study in disguise.
    observed_share = len(treated) / (len(treated) + len(control))
    balance = check_balance(
        frame,
        spec.treatment,
        list(spec.covariates),
        threshold=config.design.balance_threshold,
        expected_share=observed_share,
    )
    print("DESIGN")
    print(f"  assignment share {observed_share:.4f} (tested against the realised design)")
    print(f"  balance: {balance.reason()}")
    if balance.worst is not None:
        print(
            f"  worst covariate: {balance.worst.covariate} (SMD {balance.worst.standardised_difference:+.4f})"
        )

    pooled_sd = float(np.sqrt((treated.var(ddof=1) + control.var(ddof=1)) / 2.0))
    per_arm = min(len(treated), len(control))
    mde = minimum_detectable_effect(per_arm, alpha=config.design.alpha, power=config.design.power)
    print(
        f"  smallest detectable effect at this size: {mde:.4f} sd = {mde * pooled_sd:,.4g} in outcome units"
    )

    # 2. Frequentist.
    if binary:
        freq = difference_in_proportions(treated, control, metric=spec.outcome)
    else:
        freq = difference_in_means(treated, control, metric=spec.outcome)
    achieved = power_for_sample_size(
        abs(freq.absolute_effect) / pooled_sd if pooled_sd else 0.0,
        per_arm,
        alpha=config.design.alpha,
    )
    print("\nFREQUENTIST")
    print(
        f"  effect {freq.absolute_effect:,.4g} ({freq.relative_effect:+.2%}) "
        f"95% CI [{freq.ci_lower:,.4g}, {freq.ci_upper:,.4g}]  p={freq.pvalue:.4g}"
    )
    print(f"  method {freq.method}; power to detect an effect this size: {achieved.power:.3f}")

    # 3. Bayesian.
    bayes = None
    if not args.no_bayesian:
        if binary:
            bayes = beta_binomial_test(
                treated,
                control,
                metric=spec.outcome,
                rope_relative=config.bayesian.rope_relative,
                seed=config.run.seed,
            )
        else:
            bayes = normal_test(
                treated,
                control,
                metric=spec.outcome,
                likelihood="normal",
                draws=config.bayesian.draws,
                tune=config.bayesian.tune,
                chains=config.bayesian.chains,
                target_accept=config.bayesian.target_accept,
                rope_relative=config.bayesian.rope_relative,
                seed=config.run.seed,
            )
        print("\nBAYESIAN")
        print(
            f"  effect {bayes.effect_mean:,.4g}  95% CrI [{bayes.ci_lower:,.4g}, {bayes.ci_upper:,.4g}]"
        )
        print(
            f"  P(treatment > control) = {bayes.probability_of_benefit:.4f}  "
            f"P(practically equivalent) = {bayes.probability_practically_equivalent:.4f}"
        )
        print(f"  expected loss from shipping {bayes.expected_loss:,.4g}  |  {bayes.method}")
        if bayes.diagnostics:
            print(
                "  diagnostics: " + ", ".join(f"{k}={v:.3f}" for k, v in bayes.diagnostics.items())
            )

    # 4. Decision.
    decision = recommend(
        freq.absolute_effect,
        (freq.ci_lower, freq.ci_upper),
        baseline=freq.control_mean,
        probability_of_benefit=bayes.probability_of_benefit if bayes else None,
        min_probability=config.decision.min_probability_of_benefit,
        min_relative_effect=config.decision.min_relative_effect,
    )
    print(f"\nDECISION: {decision.verdict.upper()}")
    print(f"  {decision.reason}\n")

    report = {
        "dataset": spec.name,
        "randomised": spec.randomised,
        "design": {
            "balance_passed": balance.passed,
            "balance_reason": balance.reason(),
            "imbalanced_covariates": [c.to_dict() for c in balance.imbalanced],
            "minimum_detectable_effect_sd": mde,
            "achieved_power": achieved.power,
        },
        "frequentist": freq.to_dict(),
        "bayesian": bayes.to_dict() if bayes else None,
        "decision": decision.to_dict(),
    }
    output_dir = Path(args.output_dir or config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"experiment_{spec.name}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if not args.no_warehouse:
        with Warehouse(settings.warehouse_path) as warehouse:
            now = datetime.now(UTC)
            warehouse.write(
                "estimates",
                pd.DataFrame(
                    [
                        {
                            "run_name": config.run.name,
                            "created_at": now,
                            "dataset": spec.name,
                            "estimand": "ATE",
                            "method": freq.method,
                            "estimate": freq.absolute_effect,
                            "standard_error": None,
                            "ci_lower": freq.ci_lower,
                            "ci_upper": freq.ci_upper,
                            "n_units": freq.n_treated + freq.n_control,
                            "n_treated": freq.n_treated,
                            "truth": None,
                            "bias": None,
                            "covers_truth": None,
                            "notes": "randomised experiment",
                        }
                    ]
                ),
            )
            warehouse.write(
                "decisions",
                pd.DataFrame(
                    [
                        {
                            "run_name": config.run.name,
                            "created_at": now,
                            "dataset": spec.name,
                            "recommendation": decision.verdict,
                            "effect": decision.effect,
                            "probability": decision.probability_of_benefit,
                            "reason": decision.reason,
                        }
                    ]
                ),
            )
            warehouse.write(
                "diagnostics",
                pd.DataFrame(
                    [
                        {
                            "run_name": config.run.name,
                            "created_at": now,
                            "dataset": spec.name,
                            "check_name": "covariate_balance",
                            "passed": balance.passed,
                            "statistic": (
                                abs(balance.worst.standardised_difference)
                                if balance.worst
                                else None
                            ),
                            "threshold": config.design.balance_threshold,
                            "detail": balance.reason(),
                        }
                    ]
                ),
            )

    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

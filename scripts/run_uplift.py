#!/usr/bin/env python
"""Fit uplift models and score them against a known ranking.

Runs on the simulator by default, because scoring a *ranking* of individual
effects needs those effects, and no real dataset contains them. The oracle row
is the point: it is the ceiling the metric can reach on this data, without which
a Qini of 0.08 is uninterpretable.

Usage::

    python scripts/run_uplift.py
    python scripts/run_uplift.py --units 20000 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cip.config import AnalysisConfig
from cip.data.simulator import SimulationConfig, estimation_frame, simulate_experiment
from cip.decision import targeting_policy
from cip.logging_utils import configure_logging, get_logger
from cip.uplift import fit_uplift, qini_curve, uplift_by_decile

logger = get_logger("uplift")

LEARNERS = ("s-learner", "t-learner", "dr-learner")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--units", type=int, default=8000)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--cost", type=float, default=0.3, help="cost of treating one unit")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = AnalysisConfig.from_yaml(args.config)

    result = simulate_experiment(
        SimulationConfig(
            n_units=args.units,
            n_covariates=8,
            n_informative=4,
            effect="heterogeneous",
            average_effect=1.0,
            effect_scale=1.0,
            seed=config.run.seed,
        )
    )
    frame = estimation_frame(result)
    truth = result.true_cate
    outcome, treatment = frame[result.outcome], frame[result.treatment]

    rows = []
    best_name, best_qini, best_predictions = "", -np.inf, None
    for learner in LEARNERS:
        model = fit_uplift(
            frame,
            outcome=result.outcome,
            treatment=result.treatment,
            covariates=list(result.covariates),
            learner=learner,
            n_estimators=config.uplift.n_estimators,
            max_depth=config.uplift.max_depth,
            learning_rate=config.uplift.learning_rate,
            n_folds=args.folds,
            seed=config.run.seed,
        )
        curve = qini_curve(outcome, treatment, model.predicted_uplift)
        rows.append(
            {
                "learner": learner,
                "qini": curve.qini_coefficient,
                "auuc": curve.auuc,
                "spearman_vs_truth": float(
                    stats.spearmanr(model.predicted_uplift, truth).statistic
                ),
                "rmse_vs_truth": float(np.sqrt(((model.predicted_uplift - truth) ** 2).mean())),
                "mean_predicted_uplift": float(model.predicted_uplift.mean()),
            }
        )
        if curve.qini_coefficient > best_qini:
            best_name, best_qini, best_predictions = (
                learner,
                curve.qini_coefficient,
                model.predicted_uplift,
            )

    assert best_predictions is not None
    oracle = qini_curve(outcome, treatment, truth)
    rng = np.random.default_rng(config.run.seed)
    random = qini_curve(outcome, treatment, rng.normal(size=len(frame)))
    best_curve = qini_curve(outcome, treatment, best_predictions)
    deciles = uplift_by_decile(outcome, treatment, best_predictions, n_bins=config.uplift.n_bins)
    policy = targeting_policy(best_predictions, cost_per_treatment=args.cost, value_per_outcome=1.0)

    report = {
        "units": int(len(frame)),
        "true_ate": result.true_ate,
        "true_cate_sd": float(truth.std(ddof=1)),
        "learners": rows,
        "best": {"learner": best_name, "qini": best_qini},
        "oracle_qini": oracle.qini_coefficient,
        "random_qini": random.qini_coefficient,
        "share_of_oracle": (
            best_qini / oracle.qini_coefficient if oracle.qini_coefficient else float("nan")
        ),
        "curve": best_curve.to_frame().iloc[:: max(1, len(frame) // 200)].to_dict("records"),
        "deciles": deciles.round(6).to_dict("records"),
        "targeting": {
            "cost_per_treatment": args.cost,
            "treated_share": float(policy["treat"].mean()),
            "expected_value_if_treating_all": float(policy["expected_value"].sum()),
            "expected_value_under_policy": float(
                policy.loc[policy["treat"], "expected_value"].sum()
            ),
        },
    }

    output_dir = Path(args.output_dir or config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "uplift_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print(
        f"\n{len(frame):,} units, true ATE {result.true_ate:.4f}, true CATE sd {truth.std(ddof=1):.4f}\n"
    )
    header = f"{'learner':12s} {'Qini':>9s} {'% of oracle':>12s} {'spearman':>10s} {'RMSE':>8s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        share = row["qini"] / oracle.qini_coefficient if oracle.qini_coefficient else float("nan")
        print(
            f"{row['learner']:12s} {row['qini']:9.4f} {share:11.1%} {row['spearman_vs_truth']:10.4f} {row['rmse_vs_truth']:8.4f}"
        )
    print(
        f"{'random':12s} {random.qini_coefficient:9.4f} {random.qini_coefficient / oracle.qini_coefficient:11.1%}"
    )
    print(f"{'oracle':12s} {oracle.qini_coefficient:9.4f} {1.0:11.1%}")
    print(
        f"\ntargeting at a cost of {args.cost}: treat {report['targeting']['treated_share']:.1%} of units"
    )
    print(
        f"  expected value treating everyone: {report['targeting']['expected_value_if_treating_all']:,.1f}"
    )
    print(
        f"  expected value under the policy:  {report['targeting']['expected_value_under_policy']:,.1f}"
    )
    print(f"\nreport: {output_dir / 'uplift_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

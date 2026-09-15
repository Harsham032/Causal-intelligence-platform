#!/usr/bin/env python
"""Score every causal estimator against a known treatment effect.

Two benchmarks, because they answer different questions.

``--benchmark lalonde`` uses a randomised experiment to supply the truth and a
survey comparison group to fabricate the observational study. The bias it
measures is real: the same programme, the same people, a comparison group that
is wrong in exactly the way real observational data is wrong.

``--benchmark simulation`` generates data where every unit's effect is known, so
interval coverage can be measured across repeated runs - which no single real
dataset can support.

Usage::

    python scripts/run_benchmark.py --benchmark lalonde
    python scripts/run_benchmark.py --benchmark simulation --runs 50
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

from cip.causal import (
    causal_forest,
    double_machine_learning,
    estimate_propensity,
    inverse_probability_weighting,
    matched_estimate,
    trim_to_overlap,
)
from cip.config import AnalysisConfig, load_settings
from cip.data import Warehouse, load_lalonde_benchmark
from cip.data.simulator import SimulationConfig, estimation_frame, simulate_experiment
from cip.evaluation import coverage_of_intervals, estimator_bias
from cip.frequentist import difference_in_means
from cip.logging_utils import configure_logging, get_logger

logger = get_logger("benchmark")


def lalonde_benchmark(config: AnalysisConfig) -> dict:
    """Every estimator against the experimental answer."""
    experimental, observational, spec = load_lalonde_benchmark()
    covariates = list(spec.covariates)

    treated = experimental.loc[experimental[spec.treatment] == 1, spec.outcome]
    control = experimental.loc[experimental[spec.treatment] == 0, spec.outcome]
    benchmark = difference_in_means(treated, control, metric=spec.outcome)
    truth = benchmark.absolute_effect
    logger.info("experimental_benchmark", effect=round(truth, 2))

    propensity = estimate_propensity(
        observational, spec.treatment, covariates, model=config.causal.propensity_model
    )
    mask, overlap = trim_to_overlap(
        propensity.scores,
        observational[spec.treatment].to_numpy(),
        trim=config.causal.trim_propensity,
    )
    trimmed = observational[mask].reset_index(drop=True)
    trimmed_scores = propensity.scores[mask]

    rows: list[dict] = []

    def record(method: str, estimate: float, lower: float, upper: float, note: str = "") -> None:
        rows.append(
            {
                "method": method,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "truth": truth,
                "bias": estimate - truth,
                "covers_truth": bool(lower <= truth <= upper),
                "notes": note,
            }
        )

    full_treated = observational.loc[observational[spec.treatment] == 1, spec.outcome]
    full_control = observational.loc[observational[spec.treatment] == 0, spec.outcome]
    naive_full = difference_in_means(full_treated, full_control)
    record(
        "naive difference in means",
        naive_full.absolute_effect,
        naive_full.ci_lower,
        naive_full.ci_upper,
        "full observational sample",
    )

    kept_treated = trimmed.loc[trimmed[spec.treatment] == 1, spec.outcome]
    kept_control = trimmed.loc[trimmed[spec.treatment] == 0, spec.outcome]
    naive_trimmed = difference_in_means(kept_treated, kept_control)
    record(
        "naive, common support",
        naive_trimmed.absolute_effect,
        naive_trimmed.ci_lower,
        naive_trimmed.ci_upper,
        "after overlap trimming, no model",
    )

    ipw = inverse_probability_weighting(
        trimmed[spec.outcome], trimmed[spec.treatment], trimmed_scores, estimand="ATT"
    )
    record(
        "IPW (ATT)",
        ipw.estimate,
        ipw.ci_lower,
        ipw.ci_upper,
        f"ESS {ipw.effective_sample_size:.0f}",
    )

    matched = matched_estimate(trimmed[spec.outcome], trimmed[spec.treatment], trimmed_scores)
    record("1-NN propensity matching", matched.estimate, matched.ci_lower, matched.ci_upper)

    dml = double_machine_learning(
        trimmed,
        outcome=spec.outcome,
        treatment=spec.treatment,
        covariates=covariates,
        n_folds=config.causal.n_folds,
        n_estimators=config.causal.n_estimators,
        min_samples_leaf=config.causal.min_samples_leaf,
        seed=config.run.seed,
    )
    record("double machine learning", dml.estimate, dml.ci_lower, dml.ci_upper)

    forest = causal_forest(
        trimmed,
        outcome=spec.outcome,
        treatment=spec.treatment,
        covariates=covariates,
        n_folds=config.causal.n_folds,
        n_estimators=config.causal.n_estimators,
        min_samples_leaf=config.causal.min_samples_leaf,
        seed=config.run.seed,
    )
    record("causal forest", forest.estimate, forest.ci_lower, forest.ci_upper)

    return {
        "benchmark": "lalonde",
        "truth": truth,
        "truth_interval": [benchmark.ci_lower, benchmark.ci_upper],
        "experimental_n": {"treated": benchmark.n_treated, "control": benchmark.n_control},
        "propensity_auc": propensity.auc,
        "overlap": overlap.to_dict(),
        "trimmed_n": {
            "treated": int((trimmed[spec.treatment] == 1).sum()),
            "control": int((trimmed[spec.treatment] == 0).sum()),
        },
        "results": rows,
    }


def simulation_benchmark(config: AnalysisConfig, runs: int) -> dict:
    """Bias and interval coverage over repeated simulated studies."""
    estimates: dict[str, list[float]] = {"naive": [], "IPW": [], "matching": []}
    intervals: dict[str, list[tuple[float, float]]] = {"naive": [], "IPW": [], "matching": []}
    truths: list[float] = []

    for run in range(runs):
        result = simulate_experiment(
            SimulationConfig(
                n_units=3000,
                n_covariates=6,
                n_informative=3,
                confounding=1.5,
                seed=config.run.seed + run,
            )
        )
        frame = estimation_frame(result)
        truths.append(result.true_ate)

        treated = frame.loc[frame[result.treatment] == 1, result.outcome]
        control = frame.loc[frame[result.treatment] == 0, result.outcome]
        naive = difference_in_means(treated, control)
        estimates["naive"].append(naive.absolute_effect)
        intervals["naive"].append((naive.ci_lower, naive.ci_upper))

        propensity = estimate_propensity(frame, result.treatment, list(result.covariates))
        ipw = inverse_probability_weighting(
            frame[result.outcome], frame[result.treatment], propensity.scores
        )
        estimates["IPW"].append(ipw.estimate)
        intervals["IPW"].append((ipw.ci_lower, ipw.ci_upper))

        matched = matched_estimate(
            frame[result.outcome], frame[result.treatment], propensity.scores
        )
        estimates["matching"].append(matched.estimate)
        intervals["matching"].append((matched.ci_lower, matched.ci_upper))

    truth = float(np.mean(truths))
    summary = []
    for method, values in estimates.items():
        bias = estimator_bias(values, truth)
        lowers = [lo for lo, _ in intervals[method]]
        uppers = [hi for _, hi in intervals[method]]
        coverage = coverage_of_intervals(lowers, uppers, truth, nominal=0.95)
        summary.append({"method": method, **bias.to_dict(), "coverage": coverage.to_dict()})

    return {"benchmark": "simulation", "runs": runs, "truth": truth, "results": summary}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--benchmark", choices=["lalonde", "simulation", "both"], default="both")
    parser.add_argument(
        "--runs", type=int, default=50, help="simulated studies for the coverage benchmark"
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-warehouse", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = AnalysisConfig.from_yaml(args.config)
    settings = load_settings()

    report: dict = {"config": args.config, "seed": config.run.seed}
    if args.benchmark in ("lalonde", "both"):
        report["lalonde"] = lalonde_benchmark(config)
    if args.benchmark in ("simulation", "both"):
        report["simulation"] = simulation_benchmark(config, args.runs)

    output_dir = Path(args.output_dir or config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    if "lalonde" in report:
        block = report["lalonde"]
        print(
            f"\nExperimental benchmark: {block['truth']:,.0f} "
            f"(95% CI {block['truth_interval'][0]:,.0f} to {block['truth_interval'][1]:,.0f})"
        )
        print(f"Propensity AUC {block['propensity_auc']:.4f}; {block['overlap']['detail']}")
        print(
            f"After trimming: {block['trimmed_n']['treated']} treated, {block['trimmed_n']['control']} control\n"
        )
        header = f"{'method':28s} {'estimate':>10s} {'bias':>10s} {'covers':>8s}"
        print(header)
        print("-" * len(header))
        for row in block["results"]:
            print(
                f"{row['method']:28s} {row['estimate']:10,.0f} {row['bias']:10,.0f} "
                f"{('yes' if row['covers_truth'] else 'NO'):>8s}"
            )

    if "simulation" in report:
        block = report["simulation"]
        print(
            f"\nSimulation benchmark over {block['runs']} studies, true effect {block['truth']:.4f}\n"
        )
        header = f"{'method':12s} {'bias':>9s} {'rmse':>9s} {'coverage':>10s} {'nominal':>9s}"
        print(header)
        print("-" * len(header))
        for row in block["results"]:
            print(
                f"{row['method']:12s} {row['bias']:9.4f} {row['rmse']:9.4f} "
                f"{row['coverage']['empirical']:10.3f} {row['coverage']['nominal']:9.2f}"
            )

    if not args.no_warehouse:
        with Warehouse(settings.warehouse_path) as warehouse:
            now = datetime.now(UTC)
            if "lalonde" in report:
                frame = pd.DataFrame(report["lalonde"]["results"])
                frame.insert(0, "run_name", config.run.name)
                frame.insert(1, "created_at", now)
                frame.insert(2, "dataset", "lalonde")
                frame.insert(3, "estimand", "ATT")
                warehouse.write(
                    "estimates",
                    frame[
                        [
                            "run_name",
                            "created_at",
                            "dataset",
                            "estimand",
                            "method",
                            "estimate",
                            "ci_lower",
                            "ci_upper",
                            "truth",
                            "bias",
                            "covers_truth",
                            "notes",
                        ]
                    ],
                )

    print(f"\nreport: {output_dir / 'benchmark_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""End-to-end walkthrough: design, estimate, score, decide.

Runs in about a minute with no network access. Each section prints the number
and the reason it matters, in the order a real analysis would take them.

    python examples/end_to_end.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cip.bayesian import beta_binomial_test
from cip.causal import (
    estimate_propensity,
    inverse_probability_weighting,
    matched_estimate,
    trim_to_overlap,
)
from cip.data import load_lalonde_benchmark
from cip.data.simulator import SimulationConfig, estimation_frame, simulate_experiment
from cip.decision import recommend, targeting_policy
from cip.design import check_balance, sample_size_for_proportions
from cip.errors import AssumptionError
from cip.evaluation import unmeasured_confounding
from cip.frequentist import adjust_pvalues, difference_in_means
from cip.logging_utils import configure_logging
from cip.uplift import fit_uplift, qini_curve


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    configure_logging("WARNING")

    rule("1. Design: is the experiment worth running?")
    sized = sample_size_for_proportions(0.10, 0.05, power=0.80)
    print("To detect a 5% relative lift on a 10% conversion rate at 80% power:")
    print(f"  {sized.per_arm:,} units per arm ({sized.total:,} total)")
    print("  An experiment smaller than this does not return 'no effect'. It returns noise.")

    rule("2. A randomised experiment supplies a truth")
    experimental, observational, spec = load_lalonde_benchmark()
    treated = experimental.loc[experimental[spec.treatment] == 1, spec.outcome]
    control = experimental.loc[experimental[spec.treatment] == 0, spec.outcome]
    benchmark = difference_in_means(treated, control)
    truth = benchmark.absolute_effect
    print(
        f"NSW job training, randomised: ${truth:,.0f} "
        f"(95% CI ${benchmark.ci_lower:,.0f} to ${benchmark.ci_upper:,.0f})"
    )
    print(f"  {benchmark.n_treated} treated, {benchmark.n_control} control")

    balance = check_balance(
        experimental,
        spec.treatment,
        list(spec.covariates),
        expected_share=benchmark.n_treated / (benchmark.n_treated + benchmark.n_control),
    )
    worst = balance.worst
    print(f"  balance: {'passed' if balance.passed else 'failed'}", end="")
    if worst is not None:
        print(f" (worst: {worst.covariate}, SMD {worst.standardised_difference:+.3f})")
    print("  Selected after randomisation on earnings availability - and it shows.")

    rule("3. The same effect, from observational data")
    obs_treated = observational.loc[observational[spec.treatment] == 1, spec.outcome]
    obs_control = observational.loc[observational[spec.treatment] == 0, spec.outcome]
    naive = difference_in_means(obs_treated, obs_control)
    print(f"Naive comparison against survey controls: ${naive.absolute_effect:,.0f}")
    print(f"  bias against the experiment: ${naive.absolute_effect - truth:,.0f}")

    propensity = estimate_propensity(observational, spec.treatment, list(spec.covariates))
    print(f"\nPropensity AUC {propensity.auc:.4f} - the arms are nearly separable.")
    try:
        from cip.causal import check_overlap

        check_overlap(propensity.scores, observational[spec.treatment].to_numpy())
    except AssumptionError as exc:
        print(f"  overlap check REFUSED: {str(exc)[:110]}...")

    mask, overlap = trim_to_overlap(propensity.scores, observational[spec.treatment].to_numpy())
    trimmed = observational[mask].reset_index(drop=True)
    scores = propensity.scores[mask]
    print(
        f"  proceeding explicitly on common support: {mask.sum():,} of {len(observational):,} units"
    )

    kept_t = trimmed.loc[trimmed[spec.treatment] == 1, spec.outcome]
    kept_c = trimmed.loc[trimmed[spec.treatment] == 0, spec.outcome]
    trimmed_naive = difference_in_means(kept_t, kept_c)
    ipw = inverse_probability_weighting(
        trimmed[spec.outcome], trimmed[spec.treatment], scores, estimand="ATT"
    )
    matched = matched_estimate(trimmed[spec.outcome], trimmed[spec.treatment], scores)

    print(f"\n{'method':28s} {'estimate':>10s} {'bias':>10s}")
    for name, value in (
        ("naive, full sample", naive.absolute_effect),
        ("naive, common support", trimmed_naive.absolute_effect),
        ("IPW (ATT)", ipw.estimate),
        ("1-NN matching", matched.estimate),
    ):
        print(f"{name:28s} {value:10,.0f} {value - truth:10,.0f}")
    print("\n  Trimming alone removed most of the bias - before any estimator ran.")
    if not matched.interval_is_trustworthy:
        print(f"  NOTE on matching: {matched.interval_caveat[:100]}...")

    rule("4. How much hidden confounding would overturn this?")
    sensitivity = unmeasured_confounding(ipw.estimate, ipw.standard_error)
    print(f"  {sensitivity.interpretation}")

    rule("5. Many metrics, one experiment")
    rng = np.random.default_rng(0)
    nulls = rng.uniform(size=20)
    raw = adjust_pvalues(nulls, method="none")
    corrected = adjust_pvalues(nulls, method="benjamini-hochberg")
    print(
        f"  20 metrics with no real effect: {len(raw.discoveries)} 'significant' uncorrected, "
        f"{len(corrected.discoveries)} after Benjamini-Hochberg"
    )

    rule("6. A Bayesian answer to the question actually asked")
    trial = simulate_experiment(
        SimulationConfig(
            n_units=6000,
            outcome="binary",
            baseline=0.2,
            average_effect=0.03,
            effect_scale=0.0,
            seed=7,
        )
    )
    frame = estimation_frame(trial)
    t = frame.loc[frame[trial.treatment] == 1, trial.outcome]
    c = frame.loc[frame[trial.treatment] == 0, trial.outcome]
    posterior = beta_binomial_test(t, c, rope_relative=0.01)
    print(
        f"  effect {posterior.effect_mean:+.4f}, 95% CrI "
        f"[{posterior.ci_lower:+.4f}, {posterior.ci_upper:+.4f}]"
    )
    print(f"  P(treatment > control) = {posterior.probability_of_benefit:.4f}")
    print(f"  P(practically equivalent) = {posterior.probability_practically_equivalent:.4f}")

    rule("7. Who responds, not whether anyone does")
    heterogeneous = simulate_experiment(
        SimulationConfig(
            n_units=4000,
            n_covariates=6,
            n_informative=3,
            effect="heterogeneous",
            average_effect=1.0,
            effect_scale=1.0,
            seed=9,
        )
    )
    hframe = estimation_frame(heterogeneous)
    model = fit_uplift(
        hframe,
        outcome=heterogeneous.outcome,
        treatment=heterogeneous.treatment,
        covariates=list(heterogeneous.covariates),
        learner="s-learner",
        n_estimators=80,
        max_depth=3,
        n_folds=3,
    )
    fitted = qini_curve(
        hframe[heterogeneous.outcome], hframe[heterogeneous.treatment], model.predicted_uplift
    )
    oracle = qini_curve(
        hframe[heterogeneous.outcome], hframe[heterogeneous.treatment], heterogeneous.true_cate
    )
    print(
        f"  Qini {fitted.qini_coefficient:.4f} against an oracle ceiling of "
        f"{oracle.qini_coefficient:.4f} ({fitted.qini_coefficient / oracle.qini_coefficient:.1%})"
    )
    policy = targeting_policy(model.predicted_uplift, cost_per_treatment=0.3)
    everyone = float(policy["expected_value"].sum())
    targeted = float(policy.loc[policy["treat"], "expected_value"].sum())
    print(
        f"  treating everyone: {everyone:,.0f}; treating the profitable "
        f"{policy['treat'].mean():.0%}: {targeted:,.0f}"
    )

    rule("8. An estimate is not a decision")
    decision = recommend(
        ipw.estimate,
        (ipw.ci_lower, ipw.ci_upper),
        baseline=float(kept_c.mean()),
        min_relative_effect=0.01,
    )
    print(f"  {decision.verdict.upper()}: {decision.reason}")
    print("\nEvery number above came from this run. See docs/results.md for the full set.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

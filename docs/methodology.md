# Methodology

How the numbers in `docs/results.md` are produced, and what they can and cannot
support.

## 1. What is being estimated

Three estimands appear, and conflating them is the most common error in applied
causal work:

| Estimand | Question | Where it is used |
| --- | --- | --- |
| **ATE** | What is the average effect across everyone? | Randomised experiments; simulation benchmarks |
| **ATT** | What is the average effect among those who were treated? | The LaLonde benchmark, IPW and matching |
| **CATE** | What is the effect for *this* unit? | Uplift modelling, causal forests |

They differ whenever the effect is heterogeneous and treatment is not random. In
the simulator with confounding, the true ATE is 1.000 and the true ATT is 1.187
— the people who took treatment were the people it helped more. Reporting one
and calling it the other is a 19% error before any estimation happens.

## 2. Identification, and which assumptions are checkable

Every causal estimate rests on assumptions. They divide sharply:

**Checkable, and checked.**

*Overlap* — for every covariate profile, both arms are possible. Testable from
the propensity distribution, and it fails routinely. `check_overlap` raises when
enforcing it would discard more than half the sample; on the LaLonde
observational data it would discard 89.5%, and it raises.

*Parallel trends* (difference-in-differences) — not verifiable, since it concerns
a world we do not observe, but the pre-treatment period gives evidence about it.
`difference_in_differences` regresses outcome on a group-by-time interaction
using pre-treatment periods only, and can be configured to refuse.

*Randomisation* — a claim about the assignment mechanism, testable against the
realised assignment. `check_balance` tests both covariate balance and the
assignment share.

**Not checkable, and therefore bounded instead.**

*Unconfoundedness* — every covariate driving both treatment and outcome is in
the model. If the confounder were measured, it would not be unmeasured. The only
honest response is sensitivity analysis: not "is this true?" but "how wrong
would it have to be to overturn the result?" `cip.evaluation.sensitivity`
answers that in two ways.

## 3. Why the standardised mean difference, not a p-value

Balance is reported as the standardised mean difference — the gap between arms
in pooled standard deviations — rather than as a t-test p-value.

A p-value here answers a question nobody asked. With 16,177 rows, a difference
of no practical consequence is "significant". With 445, a serious imbalance is
not. The statistic moves with sample size while the problem does not. The SMD's
conventional 0.10 action threshold means the same thing at any scale.

## 4. Metric choices

| Metric | Used for | Why this one |
| --- | --- | --- |
| Welch's t | Continuous two-arm comparison | Equal variances across arms is an assumption nobody checks and treatments routinely violate; Welch costs nothing when they are equal |
| Newcombe hybrid score | Proportion differences | The normal approximation can produce bounds outside the feasible range at the low rates where conversion experiments live |
| Arcsine transform | Proportion sample sizing | Its variance does not depend on the rate; the pooled-variance formula understates the requirement near zero |
| Benjamini-Hochberg | Screening many metrics | Controls the false discovery rate and is uniformly more powerful than Bonferroni |
| Bonferroni | Guardrail metrics | Controls the family-wise error rate, which is what matters when one false positive triggers a rollback |
| Qini coefficient | Uplift ranking | Scores groups rather than units, which is the only way to evaluate a prediction that is never observed |
| Kish effective sample size | Weighted estimates | A stabilised estimate can still be carried by a handful of units, and the standard error will not say so |

## 5. Cross-fitting

Both the DML estimator and every uplift learner predict each unit from models
that never saw it.

Without it, a flexible model reproduces the noise in its own training rows, and
that overfitting lands directly in the treatment effect as bias — or, for
uplift, produces a Qini curve measuring memorisation rather than targeting.
`double_machine_learning` rejects `n_folds=1` rather than silently running
without the property it exists to provide.

Orthogonalisation and cross-fitting only work together: DML removes the
covariates' influence from both treatment and outcome and regresses one residual
on the other, so first-stage errors cancel to first order. Drop either half and
the bias returns.

## 6. Reproducibility

- One seed governs the simulator, every model's randomness, the bootstrap and
  the MCMC sampler.
- No data is downloaded at any point.
- Every analysis runs in CI on every change at the reduced configuration, so a
  break cannot reach `main` unnoticed.
- CI asserts that the benchmark still measures a bias — a benchmark reporting
  that the naive comparison is unbiased has lost the thing it exists to measure.

```bash
make install
make experiment   # about 10 seconds
make benchmark    # about 12 minutes at the default configuration
make uplift       # about 20 minutes at the default configuration
```

## 7. Threats to validity

| Threat | Effect | Mitigation, or why not |
| --- | --- | --- |
| Uplift results are simulated | Measures recovery of injected structure, not discovery of real structure | Unavoidable: individual effects are unobservable in principle. Bounded explicitly in results §11 |
| LaLonde estimates are on common support | The estimand is not the original population | Stated at every point of use; the trimming step is a separate explicit call |
| Matching intervals are invalid | Reported coverage 0.10 against a nominal 0.95 | Measured, documented, and flagged on the result object. Abadie-Imbens variance is the fix and is not implemented |
| Unconfoundedness assumed | Any omitted common cause biases every observational estimate here | Not testable. Sensitivity analysis bounds how strong one would need to be |
| Single seed for uplift | The learner ranking could be a favourable draw | The S-learner and DR-learner differ by 0.0006 in Qini, which one seed cannot separate. Listed as a follow-up |
| Two-way fixed effects on staggered adoption | Known to be biased when treatment timing varies | The `castle` panel is staggered; the estimator implemented here is not appropriate for it and Callaway-Sant'Anna is not implemented |
| NSW sample selected post-randomisation | Covariate imbalance in a randomised experiment | Measured and reported: worst SMD −0.304 on `nodegree`, while both earnings variables balance |

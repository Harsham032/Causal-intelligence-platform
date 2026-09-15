# Architecture

## Flow

```
                      ┌──────────────────────────┐
                      │ causaldata (offline)     │
                      │ simulator (known truth)  │
                      └────────────┬─────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
   ┌──────▼───────┐        ┌───────▼────────┐      ┌────────▼────────┐
   │ design       │        │ causal         │      │ uplift          │
   │ power        │        │ propensity     │      │ T / S / DR      │
   │ balance      │        │ overlap guard  │      │ cross-fitted    │
   └──────┬───────┘        │ IPW / matching │      └────────┬────────┘
          │                │ DiD / DML / CF │               │
          │                └───────┬────────┘               │
   ┌──────▼───────┐                │              ┌─────────▼────────┐
   │ frequentist  │                │              │ Qini / AUUC      │
   │ bayesian     │                │              │ decile table     │
   └──────┬───────┘                │              └─────────┬────────┘
          │                        │                        │
          └────────────────────────┼────────────────────────┘
                                   │
                        ┌──────────▼───────────┐
                        │ evaluation           │
                        │ bias / coverage      │
                        │ sensitivity          │
                        └──────────┬───────────┘
                                   │
                        ┌──────────▼───────────┐
                        │ decision             │
                        │ credible? material?  │
                        └──────────┬───────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   │                               │
           ┌───────▼────────┐             ┌────────▼────────┐
           │ DuckDB         │             │ Streamlit       │
           │ warehouse      │────────────▶│ dashboard       │
           └────────────────┘             └─────────────────┘
```

Results flow one way. The pipeline writes; the dashboard reads. Nothing is
recomputed on view.

## Design decisions

### Guards that refuse rather than return a number

The failure mode this platform is built against is not a wrong number — it is a
plausible number that nobody questions. Three guards therefore raise:

- `check_overlap` raises when common support would discard more than half the
  sample. On the LaLonde data it does, at 89.5%. Proceeding requires a separate
  explicit call to `trim_to_overlap`, because trimming changes the estimand.
- `difference_in_differences(refuse_on_pre_trend=True)` raises when the groups
  were already diverging, since that divergence would otherwise be reported as
  the treatment effect.
- `double_machine_learning` rejects `n_folds=1`, which would be DML without
  cross-fitting — the bias it exists to avoid.

### The estimand is never inferred

`inverse_probability_weighting` takes `estimand="ATE"` or `"ATT"` and computes
different weights for each. There is no default that quietly picks one. Under
heterogeneous effects with non-random assignment these differ by 19% in the
simulator, and a function that guesses would be wrong by that much without
saying so.

### Results are stored, not printed

Every analysis writes to DuckDB: estimates with their truth and bias where a
truth exists, diagnostics with pass/fail, decisions with the thresholds they
were judged against. A run six months from now can be compared with this one,
and the dashboard reads rows rather than refitting models.

DuckDB rather than PostgreSQL by default because it is a single file with no
server, which keeps the whole platform reproducible from a clean checkout.
PostgreSQL is supported through the same schema for a shared warehouse.

### The likelihood is a parameter, not a default

`normal_test` exposes `likelihood`. Student-t estimates a robust location shift;
Normal estimates the shift in the mean. On NSW earnings these differ by a third
($1,277 against $1,780), and which one is correct depends entirely on whether
the question is about typical outcomes or about totals. Burying that in an
implementation detail would decide it for the user.

### Decisions separate credibility from materiality

An estimate is not a decision. `recommend` applies two gates — is the effect
distinguishable from nothing, and is it large enough to be worth the cost — and
reports the thresholds alongside the verdict, so disagreeing means disagreeing
with a stated number. A result that is credible but immaterial gets its own
verdict, because "it works and is not worth doing" calls for a different
follow-up than "it does not work".

## Module layout

| Module | Responsibility |
| --- | --- |
| `cip.data` | Offline datasets, the ground-truth simulator, the DuckDB warehouse |
| `cip.design` | Sample size, power, minimum detectable effect, randomisation checks |
| `cip.frequentist` | Two-arm tests, bootstrap, multiple-testing control |
| `cip.bayesian` | Conjugate and MCMC posteriors, probability of benefit, ROPE |
| `cip.causal` | Propensity scores, overlap guards, IPW, matching, DiD, DML, causal forest |
| `cip.uplift` | T/S/DR learners, Qini and AUUC, decile diagnostics |
| `cip.evaluation` | Bias, interval coverage, sensitivity to hidden confounding |
| `cip.decision` | Ship/hold verdicts and cost-aware targeting policies |

## Scaling

Stated as a design position, not a measured claim — nothing here has run beyond
the sizes in `docs/results.md`:

- **The estimators are single-machine.** The largest sample measured is 16,177
  rows. EconML's cross-fitting parallelises across folds; beyond a few million
  rows the nuisance models would need subsampling, which changes the variance.
- **The warehouse is append-only** and would move to PostgreSQL before several
  analysts shared it.
- **MCMC is the slow path.** The conjugate route is exact and instant for binary
  outcomes and should be preferred wherever it applies; MCMC is reserved for
  continuous outcomes where no conjugate form exists.
- **Uplift scales with the outcome model**, not with the uplift machinery. The
  Qini computation is a sort and two cumulative sums.

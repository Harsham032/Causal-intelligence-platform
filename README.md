# causal-intelligence-platform

An experimentation and causal inference platform: experiment design, frequentist
and Bayesian analysis, treatment-effect estimation from observational data,
uplift modelling, and a decision layer that turns an estimate into a
recommendation.

Every number in this repository was measured by running the code in it — including
the places where the code is wrong, which are reported rather than removed.

---

## Why this is harder than it looks

The fundamental problem of causal inference is that you never observe both
outcomes for the same unit. Everything downstream follows from that:

- **There is no label to validate against.** A classifier can be scored on held-out
  data. A treatment-effect estimate cannot, because the counterfactual was never
  observed. Most causal write-ups therefore report a number nobody can check.
- **The assumptions do the work, not the estimator.** A sophisticated method on
  data with no overlap produces a confident, precise, wrong answer. Measured
  here: restricting to common support removes **76% of the bias** before any
  estimator runs.
- **Sophistication can hurt.** On the benchmark below, double machine learning
  and causal forests — designed for exactly this problem — finish behind
  1-nearest-neighbour matching, because after enforcing overlap there are 167
  treated units left.
- **A good point estimate can carry a worthless interval.** This repository's
  matching estimator has the *lowest bias* of six methods and its 95% interval
  covers the truth **10%** of the time. Both facts are measured, and §4 of
  `docs/results.md` says so.

---

## The benchmark

The National Supported Work demonstration randomised job training, so its
difference in means is an unbiased treatment effect: **$1,794**. Swapping its
controls for survey respondents fabricates an observational study where the
truth is already known — the same people received the same programme — and every
estimator can be scored against it.

| Method | Estimate | Bias | Covers truth |
| --- | ---: | ---: | :---: |
| naive difference in means | −8,498 | **−10,292** | no |
| naive, common support | −712 | −2,506 | no |
| IPW (ATT) | 1,058 | −736 | yes |
| **1-NN propensity matching** | **1,644** | **−150** | yes |
| double machine learning | 16 | −1,779 | yes |
| causal forest | 361 | −1,434 | yes |

Full numbers, interval coverage, and the limitations that bound them are in
[`docs/results.md`](docs/results.md).

---

## What is in the box

| Layer | Module | What it does |
| --- | --- | --- |
| Data | `cip.data` | Six offline datasets, a simulator with known per-unit effects, a DuckDB warehouse |
| Design | `cip.design` | Sample size, power, minimum detectable effect, balance and assignment checks |
| Frequentist | `cip.frequentist` | Welch and Newcombe intervals, bootstrap, Bonferroni and Benjamini-Hochberg |
| Bayesian | `cip.bayesian` | Conjugate and MCMC posteriors, P(benefit), credible intervals, ROPE |
| Causal | `cip.causal` | Propensity scores, overlap guards, IPW, matching, difference-in-differences, DML, causal forest |
| Uplift | `cip.uplift` | T/S/DR learners with cross-fitting, Qini, AUUC, decile diagnostics |
| Evaluation | `cip.evaluation` | Bias, interval coverage, Rosenbaum bounds, confounding strength |
| Decision | `cip.decision` | Ship/hold verdicts against stated thresholds, cost-aware targeting |
| Dashboard | `app/dashboard.py` | Reads pipeline results; recomputes nothing |

---

## Quickstart

Requires Python 3.11 or newer. No network access, no credentials, no downloads.

```bash
git clone https://github.com/Harsham032/causal-intelligence-platform.git
cd causal-intelligence-platform
make install
```

```bash
make experiment    # analyse a randomised trial end to end
make benchmark     # score every estimator against a known effect
make uplift        # fit uplift models and score them against the true ranking
make dashboard     # serve the dashboard on :8501
```

Quality gates:

```bash
make check         # ruff + black --check + mypy + pytest
make coverage      # tests with a coverage report
make secrets-scan  # look for credential-shaped strings in tracked files
```

`make help` lists every target.

---

## The code refuses to answer some questions

The failure mode this platform is built against is not a wrong number. It is a
plausible number nobody questions.

```python
from cip.causal import estimate_propensity, check_overlap

propensity = estimate_propensity(observational, "treat", covariates)
check_overlap(propensity.scores, observational["treat"].to_numpy())
# AssumptionError: enforcing overlap would discard 89.5% of the sample
# (14473 of 16177). The remaining units are not the population the question
# was about; report the overlap failure rather than an effect estimated on
# what is left.
```

Proceeding is possible and deliberately separate:

```python
from cip.causal import trim_to_overlap

mask, report = trim_to_overlap(propensity.scores, treatment)
# now the estimand is "the effect among units that could plausibly have
# received either arm" — and the caller has to know that
```

Two other guards behave the same way: difference-in-differences can refuse when
the groups were already diverging before treatment, and double machine learning
rejects a single fold, which would be DML without the cross-fitting that makes
it work.

---

## Uplift modelling

Scored against effects the models never saw, on 6,000 simulated units:

| Learner | Qini | % of oracle | Spearman vs truth |
| --- | ---: | ---: | ---: |
| **s-learner** | 0.0652 | **93.8%** | 0.9334 |
| dr-learner | 0.0646 | 93.0% | 0.8965 |
| t-learner | 0.0606 | 87.3% | 0.8200 |
| random | 0.0035 | 5.1% | ≈0 |
| oracle | 0.0695 | 100% | 1.000 |

The oracle row is what makes the others readable: 0.065 is not a weak score, it
is 94% of the ceiling this data allows.

Targeting the profitable 74.9% of units returns **4,376** against **3,778** for
treating everyone — a 15.8% improvement from deciding *who* rather than
*whether*.

---

## Configuration

Pipeline settings live in YAML and are overridable per run:

```bash
python scripts/run_benchmark.py --config configs/default.yaml --runs 100
python scripts/run_experiment.py --dataset nsw --set design.alpha=0.01
```

Deployment settings come from `CIP_*` environment variables:

```bash
export CIP_WAREHOUSE_PATH=data/processed/cip.duckdb
```

`CIP_POSTGRES_URL` takes a SQLAlchemy URL for a shared warehouse. Keep it in a
secret store, not in shell history; the full list of variables is in
[`.env.example`](.env.example).

---

## Repository layout

```
app/           Streamlit dashboard
configs/       run configurations (full and reduced)
data/          datasets and the warehouse (gitignored; see data/README.md)
docs/          architecture, methodology, data sources, results
scripts/       experiment, benchmark, uplift, Criteo downloader, secrets scan
src/cip/       the package
tests/         123 tests, 87% branch coverage
```

---

## Limitations

- **The Criteo Uplift dataset was not used.** It is the specified primary uplift
  source and this environment has no network route to it. A downloader ships;
  no published number was measured on it.
- **Uplift results are simulated.** Individual treatment effects are
  unobservable in principle, so this is not avoidable — but it means the results
  measure recovery of injected structure, not discovery of real structure.
- **Matching intervals in this repository are invalid** — coverage 0.10 against
  a nominal 0.95. The point estimate is sound; the fix is the Abadie-Imbens
  variance estimator, which is not implemented.
- **LaLonde estimates are for units on common support**, 10.5% of the sample —
  not for the original population.
- **Unconfoundedness is assumed and cannot be tested.** Sensitivity analysis
  bounds how strong a hidden confounder would need to be; it cannot rule one out.
- **The dashboard was not exercised against a browser.** It parses and lints and
  reads real report files; no interaction test was run.
- **Staggered difference-in-differences is not handled correctly.** Two-way fixed
  effects is biased when treatment timing varies, and Callaway-Sant'Anna is not
  implemented.

---

## Further reading

- [`docs/results.md`](docs/results.md) — every measurement, and what is wrong
- [`docs/methodology.md`](docs/methodology.md) — estimands, identification,
  metric choices, threats to validity
- [`docs/architecture.md`](docs/architecture.md) — components and design decisions
- [`docs/data-sources.md`](docs/data-sources.md) — what the data is and why

---

## License

[MIT](LICENSE)

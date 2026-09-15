# Data sources

Every dataset used by the published results ships inside the
[`causaldata`](https://pypi.org/project/causaldata/) package. `make install` is
the whole acquisition step, no credentials are needed, and a full run makes no
network request. An analysis that cannot be reproduced from a clean checkout is
not reproducible.

## What is used

| Dataset | Rows | Origin | Role |
| --- | ---: | --- | --- |
| `nsw` | 445 | National Supported Work demonstration, Dehejia-Wahba sample | Randomised job-training experiment. Supplies the experimental benchmark. |
| `cps` | 15,992 | Current Population Survey | Never-treated comparison pool. Pairs with the NSW treated units. |
| `nhefs` | 1,566 | NHEFS smoking-cessation follow-up | The textbook inverse-probability-weighting example. |
| `thornton` | 4,820 | Randomised HIV-testing incentives, Malawi | Large randomised trial with a binary outcome. |
| `organ_donations` | 162 | US state organ-donation rates by quarter | Difference-in-differences: California's 2011 active-choice switch. |
| `castle` | 550 | Castle-doctrine law adoption by state and year | Staggered difference-in-differences panel. |

Citations for the underlying studies are in the `causaldata` package
documentation; the package redistributes them, this repository does not.

## The LaLonde pairing

The `nsw` and `cps` tables are used together, and the construction is the
methodological centre of this repository.

LaLonde's question was whether observational methods can recover an answer that
a randomised experiment already knows. The construction: take the *treated*
units from the experiment, discard the experiment's own controls, and substitute
survey respondents who never received the programme. The same people received
the same training, so the true treatment effect is unchanged — but the
comparison group is now incomparable in exactly the way real observational data
is incomparable.

Any estimator applied to that sample can be scored against the number the
experiment measured. Measured here, the experiment gives **$1,794** and the
naive observational comparison gives **−$8,498**: a bias of $10,292 for methods
to close. `docs/results.md` §3 reports how much each one closes.

Dehejia and Wahba's contribution was showing that the answer depends critically
on restricting to common support, which this repository measures directly: the
restriction alone removes 76% of the bias.

## Simulated data

`cip.data.simulator` generates experiments in which both potential outcomes are
written down. This is not a substitute for real data and it is not used where
real data would serve. It is used for the two things real data cannot support:

**Scoring an uplift model.** Uplift ranks units by their individual treatment
effect. A unit is only ever observed under one arm, so that effect is
unobservable in principle — not merely unmeasured. Without a simulation there is
no ground-truth ranking to score against.

**Measuring interval coverage.** A 95% interval's claim is about repeated
sampling. Checking it needs many studies with the same known truth, which one
real dataset cannot provide.

What a simulation can tell you is whether an estimator recovers structure that
was injected. What it cannot tell you is whether that structure resembles
anything real. §11 of `docs/results.md` states the bound explicitly.

## The Criteo Uplift Prediction Dataset

The project specification names the Criteo Uplift Prediction Dataset (25 million
rows, a real randomised advertising experiment) as the primary uplift source. It
is **not used by any published result here.**

The environment in which these measurements were produced has no network route
to Criteo's hosts — all three known endpoints return HTTP 403 at the proxy:

```
http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz          403
https://ailab.criteo.com/criteo-uplift-prediction-dataset/       403 (CONNECT)
https://criteo-bucket.s3.eu-central-1.amazonaws.com/...          403
```

A downloader ships at `scripts/download_criteo.py` and the loader is covered by
tests against a fixture, so the path is exercised. But no number in this
repository was measured on that data, and none claims to be.

On a machine that can reach it, `python scripts/download_criteo.py --output
data/external` fetches and verifies the archive; `docs/results.md` §12 lists the
Criteo run as the second most valuable follow-up.

## Handling rules

1. Downloaded data lands in `data/external/`, which is git-ignored.
2. The DuckDB warehouse lands in `data/processed/`, also ignored.
3. CI fails if any tracked file exceeds 2MB, or if a `.parquet`, `.duckdb` or
   `.csv.gz` is ever added to the index.
4. Credentials are read from the environment; `.env.example` carries
   placeholders only and `.env` is never committed.

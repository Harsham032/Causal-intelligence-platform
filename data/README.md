# Data directory

## Layout

```
data/
├── raw/          ignored - downloaded archives, if any
├── interim/      ignored - cleaned intermediates
├── processed/    ignored - prepared tables and the DuckDB warehouse
├── external/     ignored - datasets fetched from third parties
└── README.md     this file
```

**Nothing here is committed.** Every dataset the published results use ships
inside the `causaldata` package, so `make install` is the whole acquisition
step and a full run needs no network access.

## What the analyses use

| Dataset | Rows | Role |
| --- | ---: | --- |
| `nsw` | 445 | Randomised job-training experiment. The experimental benchmark. |
| `cps` | 15,992 | Survey comparison group. Pairs with the NSW treated units to fabricate an observational study whose true effect is already known. |
| `nhefs` | 1,566 | Smoking cessation and weight change. The textbook inverse-probability-weighting example. |
| `thornton` | 4,820 | Randomised HIV-testing incentives. A large trial with a binary outcome. |
| `organ_donations` | 162 | State donation rates by quarter. A difference-in-differences design. |
| `castle` | 550 | Castle-doctrine adoption by state and year. A staggered difference-in-differences panel. |

Plus a seeded simulator (`cip.data.simulator`) that writes down both potential
outcomes, so every unit's individual treatment effect is known exactly. That is
the only way to score an uplift model, which ranks units by an effect no real
dataset ever reveals.

## The Criteo uplift dataset

`docs/data-sources.md` specifies the Criteo Uplift Prediction Dataset as a
large-scale alternative for uplift modelling. A downloader is provided:

```bash
python scripts/download_criteo.py --output data/external
```

**It has not been run for the published results.** The environment used to
produce `docs/results.md` has no route to Criteo's hosts, so nothing in this
repository reports a number measured on it. The downloader verifies the file
after fetching and the loader is covered by tests against a fixture, but the
dataset itself is untested here and is marked as such wherever it appears.

## Handling rules

1. Downloaded data lands in `data/external/`, which is git-ignored.
2. The DuckDB warehouse lands in `data/processed/`, also ignored.
3. CI fails if any tracked file exceeds 2MB, or if a `.parquet`, `.duckdb` or
   `.csv.gz` is ever added to the index.
4. Credentials are read from the environment; `.env.example` carries
   placeholders only and `.env` is never committed.

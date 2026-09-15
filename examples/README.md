# Examples

## `end_to_end.py`

Runs the whole platform in about a minute: design sizing, the experimental
benchmark, four observational estimators against a known truth, sensitivity
analysis, multiple-testing control, a Bayesian posterior, uplift with an oracle
ceiling, and a decision. No network, no credentials, no external service.

```bash
python examples/end_to_end.py
```

Worth watching for:

- **The overlap check refuses.** It raises on the LaLonde data rather than
  returning a number, and proceeding takes a separate explicit call.
- **Trimming beats modelling.** Restricting to common support removes most of
  the $10,292 bias before any estimator runs.
- **Matching flags its own interval.** The best point estimate of the four, with
  a note saying the interval is not trustworthy and why.
- **Multiple testing, live.** Twenty metrics with no real effect produce several
  "significant" results uncorrected, and none after Benjamini-Hochberg.
- **The oracle row.** A Qini of 0.078 means nothing until you see the 0.084
  ceiling next to it.
- **The verdict is inconclusive.** The IPW interval contains zero, so the
  example ends by declining to claim an effect — which is the honest reading.

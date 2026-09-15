# Results

Every number here was produced by running the code in this repository. Nothing
is estimated, projected, or carried over from a paper. Where a measurement is
missing, or where this repository's own code is wrong, it says so.

Reproduce with:

```bash
make install
make experiment   # analyse a randomised trial end to end
make benchmark    # score every estimator against a known effect
make uplift       # fit uplift models and score them against the true ranking
```

---

## 1. Environment

| | |
| --- | --- |
| Platform | Linux x86_64 |
| CPU | Intel Xeon @ 2.80 GHz, 4 cores |
| Python | 3.11.15 |
| NumPy / pandas / SciPy | 2.4.6 / 3.0.5 / 1.17.1 |
| scikit-learn / statsmodels | 1.9.1 / 0.15.0 |
| EconML / PyMC / DuckDB | 0.17.0 / 5.28.5 / 1.5.5 |
| Measured | 2026-09-15 |

No network access is used at any point. Every dataset ships inside the
`causaldata` package, so `make install` is the entire acquisition step.

---

## 2. The benchmark, and why it is the point

Most causal inference write-ups cannot tell you whether their estimator is
right, because the true effect is unknown. Two constructions here can.

**The LaLonde pairing.** The National Supported Work demonstration randomised
job training, so the difference in means between its arms is an unbiased average
treatment effect: **$1,794** (95% CI $474 to $3,115) on 185 treated and 260
control units. Replacing the experiment's own controls with 15,992 Current
Population Survey respondents fabricates an observational study in which the
same people received the same programme — so the true effect has not moved,
while the comparison group is now wrong in exactly the way real observational
data is wrong. An estimator's bias on it is measurable rather than arguable.

**The simulator.** Both potential outcomes are written down, so every unit's
individual effect is known. That is the only way to score an uplift model, which
ranks units by an effect no real dataset ever reveals, and the only way to
measure interval coverage, which needs repetition.

---

## 3. Estimating a treatment effect from observational data

Measured against the experimental answer of $1,794:

| Method | Estimate | Bias | 95% interval | Covers truth |
| --- | ---: | ---: | :---: | :---: |
| naive difference in means | −8,498 | **−10,292** | [−9,648, −7,347] | **no** |
| naive, common support | −712 | −2,506 | [−1,944, 520] | **no** |
| IPW (ATT) | 1,058 | −736 | [−266, 2,383] | yes |
| **1-NN propensity matching** | **1,644** | **−150** | [193, 3,095] | yes |
| double machine learning | 16 | −1,779 | [−2,090, 2,122] | yes |
| causal forest | 361 | −1,434 | [−2,457, 3,179] | yes |

### The overlap check is the largest single lever

The propensity model reaches **AUC 0.971** on this sample. That is not a
modelling success — it means the arms are nearly separable, so most units have
no counterpart in the other arm. Enforcing common support drops **89.5%** of the
rows (14,473 of 16,177), leaving 167 treated and 1,537 control units.

Restricting to that region, with no model at all, moves the bias from −10,292 to
−2,506. **Roughly 76% of the total bias is removed by a data check rather than
by an estimator.** Everything the four adjusted methods do afterwards operates
on the remaining quarter.

`check_overlap` raises on this by default rather than returning a number.
Proceeding requires calling `trim_to_overlap` explicitly, because trimming
silently changes the estimand: after it, the answer is the effect among units
that could plausibly have received either arm — here, the difference between a
question about disadvantaged trainees and a question about the US labour force.

### Sophistication is not free

The two most flexible methods finish last. Double machine learning and causal
forests are designed for exactly this problem, and with 167 treated units left
after trimming they have too little data to beat 1-nearest-neighbour matching,
which lands within **$150** of the experimental answer.

All four adjusted methods produce intervals containing the truth; both naive
comparisons do not. That, rather than the ranking, is the result that
generalises.

---

## 4. Interval coverage

A point estimate says where the answer probably is. An interval claims how often
it is right, and that claim is checkable only by repetition. Over **60 simulated
observational studies** with a true effect of 1.0:

| Method | Bias | RMSE | Coverage | Nominal |
| --- | ---: | ---: | ---: | ---: |
| naive | 1.7083 | 1.7544 | 0.000 | 0.95 |
| IPW | 0.0117 | 0.0799 | 1.000 | 0.95 |
| matching | 0.1904 | 0.2097 | **0.100** | 0.95 |

### This repository's matching intervals are wrong

The point estimate is good — bias 0.19 against a true effect of 1.0, and on the
LaLonde benchmark it was the most accurate method of the six. Its nominal 95%
interval contains the truth **10%** of the time.

The cause is in `cip.causal.weighting.matched_estimate`: the standard error
treats the matched differences as independent draws. They are not. A control
unit can be matched to several treated units, and the propensity score they were
matched on was itself estimated from the same data. Abadie and Imbens showed
that the bootstrap does not fix this either, so there is no cheap correction to
apply; their variance estimator is the real answer and is **not implemented
here**.

The result object carries `interval_is_trustworthy=False` and names what to use
instead. The honest summary is: use matching's point estimate, take the interval
from IPW.

IPW's coverage of 1.000 is the opposite failure and a much milder one. Its
intervals are wider than they need to be, which costs power but does not mislead
anybody.

Note what this means for §3: the *intervals* in that table come from estimators
whose coverage differs, and the matching interval there should be read as
decorative.

---

## 5. Analysing a randomised experiment

`make experiment` walks a trial through design checks, frequentist estimation,
a posterior, and a recommendation — in that order, because each step can
invalidate the ones after it.

On the Thornton HIV-testing trial (2,211 treated, 623 control, binary outcome):

| | |
| --- | --- |
| Effect | +0.4506 (+133.0%) |
| 95% CI | [0.4087, 0.4902] |
| Method | Newcombe hybrid score |
| P(treatment > control) | 1.0000 |
| Posterior effect | +0.4498, 95% CrI [0.4085, 0.4901] |
| Verdict | **ship** |

The Bayesian and frequentist answers agree to three decimals, which is what a
flat prior on 2,834 observations should produce. Where they disagree is more
interesting — see §7.

### Balance checks find real problems in real experiments

Run against the NSW experiment, the balance check does not pass:

| Covariate | Treated | Control | SMD | Balanced |
| --- | ---: | ---: | ---: | :---: |
| age | 25.82 | 25.05 | +0.107 | no |
| educ | 10.35 | 10.09 | +0.141 | no |
| black | 0.843 | 0.827 | +0.044 | yes |
| hisp | 0.060 | 0.108 | −0.175 | no |
| marr | 0.189 | 0.154 | +0.094 | yes |
| nodegree | 0.708 | 0.835 | **−0.304** | no |
| re74 | 2,095.6 | 2,107.0 | −0.002 | yes |
| re75 | 1,532.1 | 1,266.9 | +0.084 | yes |

This is a genuinely randomised experiment. The Dehejia-Wahba sample is selected
*after* randomisation on the availability of 1974 earnings, and that selection
shows. The two earnings variables — the ones the outcome is measured in — are
the best balanced of the eight, which is why the experimental estimate is still
used as the benchmark.

The statistic is the standardised mean difference rather than a t-test p-value
deliberately. With 16,177 rows a trivial imbalance is "significant"; with 445 a
serious one is not. The SMD does not move with sample size and its 0.10 action
threshold means the same thing at any scale.

---

## 6. Multiple testing

An experiment reporting twenty metrics at α = 0.05 will produce a false winner
most of the time. Measured over **4,000 trials of 20 metrics with no true
effect anywhere**:

| Correction | Family-wise error rate |
| --- | ---: |
| none | **0.637** |
| Bonferroni | 0.044 |
| Benjamini-Hochberg | 0.045 |

Uncorrected, roughly two runs in three produce at least one "significant"
result from pure noise. Neither correction substitutes for naming the primary
metric before the experiment starts, which is the only thing that removes the
problem rather than managing it.

---

## 7. The likelihood is a choice about the estimand

On the NSW earnings outcome — many zeros, a long right tail — the Bayesian
model's likelihood changes the answer by a third:

| Model | Effect | 95% interval |
| --- | ---: | :---: |
| difference in means (frequentist) | $1,794 | [$474, $3,115] |
| Bayesian, Normal likelihood | $1,780 | [$416, $3,128] |
| Bayesian, Student-t likelihood | $1,277 | [$150, $2,354] |

Both Bayesian models converged cleanly (R-hat 1.000, ESS > 1,400). Neither is
wrong; they estimate different things. Student-t down-weights extreme values and
recovers a robust location shift. Normal estimates the shift in the **mean**,
which is what a difference in means estimates and what a question about total
earnings requires — the large values are the point, and down-weighting them
answers a different question.

The likelihood is therefore a parameter of `normal_test`, not a default buried
in the implementation.

---

## 8. Uplift modelling

Fitted on 6,000 simulated units where every individual effect is known, true
ATE 1.0 with a CATE standard deviation of 1.0:

| Learner | Qini | % of oracle | Spearman vs truth | RMSE vs truth |
| --- | ---: | ---: | ---: | ---: |
| **s-learner** | 0.0652 | **93.8%** | 0.9334 | 0.4218 |
| dr-learner | 0.0646 | 93.0% | 0.8965 | 0.4489 |
| t-learner | 0.0606 | 87.3% | 0.8200 | 0.6101 |
| random ranking | 0.0035 | 5.1% | ≈0 | — |
| oracle (true effect) | 0.0695 | 100% | 1.000 | 0.000 |

The oracle row is what makes the rest interpretable. A Qini of 0.065 sounds
poor in isolation; against a ceiling of 0.0695 it is 94% of the achievable
ranking quality, from a model that never saw a single individual effect.

The T-learner finishes last, as its structure predicts: two outcome models each
spend their capacity fitting a baseline far larger than the effect, and their
independent errors compound in the difference.

### Targeting is worth more than the average effect

At a treatment cost of 0.3 per unit against an outcome worth 1.0:

| Policy | Expected value |
| --- | ---: |
| treat everyone | 3,777.5 |
| treat the profitable 74.9% | **4,376.1** |

A **15.8% improvement** from deciding *who* rather than *whether*. That gap is
the entire practical argument for modelling uplift instead of the average
effect, and it exists only because the effect is heterogeneous — a model that
found no spread would correctly tell you targeting cannot help.

---

## 9. What the code refuses to do

Three guards fire on real inputs rather than being decorative:

| Guard | Fires when | Measured on |
| --- | --- | --- |
| `check_overlap` | common support would drop more than half the sample | LaLonde: 89.5% dropped, raises |
| `difference_in_differences(refuse_on_pre_trend=True)` | groups were already diverging before treatment | synthetic pre-trend, p < 0.05 |
| `_EntityHistory`-style ordering, balance `not_computable` | a covariate cannot be compared at all | Thornton: `age`, 441 of 4,820 missing |

The last one was a bug found while writing the pipeline script. Missing data
produced a NaN standardised difference, which then read as "imbalanced" because
`abs(nan) <= threshold` is False. Those are different findings — one says the
arms differ, the other says nobody can tell — and they are now reported
separately.

---

## 10. Test suite

123 tests, **87% branch coverage**, all passing on Python 3.11 and 3.12.

The suite is built on properties checkable against a known answer rather than
against previous output: a randomised design recovers its own effect, confounding
biases the naive comparison monotonically, IPW removes most of that bias, an
inverted uplift ranking scores below random, and no model beats the oracle.

One test was itself wrong and is worth recording. The DML test asserted that a
single run's 95% interval contained the truth. A correct 95% interval misses 5%
of the time by construction, so that assertion tests luck rather than the
estimator. It now asserts bias reduction — measured, DML leaves about 9%
residual bias where the naive comparison leaves 213% — and coverage is measured
where it belongs, over repeated runs, in §4.

---

## 11. Limitations

Stated because they bound what the numbers mean:

- **The Criteo Uplift dataset was not used.** It is named as the primary source
  in `docs/data-sources.md` and the environment used for these measurements has
  no network route to it (403 on every endpoint). A downloader ships and is
  tested against a fixture; no number in this repository was measured on it.
- **The uplift results are simulated.** They measure whether the estimators
  recover a structure that was injected, not whether they would find structure
  in a real marketing population. This is not a shortcut — individual treatment
  effects are unobservable in principle — but it bounds the claim.
- **The LaLonde estimates are for units on common support**, which is 10.5% of
  the observational sample. They are not estimates for the original population.
- **Matching intervals in this repository are not valid** (§4). The point
  estimate is sound.
- **Unconfoundedness is assumed and cannot be tested.** The sensitivity module
  measures how strong a hidden confounder would have to be to overturn a result;
  it cannot rule one out.
- **One simulation seed per configuration** for the uplift results. The coverage
  benchmark uses 60 draws; the uplift comparison uses one.
- **The dashboard was not exercised against a browser.** It parses, lints and
  reads the report files the pipeline writes, but no screenshot or interaction
  test was run.

---

## 12. Next experiments

In the order they would be worth running:

1. **Abadie-Imbens variance for matching.** §4 documents an interval this
   repository knows to be wrong. Implementing the correct estimator would close
   the only place where a published number here is untrustworthy.
2. **The Criteo dataset, on a machine that can reach it.** The uplift results
   would move from "recovers injected structure" to "finds real structure at 25
   million rows", which is a different and stronger claim.
3. **Multi-seed uplift comparison.** The learner ranking in §8 rests on one
   simulation. Ten seeds would show whether the S-learner's lead over the
   DR-learner survives, given they differ by 0.0006 in Qini.
4. **Sensitivity analysis on the LaLonde estimates.** §3 reports what four
   estimators found; none of them reports how much hidden confounding it could
   absorb. The module exists and is not yet wired into the benchmark.
5. **Staggered difference-in-differences.** The `castle` dataset has staggered
   adoption, which the two-way fixed-effects estimator implemented here handles
   badly — a known bias when treatment timing varies. Callaway-Sant'Anna is the
   correct estimator and is not implemented.
6. **Power analysis against the observed effects.** Every experiment analysed
   here was already run; computing what they *could* have detected would show
   which of their null results are informative and which are just small.

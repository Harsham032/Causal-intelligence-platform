"""Experiment and uplift dashboard.

Reads results the pipeline has already written rather than recomputing them, so
the page loads in milliseconds and everyone looking at it sees the same numbers.
A dashboard that recomputes on every view is a dashboard whose numbers change
while two people are discussing them.

Run with::

    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cip.config import load_settings  # noqa: E402
from cip.data import Warehouse  # noqa: E402

st.set_page_config(page_title="Causal intelligence platform", layout="wide")

VERDICT_COLOUR = {
    "ship": "#2E7D32",
    "do not ship": "#C62828",
    "credible but immaterial": "#EF6C00",
    "inconclusive": "#546E7A",
}


@st.cache_data(show_spinner=False)
def load_report(name: str) -> dict | None:
    path = ROOT / "reports" / name
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_warehouse_table(table: str) -> pd.DataFrame:
    settings = load_settings()
    path = ROOT / settings.warehouse_path
    if not path.is_file():
        return pd.DataFrame()
    try:
        with Warehouse(str(path)) as warehouse:
            return warehouse.read(table)
    except Exception:
        return pd.DataFrame()


def metric_row(items: list[tuple[str, str, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items, strict=True):
        column.metric(label, value, help=help_text)


def experiment_tab() -> None:
    st.subheader("Experiment analysis")
    reports = sorted((ROOT / "reports").glob("experiment_*.json"))
    if not reports:
        st.info("No experiment reports yet. Run `make experiment` to produce one.")
        return

    choice = st.selectbox("Dataset", [p.stem.replace("experiment_", "") for p in reports])
    report = load_report(f"experiment_{choice}.json")
    if report is None:
        st.warning("That report could not be read.")
        return

    freq = report["frequentist"]
    decision = report["decision"]
    colour = VERDICT_COLOUR.get(decision["verdict"], "#546E7A")
    st.markdown(
        f"<div style='padding:0.75rem 1rem;border-radius:6px;background:{colour};color:white;'>"
        f"<strong>{decision['verdict'].upper()}</strong><br/>{decision['reason']}</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    metric_row(
        [
            ("Control", f"{freq['control_mean']:,.4g}", "Mean outcome in the control arm"),
            ("Treatment", f"{freq['treated_mean']:,.4g}", "Mean outcome in the treated arm"),
            ("Absolute effect", f"{freq['absolute_effect']:,.4g}", None),
            ("Relative effect", f"{freq['relative_effect']:+.2%}", None),
        ]
    )
    st.caption(
        f"95% CI [{freq['ci_lower']:,.4g}, {freq['ci_upper']:,.4g}] · {freq['method']} · "
        f"n = {freq['n_treated']:,} treated / {freq['n_control']:,} control"
    )

    design = report["design"]
    with st.expander("Design checks", expanded=not design["balance_passed"]):
        st.write(
            "**Balance:** "
            + ("passed" if design["balance_passed"] else "failed")
            + f" — {design['balance_reason']}"
        )
        st.write(
            f"**Power** to detect an effect of the observed size: {design['achieved_power']:.3f}. "
            f"Smallest detectable effect at this sample size: {design['minimum_detectable_effect_sd']:.4f} sd."
        )
        if design["imbalanced_covariates"]:
            st.dataframe(pd.DataFrame(design["imbalanced_covariates"]), width="stretch")

    bayes = report.get("bayesian")
    if bayes:
        st.markdown("#### Posterior")
        metric_row(
            [
                ("P(treatment > control)", f"{bayes['probability_of_benefit']:.4f}", None),
                (
                    "P(practically equivalent)",
                    f"{bayes['probability_practically_equivalent']:.4f}",
                    "Probability the effect is inside the region of practical equivalence",
                ),
                (
                    "Expected loss",
                    f"{bayes['expected_loss']:,.4g}",
                    "Average cost if we ship and the treatment is in fact worse",
                ),
                (
                    "95% credible interval",
                    f"[{bayes['ci_lower']:,.3g}, {bayes['ci_upper']:,.3g}]",
                    None,
                ),
            ]
        )
        st.caption(bayes["method"])


def benchmark_tab() -> None:
    st.subheader("Estimator benchmark")
    report = load_report("benchmark_report.json")
    if report is None:
        st.info("No benchmark yet. Run `make benchmark` to produce one.")
        return

    lalonde = report.get("lalonde")
    if lalonde:
        st.markdown("#### Against a randomised experiment")
        st.caption(
            f"Truth: {lalonde['truth']:,.0f} (95% CI {lalonde['truth_interval'][0]:,.0f} to "
            f"{lalonde['truth_interval'][1]:,.0f}) from {lalonde['experimental_n']['treated']} treated "
            f"and {lalonde['experimental_n']['control']} control units."
        )
        frame = pd.DataFrame(lalonde["results"])
        chart_data = frame.assign(abs_bias=frame["bias"].abs())
        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X("estimate:Q", title="Estimated effect"),
                y=alt.Y("method:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.covers_truth, alt.value("#2E7D32"), alt.value("#C62828")
                ),
                tooltip=["method", "estimate", "bias", "covers_truth"],
            )
            .properties(height=240)
        )
        rule = (
            alt.Chart(pd.DataFrame({"truth": [lalonde["truth"]]}))
            .mark_rule(strokeDash=[6, 4], color="#212121")
            .encode(x="truth:Q")
        )
        st.altair_chart(chart + rule, use_container_width=True)
        st.dataframe(
            frame[["method", "estimate", "bias", "ci_lower", "ci_upper", "covers_truth", "notes"]],
            width="stretch",
        )
        overlap = lalonde["overlap"]
        st.warning(
            f"Propensity AUC {lalonde['propensity_auc']:.4f} — the arms are nearly separable. "
            f"{overlap['detail']}. Estimates are for the units on common support, not the "
            "original population."
        )

    simulation = report.get("simulation")
    if simulation:
        st.markdown("#### Interval coverage over repeated studies")
        rows = [
            {
                "method": row["method"],
                "bias": row["bias"],
                "rmse": row["rmse"],
                "coverage": row["coverage"]["empirical"],
                "nominal": row["coverage"]["nominal"],
                "mean_width": row["coverage"]["mean_width"],
            }
            for row in simulation["results"]
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.caption(
            f"{simulation['runs']} simulated studies, true effect {simulation['truth']:.4f}. "
            "Coverage is the share of runs whose interval contained the truth; a 95% interval "
            "covering far less than 0.95 is overconfident."
        )


def uplift_tab() -> None:
    st.subheader("Uplift and targeting")
    report = load_report("uplift_report.json")
    if report is None:
        st.info("No uplift report yet. Run `make uplift` to produce one.")
        return

    metric_row(
        [
            ("Qini (best learner)", f"{report['best']['qini']:.4f}", None),
            (
                "Oracle Qini",
                f"{report['oracle_qini']:.4f}",
                "Ranking by the true effect — the ceiling any model can reach",
            ),
            ("Share of oracle", f"{report['best']['qini'] / report['oracle_qini']:.1%}", None),
            ("Random ranking", f"{report['random_qini']:.4f}", None),
        ]
    )

    learners = pd.DataFrame(report["learners"])
    st.dataframe(learners, width="stretch")

    curve = pd.DataFrame(report["curve"])
    melted = curve.melt("fraction_targeted", var_name="series", value_name="incremental")
    st.altair_chart(
        alt.Chart(melted)
        .mark_line()
        .encode(
            x=alt.X("fraction_targeted:Q", title="Fraction of population targeted"),
            y=alt.Y("incremental:Q", title="Incremental outcomes"),
            color=alt.Color("series:N", title=None),
        )
        .properties(height=320),
        use_container_width=True,
    )

    deciles = pd.DataFrame(report["deciles"])
    st.markdown("#### Observed uplift by predicted-uplift bin")
    st.caption(
        "The diagnostic that matters most: if observed uplift does not fall as predicted "
        "uplift falls, the ranking is not real whatever the Qini coefficient says."
    )
    st.altair_chart(
        alt.Chart(deciles)
        .mark_bar()
        .encode(
            x=alt.X("bin:O", title="Bin (1 = highest predicted uplift)"),
            y=alt.Y("observed_uplift:Q", title="Observed uplift"),
            tooltip=list(deciles.columns),
        )
        .properties(height=280),
        use_container_width=True,
    )


def history_tab() -> None:
    st.subheader("Stored results")
    estimates = load_warehouse_table("estimates")
    if estimates.empty:
        st.info("The warehouse is empty. Run any pipeline command to populate it.")
        return
    st.dataframe(estimates, width="stretch")
    decisions = load_warehouse_table("decisions")
    if not decisions.empty:
        st.markdown("#### Decisions")
        st.dataframe(decisions, width="stretch")


def main() -> None:
    st.title("Causal intelligence platform")
    st.caption(
        "Treatment-effect estimation, uplift modelling and evidence-based decisions. "
        "Every figure is read from a pipeline run; nothing on this page is recomputed."
    )
    experiment, benchmark, uplift, history = st.tabs(
        ["Experiment", "Estimator benchmark", "Uplift", "History"]
    )
    with experiment:
        experiment_tab()
    with benchmark:
        benchmark_tab()
    with uplift:
        uplift_tab()
    with history:
        history_tab()


main()

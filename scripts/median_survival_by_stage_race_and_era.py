
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times

import overall_survival_data
import prostate_cancer_specific_survival_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_LABELS = overall_survival_data.ERA_LABELS
STAGES = ["M1a", "M1b", "M1c"]
ERAS = [2, 3, 4]
ANALYTIC_HORIZON = 24
RACE_STRATA = [
    ("Overall", None),
    ("Non-Hispanic White", "Non-Hispanic White"),
    ("Non-Hispanic Black", "Non-Hispanic Black"),
]


def km_median_ci(times, events):
    if len(times) == 0 or events.sum() == 0:
        return np.nan, np.nan, np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)
    med = kmf.median_survival_time_
    ci = median_survival_times(kmf.confidence_interval_)
    lo, hi = ci.iloc[0, 0], ci.iloc[0, 1]
    return med, lo, hi


def fmt_median(med, lo, hi):
    if not np.isfinite(med):
        return "NR"
    lo_s = f"{lo:.0f}" if np.isfinite(lo) else "NR"
    hi_s = f"{hi:.0f}" if np.isfinite(hi) else "NR"
    return f"{med:.0f} ({lo_s}-{hi_s})"


def km_two_year_and_median(times, events):
    if len(times) == 0 or events.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)
    survival = float(kmf.predict(ANALYTIC_HORIZON))
    ci = kmf.confidence_interval_survival_function_
    ci_at_horizon = ci.loc[ci.index[ci.index <= ANALYTIC_HORIZON][-1]]
    median = kmf.median_survival_time_
    median_ci = median_survival_times(kmf.confidence_interval_)
    median_lo, median_hi = median_ci.iloc[0, 0], median_ci.iloc[0, 1]
    return (
        survival,
        float(ci_at_horizon.iloc[0]),
        float(ci_at_horizon.iloc[1]),
        median,
        median_lo,
        median_hi,
    )


def fmt_summary(summary):
    survival, survival_lo, survival_hi, median, median_lo, median_hi = summary
    if np.isnan(survival):
        return ""
    return (
        f"2-year {100 * survival:.1f}% "
        f"({100 * survival_lo:.1f}-{100 * survival_hi:.1f}%); "
        f"median {fmt_median(median, median_lo, median_hi)} mo"
    )


def fmt_p(p):
    if np.isnan(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def calc_logrank_p(df_sub, time_col, event_col):
    sub = (
        df_sub[df_sub["era"].isin(ERAS)]
        .dropna(subset=[time_col, event_col])
        .copy()
    )
    if len(sub) == 0 or sub["era"].nunique() <= 1:
        return np.nan
    beyond_horizon = sub[time_col] > ANALYTIC_HORIZON
    sub.loc[beyond_horizon, event_col] = 0
    sub.loc[beyond_horizon, time_col] = ANALYTIC_HORIZON
    lr = multivariate_logrank_test(sub[time_col], sub["era"], sub[event_col])
    return lr.p_value


def build_table(df_os, df_pca):
    rows = []
    endpoints = [
        ("OS", df_os, "os_months", "os_event"),
        ("PCSS", df_pca, "pca_months", "pca_event"),
    ]
    for endpoint_label, df, time_col, event_col in endpoints:
        df_eras = df[df["era"].isin(ERAS) & df["m_stage"].isin(STAGES)].copy()
        for stage in STAGES:
            df_stage = df_eras[df_eras["m_stage"] == stage]
            for stratum_label, race_value in RACE_STRATA:
                df_grp = (
                    df_stage
                    if race_value is None
                    else df_stage[df_stage["race"] == race_value]
                )
                row = {
                    "Endpoint": endpoint_label,
                    "Stage": stage,
                    "Stratum": stratum_label,
                }
                for era in ERAS:
                    df_era = df_grp[df_grp["era"] == era]
                    summary = km_two_year_and_median(
                        df_era[time_col],
                        df_era[event_col],
                    )
                    row[ERA_LABELS[era]] = (
                        f"n={len(df_era):,}; {fmt_summary(summary)}"
                    )
                row["24-month log-rank p"] = fmt_p(
                    calc_logrank_p(df_grp, time_col, event_col)
                )
                rows.append(row)
    table = pd.DataFrame(rows)
    stratum_labels = {
        "Overall": "overall",
        "Non-Hispanic White": "NHW",
        "Non-Hispanic Black": "NHB",
    }
    table["Analysis"] = table.apply(
        lambda row: (
            f"{row['Endpoint']}/{row['Stage']}/"
            f"{stratum_labels.get(row['Stratum'], row['Stratum'])}"
        ),
        axis=1,
    )
    columns = (
        ["Analysis"]
        + [ERA_LABELS[e] for e in ERAS]
        + ["24-month log-rank p"]
    )
    return table[columns]


def main():
    df_os = overall_survival_data.load_and_prepare_data()
    df_pca = prostate_cancer_specific_survival_data.load_and_prepare_data()
    table = build_table(df_os, df_pca)
    out_path = OUTPUT_DIR / "18_median_survival_by_stage_race_era.csv"
    table.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

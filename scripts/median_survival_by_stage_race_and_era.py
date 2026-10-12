from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times

import overall_survival_data
import prostate_cancer_specific_survival_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
ERA_LABELS = overall_survival_data.ERA_LABELS
STAGES = ["M1a", "M1b", "M1c"]
ERAS = [2, 3, 4]
ANALYSIS_MONTHS = 24
RACE_STRATA = [
    ("Overall", None),
    ("Non-Hispanic White", "Non-Hispanic White"),
    ("Non-Hispanic Black", "Non-Hispanic Black"),
]


def format_median(median, lower, upper):
    if not np.isfinite(median):
        return "NR"
    lower_text = f"{lower:.0f}" if np.isfinite(lower) else "NR"
    upper_text = f"{upper:.0f}" if np.isfinite(upper) else "NR"
    return f"{median:.0f} ({lower_text}-{upper_text})"


def km_two_year_and_median(times, events):
    if len(times) == 0 or events.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    km_model = KaplanMeierFitter()
    km_model.fit(times, event_observed=events)
    survival = float(km_model.predict(ANALYSIS_MONTHS))
    ci = km_model.confidence_interval_survival_function_
    ci_at_horizon = ci.loc[ci.index[ci.index <= ANALYSIS_MONTHS][-1]]
    median = km_model.median_survival_time_
    median_ci = median_survival_times(km_model.confidence_interval_)
    median_lo, median_hi = median_ci.iloc[0, 0], median_ci.iloc[0, 1]
    return (
        survival,
        float(ci_at_horizon.iloc[0]),
        float(ci_at_horizon.iloc[1]),
        median,
        median_lo,
        median_hi,
    )


def format_survival_summary(summary):
    survival, survival_lo, survival_hi, median, median_lo, median_hi = summary
    if np.isnan(survival):
        return ""
    return (
        f"2-year {100 * survival:.1f}% "
        f"({100 * survival_lo:.1f}-{100 * survival_hi:.1f}%); "
        f"median {format_median(median, median_lo, median_hi)} mo"
    )


def format_p_value(p):
    if np.isnan(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def logrank_p_value(data_subset, time_column, event_column):
    subset = (
        data_subset[data_subset["era"].isin(ERAS)]
        .dropna(subset=[time_column, event_column])
        .copy()
    )
    if len(subset) == 0 or subset["era"].nunique() <= 1:
        return np.nan
    beyond_horizon = subset[time_column] > ANALYSIS_MONTHS
    subset.loc[beyond_horizon, event_column] = 0
    subset.loc[beyond_horizon, time_column] = ANALYSIS_MONTHS
    logrank_result = multivariate_logrank_test(
        subset[time_column], subset["era"], subset[event_column]
    )
    return logrank_result.p_value


def build_table(os_data, pcss_data):
    rows = []
    endpoints = [
        ("OS", os_data, "os_months", "os_event"),
        ("PCSS", pcss_data, "pca_months", "pca_event"),
    ]
    for endpoint_label, data, time_column, event_column in endpoints:
        era_cohort = data[data["era"].isin(ERAS) & data["m_stage"].isin(STAGES)].copy()
        for stage in STAGES:
            stage_data = era_cohort[era_cohort["m_stage"] == stage]
            for stratum_label, race_value in RACE_STRATA:
                group_data = (
                    stage_data
                    if race_value is None
                    else stage_data[stage_data["race"] == race_value]
                )
                row = {
                    "Endpoint": endpoint_label,
                    "Stage": stage,
                    "Stratum": stratum_label,
                }
                for era in ERAS:
                    era_data = group_data[group_data["era"] == era]
                    summary = km_two_year_and_median(
                        era_data[time_column],
                        era_data[event_column],
                    )
                    row[ERA_LABELS[era]] = (
                        f"n={len(era_data):,}; {format_survival_summary(summary)}"
                    )
                row["24-month log-rank p"] = format_p_value(
                    logrank_p_value(group_data, time_column, event_column)
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
    columns = ["Analysis"] + [ERA_LABELS[e] for e in ERAS] + ["24-month log-rank p"]
    return table[columns]


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    os_data = overall_survival_data.load_and_prepare_data()
    pcss_data = prostate_cancer_specific_survival_data.load_and_prepare_data()
    table = build_table(os_data, pcss_data)
    output_path = TABLE_DIR / "18_median_survival_by_stage_race_era.csv"
    table.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

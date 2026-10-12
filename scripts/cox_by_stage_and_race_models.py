import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from overall_survival_data import ERA_LABELS

ERAS_INCLUDED = [3, 4]
RACES = ["Non-Hispanic White", "Non-Hispanic Black"]
STAGES = ["M1a", "M1b", "M1c"]
ANALYSIS_MONTHS = 24


def format_hazard_ratio(hr, lower, upper):
    return f"{hr:.4f} ({lower:.4f}-{upper:.4f})"


def format_p_value(p):
    if np.isnan(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def prepare_cox_data(data_subset, time_column, event_column):
    cox_data = (
        data_subset[
            [time_column, event_column, "era", "age", "prostatectomy", "radiotherapy"]
        ]
        .copy()
        .dropna()
    )
    mask = cox_data[time_column] > ANALYSIS_MONTHS
    cox_data.loc[mask, event_column] = 0
    cox_data.loc[mask, time_column] = ANALYSIS_MONTHS
    cox_data["era4"] = (cox_data["era"] == 4).astype(int)
    columns = [
        time_column,
        event_column,
        "era4",
        "age",
        "prostatectomy",
        "radiotherapy",
    ]
    for term in ["prostatectomy", "radiotherapy"]:
        counts = pd.crosstab(cox_data[term], cox_data[event_column]).reindex(
            index=[0, 1],
            columns=[0, 1],
            fill_value=0,
        )
        if cox_data[term].nunique() < 2 or (counts == 0).any().any():
            columns.remove(term)
    return cox_data[columns]


def fit_unadjusted(cox_data, time_column, event_column):
    cox_model = CoxPHFitter()
    subset = cox_data[[time_column, event_column, "era4"]].dropna()
    cox_model.fit(
        subset, duration_col=time_column, event_col=event_column, show_progress=False
    )
    row = cox_model.summary.loc["era4"]
    return {
        "HR": np.exp(row["coef"]),
        "CI_lo": np.exp(row["coef lower 95%"]),
        "CI_hi": np.exp(row["coef upper 95%"]),
        "p": row["p"],
    }


def fit_adjusted(cox_data, time_column, event_column):
    cox_model = CoxPHFitter()
    cox_model.fit(
        cox_data, duration_col=time_column, event_col=event_column, show_progress=False
    )
    row = cox_model.summary.loc["era4"]
    return {
        "HR": np.exp(row["coef"]),
        "CI_lo": np.exp(row["coef lower 95%"]),
        "CI_hi": np.exp(row["coef upper 95%"]),
        "p": row["p"],
    }


def build_table(os_data, pcss_data):
    rows = []
    configs = [
        ("Overall Survival", os_data, "os_months", "os_event"),
        ("Prostate Cancer-Specific Survival", pcss_data, "pca_months", "pca_event"),
    ]
    for surv_label, data, time_column, event_column in configs:
        for race in RACES:
            for stage in STAGES:
                data_subset = data[
                    (data["race"] == race)
                    & (data["m_stage"] == stage)
                    & data["era"].isin(ERAS_INCLUDED)
                ].copy()
                if len(data_subset) == 0:
                    continue
                n_era3 = (data_subset["era"] == 3).sum()
                n_era4 = (data_subset["era"] == 4).sum()
                cox_data = prepare_cox_data(data_subset, time_column, event_column)
                try:
                    unadjusted = fit_unadjusted(cox_data, time_column, event_column)
                except Exception:
                    unadjusted = {
                        "HR": np.nan,
                        "CI_lo": np.nan,
                        "CI_hi": np.nan,
                        "p": np.nan,
                    }
                try:
                    adjusted = fit_adjusted(cox_data, time_column, event_column)
                except Exception:
                    adjusted = {
                        "HR": np.nan,
                        "CI_lo": np.nan,
                        "CI_hi": np.nan,
                        "p": np.nan,
                    }
                rows.append(
                    {
                        "Survival": surv_label,
                        "Race": race,
                        "Stage": stage,
                        f"n {ERA_LABELS[3]}": f"{n_era3:,}",
                        f"n {ERA_LABELS[4]}": f"{n_era4:,}",
                        "HR (95% CI)": format_hazard_ratio(
                            unadjusted["HR"], unadjusted["CI_lo"], unadjusted["CI_hi"]
                        ),
                        "p-value": format_p_value(unadjusted["p"]),
                        "Adjusted HR (95% CI)": format_hazard_ratio(
                            adjusted["HR"], adjusted["CI_lo"], adjusted["CI_hi"]
                        ),
                        "Adj p-value": format_p_value(adjusted["p"]),
                    }
                )
    columns = [
        "Survival",
        "Race",
        "Stage",
        f"n {ERA_LABELS[3]}",
        f"n {ERA_LABELS[4]}",
        "HR (95% CI)",
        "p-value",
        "Adjusted HR (95% CI)",
        "Adj p-value",
    ]
    return pd.DataFrame(rows, columns=columns)

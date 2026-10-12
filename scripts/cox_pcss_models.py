import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from prostate_cancer_specific_survival_data import ERA_LABELS

ERAS_INCLUDED = [3, 4]
ERA_REF = ERA_LABELS[3]
RACE_REF = "Non-Hispanic White"
M_STAGE_REF = "M1b"
ERA_ORDER = [ERA_LABELS[4]]
RACE_ORDER = [
    "Non-Hispanic Black",
    "Hispanic",
    "Non-Hispanic Asian/PI",
    "Non-Hispanic AI/AN",
    "Non-Hispanic Unknown Race",
]
M_STAGE_ORDER = ["M1a", "M1c", "M1NOS"]
ANALYSIS_MONTHS = 24


def prepare_cox_data(data):
    cox_data = (
        data[
            [
                "pca_months",
                "pca_event",
                "era_label",
                "age",
                "race",
                "m_stage",
                "prostatectomy",
                "radiotherapy",
            ]
        ]
        .copy()
        .dropna()
    )
    mask = cox_data["pca_months"] > ANALYSIS_MONTHS
    cox_data.loc[mask, "pca_event"] = 0
    cox_data.loc[mask, "pca_months"] = ANALYSIS_MONTHS
    cox_data["era_label"] = cox_data["era_label"].astype(str)
    era_dummies = pd.get_dummies(cox_data["era_label"], prefix="era", drop_first=False)
    reference_column = f"era_{ERA_REF}"
    if reference_column in era_dummies.columns:
        era_dummies = era_dummies.drop(columns=[reference_column])
    race_dummies = pd.get_dummies(cox_data["race"], prefix="race", drop_first=False)
    reference_column = f"race_{RACE_REF}"
    if reference_column in race_dummies.columns:
        race_dummies = race_dummies.drop(columns=[reference_column])
    stage_dummies = pd.get_dummies(
        cox_data["m_stage"], prefix="m_stage", drop_first=False
    )
    reference_column = f"m_stage_{M_STAGE_REF}"
    if reference_column in stage_dummies.columns:
        stage_dummies = stage_dummies.drop(columns=[reference_column])
    return pd.concat(
        [
            cox_data[
                ["pca_months", "pca_event", "age", "prostatectomy", "radiotherapy"]
            ],
            era_dummies,
            race_dummies,
            stage_dummies,
        ],
        axis=1,
    )


def fit_univariate(cox_data, covariate):
    cox_model = CoxPHFitter()
    subset = cox_data[["pca_months", "pca_event", covariate]].dropna()
    cox_model.fit(
        subset, duration_col="pca_months", event_col="pca_event", show_progress=False
    )
    row = cox_model.summary.loc[covariate]
    return {
        "HR": np.exp(row["coef"]),
        "CI_lo": np.exp(row["coef lower 95%"]),
        "CI_hi": np.exp(row["coef upper 95%"]),
        "p": row["p"],
    }


def fit_multivariable(cox_data):
    cox_model = CoxPHFitter()
    cox_model.fit(
        cox_data, duration_col="pca_months", event_col="pca_event", show_progress=False
    )
    s = cox_model.summary.copy()
    s["HR"] = np.exp(s["coef"])
    s["CI_lo"] = np.exp(s["coef lower 95%"])
    s["CI_hi"] = np.exp(s["coef upper 95%"])
    return s[["HR", "CI_lo", "CI_hi", "p"]]


def format_hazard_ratio(hr, lower, upper):
    return f"{hr:.4f} ({lower:.4f}-{upper:.4f})"


def format_p_value(p):
    if np.isnan(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def count_pct(data, col, val):
    n = (data[col] == val).sum()
    pct = 100.0 * n / len(data) if len(data) > 0 else 0
    return f"{n:,} ({pct:.2f}%)"


def build_table(data, cox_data, adjusted_results):
    rows = []

    def add_ref(label, total_n, total_pct):
        rows.append(
            {
                "Variable": label,
                "Sample size (%)": f"{total_n:,} ({total_pct:.2f}%)",
                "HR (95% CI)": "1 (Referent)",
                "p-value": "",
                "Adjusted HR (95% CI)": "1 (Referent)",
                "Adj p-value": "",
            }
        )

    def add_row(label, n_pct, unadjusted, adjusted_row):
        rows.append(
            {
                "Variable": label,
                "Sample size (%)": n_pct,
                "HR (95% CI)": format_hazard_ratio(
                    unadjusted["HR"], unadjusted["CI_lo"], unadjusted["CI_hi"]
                ),
                "p-value": format_p_value(unadjusted["p"]),
                "Adjusted HR (95% CI)": format_hazard_ratio(
                    adjusted_row["HR"], adjusted_row["CI_lo"], adjusted_row["CI_hi"]
                ),
                "Adj p-value": format_p_value(adjusted_row["p"]),
            }
        )

    total_n = len(data)

    rows.append(
        {
            "Variable": "Era",
            "Sample size (%)": f"N = {total_n:,}",
            "HR (95% CI)": "",
            "p-value": "",
            "Adjusted HR (95% CI)": "",
            "Adj p-value": "",
        }
    )
    e_ref_n = (data["era"] == 3).sum()
    add_ref(f"  {ERA_LABELS[3]}", e_ref_n, 100.0 * e_ref_n / total_n)
    for e_label in ERA_ORDER:
        col_key = f"era_{e_label}"
        if col_key not in cox_data.columns:
            continue
        add_row(
            f"  {e_label}",
            count_pct(data, "era_label", e_label),
            fit_univariate(cox_data, col_key),
            adjusted_results.loc[col_key],
        )

    unadjusted_age = fit_univariate(cox_data, "age")
    adjusted_age = adjusted_results.loc["age"]
    rows.append(
        {
            "Variable": "Age (continuous, per year)",
            "Sample size (%)": "",
            "HR (95% CI)": format_hazard_ratio(
                unadjusted_age["HR"], unadjusted_age["CI_lo"], unadjusted_age["CI_hi"]
            ),
            "p-value": format_p_value(unadjusted_age["p"]),
            "Adjusted HR (95% CI)": format_hazard_ratio(
                adjusted_age["HR"], adjusted_age["CI_lo"], adjusted_age["CI_hi"]
            ),
            "Adj p-value": format_p_value(adjusted_age["p"]),
        }
    )

    rows.append(
        {
            "Variable": "Race/Ethnicity",
            "Sample size (%)": "",
            "HR (95% CI)": "",
            "p-value": "",
            "Adjusted HR (95% CI)": "",
            "Adj p-value": "",
        }
    )
    r_ref_n = (data["race"] == RACE_REF).sum()
    add_ref(f"  {RACE_REF}", r_ref_n, 100.0 * r_ref_n / total_n)
    for r_label in RACE_ORDER:
        col_key = f"race_{r_label}"
        if col_key not in cox_data.columns:
            continue
        add_row(
            f"  {r_label}",
            count_pct(data, "race", r_label),
            fit_univariate(cox_data, col_key),
            adjusted_results.loc[col_key],
        )

    rows.append(
        {
            "Variable": "Metastatic Classification",
            "Sample size (%)": "",
            "HR (95% CI)": "",
            "p-value": "",
            "Adjusted HR (95% CI)": "",
            "Adj p-value": "",
        }
    )
    stage_ref_n = (data["m_stage"] == M_STAGE_REF).sum()
    add_ref(
        f"  {M_STAGE_REF}",
        stage_ref_n,
        100.0 * stage_ref_n / total_n,
    )
    for stage_label in M_STAGE_ORDER:
        col_key = f"m_stage_{stage_label}"
        if col_key not in cox_data.columns:
            continue
        add_row(
            f"  {stage_label}",
            count_pct(data, "m_stage", stage_label),
            fit_univariate(cox_data, col_key),
            adjusted_results.loc[col_key],
        )

    rows.append(
        {
            "Variable": "Radical Prostatectomy",
            "Sample size (%)": "",
            "HR (95% CI)": "",
            "p-value": "",
            "Adjusted HR (95% CI)": "",
            "Adj p-value": "",
        }
    )
    rp_no_n = (data["prostatectomy"] == 0).sum()
    add_ref("  No", rp_no_n, 100.0 * rp_no_n / total_n)
    add_row(
        "  Yes",
        count_pct(data, "prostatectomy", 1),
        fit_univariate(cox_data, "prostatectomy"),
        adjusted_results.loc["prostatectomy"],
    )

    rows.append(
        {
            "Variable": "Radiotherapy",
            "Sample size (%)": "",
            "HR (95% CI)": "",
            "p-value": "",
            "Adjusted HR (95% CI)": "",
            "Adj p-value": "",
        }
    )
    rt_no_n = (data["radiotherapy"] == 0).sum()
    add_ref("  No/unknown", rt_no_n, 100.0 * rt_no_n / total_n)
    add_row(
        "  Yes",
        count_pct(data, "radiotherapy", 1),
        fit_univariate(cox_data, "radiotherapy"),
        adjusted_results.loc["radiotherapy"],
    )

    return pd.DataFrame(
        rows,
        columns=[
            "Variable",
            "Sample size (%)",
            "HR (95% CI)",
            "p-value",
            "Adjusted HR (95% CI)",
            "Adj p-value",
        ],
    )

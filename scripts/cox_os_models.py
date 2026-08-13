
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from overall_survival_data import ERA_LABELS

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


CENSOR_TIME = 24


def prepare_cox_df(df):
    cox = (
        df[
            [
                "os_months",
                "os_event",
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
    mask = cox["os_months"] > CENSOR_TIME
    cox.loc[mask, "os_event"] = 0
    cox.loc[mask, "os_months"] = CENSOR_TIME
    cox["era_label"] = cox["era_label"].astype(str)
    era_dummies = pd.get_dummies(cox["era_label"], prefix="era", drop_first=False)
    ref_col = f"era_{ERA_REF}"
    if ref_col in era_dummies.columns:
        era_dummies = era_dummies.drop(columns=[ref_col])
    race_dummies = pd.get_dummies(cox["race"], prefix="race", drop_first=False)
    ref_col = f"race_{RACE_REF}"
    if ref_col in race_dummies.columns:
        race_dummies = race_dummies.drop(columns=[ref_col])
    stage_dummies = pd.get_dummies(
        cox["m_stage"], prefix="m_stage", drop_first=False
    )
    ref_col = f"m_stage_{M_STAGE_REF}"
    if ref_col in stage_dummies.columns:
        stage_dummies = stage_dummies.drop(columns=[ref_col])
    return pd.concat(
        [
            cox[["os_months", "os_event", "age", "prostatectomy", "radiotherapy"]],
            era_dummies,
            race_dummies,
            stage_dummies,
        ],
        axis=1,
    )


def fit_univariate(df_cox, covariate):
    cph = CoxPHFitter()
    sub = df_cox[["os_months", "os_event", covariate]].dropna()
    cph.fit(sub, duration_col="os_months", event_col="os_event", show_progress=False)
    row = cph.summary.loc[covariate]
    return {
        "HR": np.exp(row["coef"]),
        "CI_lo": np.exp(row["coef lower 95%"]),
        "CI_hi": np.exp(row["coef upper 95%"]),
        "p": row["p"],
    }


def fit_multivariable(df_cox):
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col="os_months", event_col="os_event", show_progress=False)
    s = cph.summary.copy()
    s["HR"] = np.exp(s["coef"])
    s["CI_lo"] = np.exp(s["coef lower 95%"])
    s["CI_hi"] = np.exp(s["coef upper 95%"])
    return s[["HR", "CI_lo", "CI_hi", "p"]]


def fmt_hr(hr, lo, hi):
    return f"{hr:.4f} ({lo:.4f}-{hi:.4f})"


def fmt_p(p):
    if np.isnan(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def count_pct(df, col, val):
    n = (df[col] == val).sum()
    pct = 100.0 * n / len(df) if len(df) > 0 else 0
    return f"{n:,} ({pct:.2f}%)"


def build_table(df, df_cox, mv_results):
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

    def add_row(label, n_pct, uv, mv_row):
        rows.append(
            {
                "Variable": label,
                "Sample size (%)": n_pct,
                "HR (95% CI)": fmt_hr(uv["HR"], uv["CI_lo"], uv["CI_hi"]),
                "p-value": fmt_p(uv["p"]),
                "Adjusted HR (95% CI)": fmt_hr(
                    mv_row["HR"], mv_row["CI_lo"], mv_row["CI_hi"]
                ),
                "Adj p-value": fmt_p(mv_row["p"]),
            }
        )

    total_n = len(df)

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
    e_ref_n = (df["era"] == 3).sum()
    add_ref(f"  {ERA_LABELS[3]}", e_ref_n, 100.0 * e_ref_n / total_n)
    for e_label in ERA_ORDER:
        col_key = f"era_{e_label}"
        if col_key not in df_cox.columns:
            continue
        add_row(
            f"  {e_label}",
            count_pct(df, "era_label", e_label),
            fit_univariate(df_cox, col_key),
            mv_results.loc[col_key],
        )

    uv_age = fit_univariate(df_cox, "age")
    mv_age = mv_results.loc["age"]
    rows.append(
        {
            "Variable": "Age (continuous, per year)",
            "Sample size (%)": "",
            "HR (95% CI)": fmt_hr(uv_age["HR"], uv_age["CI_lo"], uv_age["CI_hi"]),
            "p-value": fmt_p(uv_age["p"]),
            "Adjusted HR (95% CI)": fmt_hr(
                mv_age["HR"], mv_age["CI_lo"], mv_age["CI_hi"]
            ),
            "Adj p-value": fmt_p(mv_age["p"]),
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
    r_ref_n = (df["race"] == RACE_REF).sum()
    add_ref(f"  {RACE_REF}", r_ref_n, 100.0 * r_ref_n / total_n)
    for r_label in RACE_ORDER:
        col_key = f"race_{r_label}"
        if col_key not in df_cox.columns:
            continue
        add_row(
            f"  {r_label}",
            count_pct(df, "race", r_label),
            fit_univariate(df_cox, col_key),
            mv_results.loc[col_key],
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
    stage_ref_n = (df["m_stage"] == M_STAGE_REF).sum()
    add_ref(
        f"  {M_STAGE_REF}",
        stage_ref_n,
        100.0 * stage_ref_n / total_n,
    )
    for stage_label in M_STAGE_ORDER:
        col_key = f"m_stage_{stage_label}"
        if col_key not in df_cox.columns:
            continue
        add_row(
            f"  {stage_label}",
            count_pct(df, "m_stage", stage_label),
            fit_univariate(df_cox, col_key),
            mv_results.loc[col_key],
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
    rp_no_n = (df["prostatectomy"] == 0).sum()
    add_ref("  No", rp_no_n, 100.0 * rp_no_n / total_n)
    add_row(
        "  Yes",
        count_pct(df, "prostatectomy", 1),
        fit_univariate(df_cox, "prostatectomy"),
        mv_results.loc["prostatectomy"],
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
    rt_no_n = (df["radiotherapy"] == 0).sum()
    add_ref("  No", rt_no_n, 100.0 * rt_no_n / total_n)
    add_row(
        "  Yes",
        count_pct(df, "radiotherapy", 1),
        fit_univariate(df_cox, "radiotherapy"),
        mv_results.loc["radiotherapy"],
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

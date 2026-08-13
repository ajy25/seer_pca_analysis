
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from overall_survival_data import ERA_LABELS

ERAS_INCLUDED = [3, 4]
RACES = ["Non-Hispanic White", "Non-Hispanic Black"]
STAGES = ["M1a", "M1b", "M1c"]


def fmt_hr(hr, lo, hi):
    return f"{hr:.4f} ({lo:.4f}-{hi:.4f})"


def fmt_p(p):
    if np.isnan(p):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


CENSOR_TIME = 24


def prepare_cox_df(df_sub, time_col, event_col):
    cox = (
        df_sub[[time_col, event_col, "era", "age", "prostatectomy", "radiotherapy"]]
        .copy()
        .dropna()
    )
    mask = cox[time_col] > CENSOR_TIME
    cox.loc[mask, event_col] = 0
    cox.loc[mask, time_col] = CENSOR_TIME
    cox["era4"] = (cox["era"] == 4).astype(int)
    columns = [
        time_col,
        event_col,
        "era4",
        "age",
        "prostatectomy",
        "radiotherapy",
    ]
    for term in ["prostatectomy", "radiotherapy"]:
        counts = pd.crosstab(cox[term], cox[event_col]).reindex(
            index=[0, 1],
            columns=[0, 1],
            fill_value=0,
        )
        if cox[term].nunique() < 2 or (counts == 0).any().any():
            columns.remove(term)
    return cox[columns]


def fit_unadjusted(df_cox, time_col, event_col):
    cph = CoxPHFitter()
    sub = df_cox[[time_col, event_col, "era4"]].dropna()
    cph.fit(sub, duration_col=time_col, event_col=event_col, show_progress=False)
    row = cph.summary.loc["era4"]
    return {
        "HR": np.exp(row["coef"]),
        "CI_lo": np.exp(row["coef lower 95%"]),
        "CI_hi": np.exp(row["coef upper 95%"]),
        "p": row["p"],
    }


def fit_adjusted(df_cox, time_col, event_col):
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col=time_col, event_col=event_col, show_progress=False)
    row = cph.summary.loc["era4"]
    return {
        "HR": np.exp(row["coef"]),
        "CI_lo": np.exp(row["coef lower 95%"]),
        "CI_hi": np.exp(row["coef upper 95%"]),
        "p": row["p"],
    }


def build_table(df_os, df_pca):
    rows = []
    configs = [
        ("Overall Survival", df_os, "os_months", "os_event"),
        ("Prostate Cancer-Specific Survival", df_pca, "pca_months", "pca_event"),
    ]
    for surv_label, df, time_col, event_col in configs:
        for race in RACES:
            for stage in STAGES:
                df_sub = df[
                    (df["race"] == race)
                    & (df["m_stage"] == stage)
                    & df["era"].isin(ERAS_INCLUDED)
                ].copy()
                if len(df_sub) == 0:
                    continue
                n_era3 = (df_sub["era"] == 3).sum()
                n_era4 = (df_sub["era"] == 4).sum()
                df_cox = prepare_cox_df(df_sub, time_col, event_col)
                try:
                    uv = fit_unadjusted(df_cox, time_col, event_col)
                except Exception:
                    uv = {"HR": np.nan, "CI_lo": np.nan, "CI_hi": np.nan, "p": np.nan}
                try:
                    mv = fit_adjusted(df_cox, time_col, event_col)
                except Exception:
                    mv = {"HR": np.nan, "CI_lo": np.nan, "CI_hi": np.nan, "p": np.nan}
                rows.append(
                    {
                        "Survival": surv_label,
                        "Race": race,
                        "Stage": stage,
                        f"n {ERA_LABELS[3]}": f"{n_era3:,}",
                        f"n {ERA_LABELS[4]}": f"{n_era4:,}",
                        "HR (95% CI)": fmt_hr(uv["HR"], uv["CI_lo"], uv["CI_hi"]),
                        "p-value": fmt_p(uv["p"]),
                        "Adjusted HR (95% CI)": fmt_hr(
                            mv["HR"], mv["CI_lo"], mv["CI_hi"]
                        ),
                        "Adj p-value": fmt_p(mv["p"]),
                    }
                )
    cols = [
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
    return pd.DataFrame(rows, columns=cols)

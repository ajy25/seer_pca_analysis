
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2_contingency
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import multivariate_logrank_test

import overall_survival_data
import prostate_cancer_specific_survival_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERA_LABELS = overall_survival_data.ERA_LABELS

TABLE1_ERA_LABELS = {
    **ERA_LABELS,
    3: "early ARPI (2010-2016)",
    4: "modern ARPI (2017-2020)",
}


def km_median_ci(times, events):
    if len(times) == 0 or events.sum() == 0:
        return np.nan, np.nan, np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)
    med = kmf.median_survival_time_
    ci = median_survival_times(kmf.confidence_interval_)
    lo, hi = ci.iloc[0, 0], ci.iloc[0, 1]
    return med, lo, hi


def derive_followup_censoring_event(end_calc_status):
    status_str = end_calc_status.astype(str).str.strip().str.lower()
    return status_str.eq("untraced").astype(int)


def fmt_median(med, lo, hi):
    if np.isnan(med):
        return "NR"
    lo_s = f"{lo:.0f}" if not np.isnan(lo) else "NR"
    hi_s = f"{hi:.0f}" if not np.isnan(hi) else "NR"
    return f"{med:.0f} ({lo_s}-{hi_s})"


def kruskal_p(groups):
    filtered = [g.dropna() for g in groups]
    if any(len(g) == 0 for g in filtered):
        return np.nan
    _, p = kruskal(*filtered)
    return p


def fmt_p(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def fmt_n_pct(count, denom):
    if denom == 0:
        return "0 (0.00%)"
    pct = 100.0 * count / denom
    return f"{count:,} ({pct:.2f}%)"


def chi_square_p(df, group_col, cat_col):
    sub = df[[group_col, cat_col]].dropna()
    if sub.empty:
        return np.nan
    contingency = pd.crosstab(sub[group_col], sub[cat_col])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan
    _, p, _, _ = chi2_contingency(contingency)
    return p


def logrank_p(df, time_col, event_col, group_col):
    sub = df[[time_col, event_col, group_col]].dropna()
    if sub.empty or sub[group_col].nunique() < 2:
        return np.nan
    result = multivariate_logrank_test(
        event_durations=sub[time_col],
        groups=sub[group_col],
        event_observed=sub[event_col],
    )
    return result.p_value


def build_table1(df_os, df_pca):
    eras = [1, 2, 3, 4]
    era_labels_map = {
        e: TABLE1_ERA_LABELS[e].replace("–", "-") for e in eras
    }
    rows = []

    row = {"Variable": "Sample size", "p-value": ""}
    for e in eras:
        row[era_labels_map[e]] = f"{int((df_os['era'] == e).sum()):,}"
    rows.append(row)

    era_ages = [df_os.loc[df_os["era"] == e, "age"].dropna() for e in eras]
    row = {
        "Variable": "Median age, years (range)",
        "p-value": fmt_p(kruskal_p(era_ages)),
    }
    for e, ages in zip(eras, era_ages):
        row[era_labels_map[e]] = (
            f"{ages.median():.0f} ({ages.min():.0f}-{ages.max():.0f})"
            if len(ages) > 0
            else ""
        )
    rows.append(row)

    df_os_tmp = df_os.copy()
    df_os_tmp["followup_event"] = derive_followup_censoring_event(
        df_os_tmp["end_calc_status"]
    )
    p_fu = logrank_p(df_os_tmp, "os_months", "followup_event", "era")
    row = {"Variable": "Median follow-up, months (range)", "p-value": fmt_p(p_fu)}
    for e in eras:
        sub = df_os_tmp[df_os_tmp["era"] == e]
        med_fu, _, _ = km_median_ci(sub["os_months"], sub["followup_event"])
        all_fu = sub["os_months"].dropna()
        row[era_labels_map[e]] = (
            f"{med_fu:.0f} ({all_fu.min():.0f}-{all_fu.max():.0f})"
            if len(all_fu) > 0
            else ""
        )
    rows.append(row)

    row = {"Variable": "Median overall survival, months (95% CI)", "p-value": ""}
    for e in eras:
        sub = df_os[df_os["era"] == e]
        if len(sub) > 0:
            med, lo, hi = km_median_ci(sub["os_months"], sub["os_event"])
            row[era_labels_map[e]] = fmt_median(med, lo, hi)
        else:
            row[era_labels_map[e]] = ""
    rows.append(row)

    row = {"Variable": "Median cause-specific survival, months (95% CI)", "p-value": ""}
    for e in eras:
        sub = df_pca[df_pca["era"] == e]
        if len(sub) > 0:
            med, lo, hi = km_median_ci(sub["pca_months"], sub["pca_event"])
            row[era_labels_map[e]] = fmt_median(med, lo, hi)
        else:
            row[era_labels_map[e]] = ""
    rows.append(row)

    race_p = chi_square_p(df_os, "era", "race")
    rows.append(
        {
            "Variable": "Race/ethnicity",
            "p-value": fmt_p(race_p),
            **{era_labels_map[e]: "" for e in eras},
        }
    )
    race_order = [
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Hispanic",
        "Non-Hispanic Asian/PI",
        "Non-Hispanic AI/AN",
        "Non-Hispanic Unknown Race",
    ]
    for race in race_order:
        row = {"Variable": f"  {race}", "p-value": ""}
        for e in eras:
            sub = df_os[df_os["era"] == e]
            denom = int(sub["race"].notna().sum())
            count = int((sub["race"] == race).sum())
            row[era_labels_map[e]] = fmt_n_pct(count, denom)
        rows.append(row)

    stage_p = chi_square_p(df_os, "era", "m_stage")
    rows.append(
        {
            "Variable": "Stage",
            "p-value": fmt_p(stage_p),
            **{era_labels_map[e]: "" for e in eras},
        }
    )
    stage_order = ["M1", "M1a", "M1b", "M1c", "M1NOS"]
    for stage in stage_order:
        row = {"Variable": f"  {stage}", "p-value": ""}
        for e in eras:
            sub = df_os[df_os["era"] == e]
            denom = int(sub["m_stage"].notna().sum())
            count = int((sub["m_stage"] == stage).sum())
            row[era_labels_map[e]] = fmt_n_pct(count, denom)
        rows.append(row)

    treatment_vars = [
        ("Radical Prostatectomy", "prostatectomy"),
        ("Radiotherapy", "radiotherapy"),
        ("Prostate Surgery", "prostate_surgery"),
    ]
    for label, col in treatment_vars:
        p_val = chi_square_p(df_os, "era", col)
        rows.append(
            {
                "Variable": label,
                "p-value": fmt_p(p_val),
                **{era_labels_map[e]: "" for e in eras},
            }
        )
        for sub_label, val in [("  Yes", 1), ("  No", 0)]:
            row = {"Variable": sub_label, "p-value": ""}
            for e in eras:
                sub = df_os[df_os["era"] == e]
                denom = int(sub[col].notna().sum())
                count = int((sub[col] == val).sum())
                row[era_labels_map[e]] = fmt_n_pct(count, denom)
            rows.append(row)

    col_order = ["Variable"] + [era_labels_map[e] for e in eras] + ["p-value"]
    return pd.DataFrame(rows, columns=col_order)


def main():
    df_os = overall_survival_data.load_and_prepare_data()
    df_pca = prostate_cancer_specific_survival_data.load_and_prepare_data()
    table = build_table1(df_os, df_pca)
    out_path = OUTPUT_DIR / "12_patient_characteristics.csv"
    table.to_csv(out_path, index=False)


if __name__ == "__main__":
    main()

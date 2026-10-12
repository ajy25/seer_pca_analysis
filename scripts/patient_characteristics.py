from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times
from scipy.stats import chi2_contingency, kruskal

import overall_survival_data
import prostate_cancer_specific_survival_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
ERA_LABELS = overall_survival_data.ERA_LABELS
TABLE1_ERA_LABELS = {
    **ERA_LABELS,
    3: "early ARPI (2010-2016)",
    4: "modern ARPI (2017-2020)",
}


def km_median_ci(times, events):
    if len(times) == 0 or events.sum() == 0:
        return np.nan, np.nan, np.nan
    km_model = KaplanMeierFitter()
    km_model.fit(times, event_observed=events)
    median = km_model.median_survival_time_
    ci = median_survival_times(km_model.confidence_interval_)
    lower, upper = ci.iloc[0, 0], ci.iloc[0, 1]
    return median, lower, upper


def derive_followup_censoring_event(end_calc_status):
    status_str = end_calc_status.astype(str).str.strip().str.lower()
    return status_str.eq("untraced").astype(int)


def format_median(median, lower, upper):
    if np.isnan(median):
        return "NR"
    lower_text = f"{lower:.0f}" if not np.isnan(lower) else "NR"
    upper_text = f"{upper:.0f}" if not np.isnan(upper) else "NR"
    return f"{median:.0f} ({lower_text}-{upper_text})"


def kruskal_p(groups):
    filtered = [g.dropna() for g in groups]
    if any(len(g) == 0 for g in filtered):
        return np.nan
    _, p = kruskal(*filtered)
    return p


def format_p_value(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def format_count_percent(count, denom):
    if denom == 0:
        return "0 (0.00%)"
    pct = 100.0 * count / denom
    return f"{count:,} ({pct:.2f}%)"


def chi_square_p(data, group_col, cat_col):
    subset = data[[group_col, cat_col]].dropna()
    if subset.empty:
        return np.nan
    contingency = pd.crosstab(subset[group_col], subset[cat_col])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan
    _, p, _, _ = chi2_contingency(contingency)
    return p


def logrank_p(data, time_column, event_column, group_col):
    subset = data[[time_column, event_column, group_col]].dropna()
    if subset.empty or subset[group_col].nunique() < 2:
        return np.nan
    result = multivariate_logrank_test(
        event_durations=subset[time_column],
        groups=subset[group_col],
        event_observed=subset[event_column],
    )
    return result.p_value


def build_table1(os_data, pcss_data):
    eras = [1, 2, 3, 4]
    era_labels_map = {e: TABLE1_ERA_LABELS[e].replace("–", "-") for e in eras}
    rows = []

    row = {"Variable": "Sample size", "p-value": ""}
    for e in eras:
        row[era_labels_map[e]] = f"{int((os_data['era'] == e).sum()):,}"
    rows.append(row)

    era_ages = [os_data.loc[os_data["era"] == e, "age"].dropna() for e in eras]
    row = {
        "Variable": "Median age, years (range)",
        "p-value": format_p_value(kruskal_p(era_ages)),
    }
    for e, ages in zip(eras, era_ages):
        row[era_labels_map[e]] = (
            f"{ages.median():.0f} ({ages.min():.0f}-{ages.max():.0f})"
            if len(ages) > 0
            else ""
        )
    rows.append(row)

    df_os_tmp = os_data.copy()
    df_os_tmp["followup_event"] = derive_followup_censoring_event(
        df_os_tmp["end_calc_status"]
    )
    p_fu = logrank_p(df_os_tmp, "os_months", "followup_event", "era")
    row = {
        "Variable": "Median follow-up, months (range)",
        "p-value": format_p_value(p_fu),
    }
    for e in eras:
        subset = df_os_tmp[df_os_tmp["era"] == e]
        med_fu, _, _ = km_median_ci(subset["os_months"], subset["followup_event"])
        all_fu = subset["os_months"].dropna()
        row[era_labels_map[e]] = (
            f"{med_fu:.0f} ({all_fu.min():.0f}-{all_fu.max():.0f})"
            if len(all_fu) > 0
            else ""
        )
    rows.append(row)

    row = {"Variable": "Median overall survival, months (95% CI)", "p-value": ""}
    for e in eras:
        subset = os_data[os_data["era"] == e]
        if len(subset) > 0:
            median, lower, upper = km_median_ci(subset["os_months"], subset["os_event"])
            row[era_labels_map[e]] = format_median(median, lower, upper)
        else:
            row[era_labels_map[e]] = ""
    rows.append(row)

    row = {"Variable": "Median cause-specific survival, months (95% CI)", "p-value": ""}
    for e in eras:
        subset = pcss_data[pcss_data["era"] == e]
        if len(subset) > 0:
            median, lower, upper = km_median_ci(
                subset["pca_months"], subset["pca_event"]
            )
            row[era_labels_map[e]] = format_median(median, lower, upper)
        else:
            row[era_labels_map[e]] = ""
    rows.append(row)

    race_p = chi_square_p(os_data, "era", "race")
    rows.append(
        {
            "Variable": "Race/ethnicity",
            "p-value": format_p_value(race_p),
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
            subset = os_data[os_data["era"] == e]
            denom = int(subset["race"].notna().sum())
            count = int((subset["race"] == race).sum())
            row[era_labels_map[e]] = format_count_percent(count, denom)
        rows.append(row)

    stage_p = chi_square_p(os_data, "era", "m_stage")
    rows.append(
        {
            "Variable": "Stage",
            "p-value": format_p_value(stage_p),
            **{era_labels_map[e]: "" for e in eras},
        }
    )
    stage_order = ["M1", "M1a", "M1b", "M1c", "M1NOS"]
    for stage in stage_order:
        row = {"Variable": f"  {stage}", "p-value": ""}
        for e in eras:
            subset = os_data[os_data["era"] == e]
            denom = int(subset["m_stage"].notna().sum())
            count = int((subset["m_stage"] == stage).sum())
            row[era_labels_map[e]] = format_count_percent(count, denom)
        rows.append(row)

    treatment_vars = [
        ("Radical Prostatectomy", "prostatectomy"),
        ("Radiotherapy", "radiotherapy"),
    ]
    for label, col in treatment_vars:
        p_value = chi_square_p(os_data, "era", col)
        rows.append(
            {
                "Variable": label,
                "p-value": format_p_value(p_value),
                **{era_labels_map[e]: "" for e in eras},
            }
        )
        negative_label = "  No/unknown" if col == "radiotherapy" else "  No"
        for sub_label, val in [("  Yes", 1), (negative_label, 0)]:
            row = {"Variable": sub_label, "p-value": ""}
            for e in eras:
                subset = os_data[os_data["era"] == e]
                denom = int(subset[col].notna().sum())
                count = int((subset[col] == val).sum())
                row[era_labels_map[e]] = format_count_percent(count, denom)
            rows.append(row)

    col_order = ["Variable"] + [era_labels_map[e] for e in eras] + ["p-value"]
    return pd.DataFrame(rows, columns=col_order)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    os_data = overall_survival_data.load_and_prepare_data()
    pcss_data = prostate_cancer_specific_survival_data.load_and_prepare_data()
    table = build_table1(os_data, pcss_data)
    output_path = TABLE_DIR / "12_patient_characteristics.csv"
    table.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()

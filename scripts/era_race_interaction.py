
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test

import overall_survival_data
import prostate_cancer_specific_survival_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
CENSOR_TIME = 24
CENTER_TIME = 6.0
Z_975 = 1.959963984540054
RACES = ["Non-Hispanic White", "Non-Hispanic Black"]
M_STAGE_REF = "M1b"


def prepare_cox_df(df, time_col, event_col, include_interaction):
    cox = (
        df[
            [
                time_col,
                event_col,
                "era",
                "race",
                "age",
                "m_stage",
                "prostatectomy",
                "radiotherapy",
            ]
        ]
        .copy()
        .dropna()
    )
    mask = cox[time_col] > CENSOR_TIME
    cox.loc[mask, event_col] = 0
    cox.loc[mask, time_col] = CENSOR_TIME
    cox["era4"] = (cox["era"] == 4).astype(int)
    cox["black"] = (cox["race"] == "Non-Hispanic Black").astype(int)
    cox["era4_x_black"] = cox["era4"] * cox["black"]
    stage_dummies = pd.get_dummies(
        cox["m_stage"],
        prefix="m_stage",
        drop_first=False,
    )
    stage_dummies = stage_dummies.drop(
        columns=[f"m_stage_{M_STAGE_REF}"],
        errors="ignore",
    )
    columns = [
        time_col,
        event_col,
        "era4",
        "black",
        "age",
        "prostatectomy",
        "radiotherapy",
    ]
    if include_interaction:
        columns.insert(4, "era4_x_black")
    return pd.concat([cox[columns], stage_dummies], axis=1)


def conventional_fit(frame, time_col, event_col):
    model = CoxPHFitter()
    model.fit(
        frame,
        duration_col=time_col,
        event_col=event_col,
        show_progress=False,
    )
    return model


def ph_screen(model, frame):
    return proportional_hazard_test(
        model,
        frame,
        time_transform="rank",
    ).summary.sort_values("p")


def to_counting_process(
    frame,
    time_col,
    event_col,
    flagged_terms,
):
    base = frame.reset_index(drop=True).copy()
    base.insert(0, "id", np.arange(len(base), dtype=int))
    durations = base[time_col].astype(int).to_numpy()
    if not np.allclose(durations, base[time_col].to_numpy()):
        raise ValueError("Expected integer SEER survival months")
    if np.any(durations <= 0):
        raise ValueError("Expected strictly positive follow-up")

    long = base.loc[base.index.repeat(durations)].copy()
    long["stop"] = long.groupby("id").cumcount() + 1
    long["start"] = long["stop"] - 1
    long["event_long"] = (
        (long[event_col] == 1) & (long["stop"] == long[time_col])
    ).astype(int)
    log_time = np.log(long["stop"].to_numpy(dtype=float) / CENTER_TIME)
    for term in flagged_terms:
        long[f"tv__{term}"] = long[term].to_numpy() * log_time
    return long.drop(columns=[time_col, event_col])


def extended_fit(frame, time_col, event_col, flagged_terms):
    long = to_counting_process(
        frame,
        time_col,
        event_col,
        flagged_terms,
    )
    model = CoxTimeVaryingFitter()
    model.fit(
        long,
        id_col="id",
        start_col="start",
        stop_col="stop",
        event_col="event_long",
        show_progress=False,
    )
    return model


def normal_two_sided_p(z):
    return 1.0 - erf(abs(z) / sqrt(2.0))


def combination_estimate(model, terms):
    covariance = model.variance_matrix_.copy()
    covariance.columns = covariance.index
    coef = sum(float(model.params_.loc[term]) for term in terms)
    variance = 0.0
    for first in terms:
        for second in terms:
            variance += float(covariance.loc[first, second])
    se = sqrt(max(variance, 0.0))
    return {
        "HR": float(np.exp(coef)),
        "CI lower": float(np.exp(coef - Z_975 * se)),
        "CI upper": float(np.exp(coef + Z_975 * se)),
        "p": normal_two_sided_p(coef / se),
    }


def single_estimate(model, term):
    row = model.summary.loc[term]
    coef = float(row["coef"])
    se = float(row["se(coef)"])
    return {
        "HR": float(np.exp(coef)),
        "CI lower": float(np.exp(coef - Z_975 * se)),
        "CI upper": float(np.exp(coef + Z_975 * se)),
        "p": float(row["p"]),
    }


def pretty_term(term):
    labels = {
        "age": "Age",
        "prostatectomy": "Prostatectomy",
        "radiotherapy": "Radiotherapy",
        "black": "Non-Hispanic Black race",
    }
    if term in labels:
        return labels[term]
    if term.startswith("m_stage_"):
        return term.replace("m_stage_", "Metastatic classification: ", 1)
    return term


def fmt_estimate(estimate):
    return (
        f"{estimate['HR']:.2f} "
        f"({estimate['CI lower']:.2f}-{estimate['CI upper']:.2f})"
    )


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def time_varying_coefficient_rows(endpoint, model, flagged_terms):
    rows = []
    for term in flagged_terms:
        row = model.summary.loc[term]
        tv_row = model.summary.loc[f"tv__{term}"]
        coef = float(row["coef"])
        se = float(row["se(coef)"])
        rows.append(
            {
                "Corresponding result": "Supplementary Table 7",
                "Endpoint": endpoint,
                "Stratum": "Overall interaction model",
                "Term": pretty_term(term),
                "HR at 6 months": float(np.exp(coef)),
                "95% CI lower at 6 months": float(
                    np.exp(coef - Z_975 * se)
                ),
                "95% CI upper at 6 months": float(
                    np.exp(coef + Z_975 * se)
                ),
                "p at 6 months": float(row["p"]),
                "Time interaction coefficient": float(tv_row["coef"]),
                "Time interaction p": float(tv_row["p"]),
                "Time function": "log",
                "HR multiplier per doubling of follow-up": float(
                    np.exp(float(tv_row["coef"]) * np.log(2.0))
                ),
            }
        )
    return rows


def fit_endpoint(endpoint, df, time_col, event_col):
    subset = df[
        df["era"].isin([3, 4]) & df["race"].isin(RACES)
    ].copy()
    reduced = prepare_cox_df(
        subset,
        time_col,
        event_col,
        include_interaction=False,
    )
    full = prepare_cox_df(
        subset,
        time_col,
        event_col,
        include_interaction=True,
    )
    reduced_conventional = conventional_fit(reduced, time_col, event_col)
    full_conventional = conventional_fit(full, time_col, event_col)
    reduced_screen = ph_screen(reduced_conventional, reduced)
    full_screen = ph_screen(full_conventional, full)

    displayed_terms = {"era4", "era4_x_black"}
    for term in displayed_terms:
        if term in full_screen.index and float(full_screen.loc[term, "p"]) < ALPHA:
            raise RuntimeError(
                f"{endpoint}: displayed term {term} failed PH screening"
            )
    flagged = sorted(
        {
            term
            for screen in [reduced_screen, full_screen]
            for term, row in screen.iterrows()
            if float(row["p"]) < ALPHA and term not in displayed_terms
        }
    )
    reduced_model = extended_fit(
        reduced,
        time_col,
        event_col,
        flagged,
    )
    full_model = extended_fit(
        full,
        time_col,
        event_col,
        flagged,
    )

    likelihood_ratio = 2.0 * (
        float(full_model.log_likelihood_)
        - float(reduced_model.log_likelihood_)
    )
    lrt_p = 1.0 - erf(sqrt(max(likelihood_ratio, 0.0) / 2.0))
    white = single_estimate(full_model, "era4")
    black = combination_estimate(
        full_model,
        ["era4", "era4_x_black"],
    )
    interaction = single_estimate(full_model, "era4_x_black")
    black_vs_white_era3 = single_estimate(full_model, "black")
    black_vs_white_era4 = combination_estimate(
        full_model,
        ["black", "era4_x_black"],
    )

    def estimate_and_p(estimate, p_value=None):
        p_value = estimate["p"] if p_value is None else p_value
        p_text = fmt_p(p_value)
        p_text = f"p{p_text}" if p_text.startswith("<") else f"p={p_text}"
        return f"{fmt_estimate(estimate)}; {p_text}"

    endpoint_label = {
        "Overall Survival": "OS",
        "Prostate Cancer-Specific Survival": "PCSS",
    }[endpoint]
    row = {
        "Endpoint": endpoint_label,
        "N (events)": f"{len(full):,} ({int(full[event_col].sum()):,})",
        "E4 vs E3, NHW aHR (95% CI); p": estimate_and_p(white),
        "E4 vs E3, NHB aHR (95% CI); p": estimate_and_p(black),
        "Interaction ratio (95% CI); LRT p": estimate_and_p(
            interaction,
            lrt_p,
        ),
        "NHB vs NHW, E3 aHR (95% CI); p": estimate_and_p(
            black_vs_white_era3
        ),
        "NHB vs NHW, E4 aHR (95% CI); p": estimate_and_p(
            black_vs_white_era4
        ),
    }
    config_row = {
        "Corresponding result": "Supplementary Table 7",
        "Endpoint": endpoint,
        "Stratum": "Overall interaction model",
        "N": len(full),
        "Events": int(full[event_col].sum()),
        "Era-term Schoenfeld p": float(full_screen.loc["era4", "p"]),
        "Interaction-term Schoenfeld p": float(
            full_screen.loc["era4_x_black", "p"]
        ),
        "Final model": "Extended Cox",
        "Time function": "log",
        "Time-varying adjustment terms": "; ".join(
            pretty_term(term) for term in flagged
        ),
        "All terms flagged by initial PH screen": "; ".join(
            pretty_term(term) for term in flagged
        ),
    }
    term_rows = []
    for model_name, screen in [
        ("Reduced model", reduced_screen),
        ("Full interaction model", full_screen),
    ]:
        for term, term_row in screen.iterrows():
            term_rows.append(
                {
                    "Endpoint": endpoint,
                    "Model": model_name,
                    "Term": term,
                    "Schoenfeld test statistic": float(
                        term_row["test_statistic"]
                    ),
                    "Schoenfeld p": float(term_row["p"]),
                }
            )
    coefficient_rows = time_varying_coefficient_rows(
        endpoint,
        full_model,
        flagged,
    )
    return row, config_row, coefficient_rows, term_rows


def manuscript_ph_tables(config_rows, coefficient_rows):
    configurations = pd.read_csv(
        OUTPUT_DIR / "21_ph_aware_model_configurations.csv"
    )
    configurations["Interaction-term Schoenfeld p"] = np.nan
    configurations = pd.concat(
        [configurations, pd.DataFrame(config_rows)],
        ignore_index=True,
    )
    coefficients = pd.read_csv(
        OUTPUT_DIR / "21_ph_aware_coefficients.csv"
    )
    coefficients = pd.concat(
        [coefficients, pd.DataFrame(coefficient_rows)],
        ignore_index=True,
    )

    output_order = {
        "Table 3": 0,
        "Supplementary Table 3": 1,
        "Supplementary Table 4": 2,
        "Supplementary Table 6": 3,
        "Supplementary Table 7": 4,
        "Supplementary Table 8": 5,
    }
    configurations["_order"] = configurations[
        "Corresponding result"
    ].map(output_order)
    configurations = configurations.sort_values(
        ["_order", "Endpoint", "Stratum"]
    )
    configurations["Era-term Schoenfeld p"] = configurations[
        "Era-term Schoenfeld p"
    ].map(fmt_p)
    configurations["Interaction-term Schoenfeld p"] = configurations[
        "Interaction-term Schoenfeld p"
    ].map(lambda value: "" if pd.isna(value) else fmt_p(value))

    def compact_terms(value):
        if pd.isna(value) or not str(value).strip():
            return ""
        term_labels = {
            "Age": "age",
            "Radiotherapy": "RT",
            "Prostatectomy": "RP",
            "Non-Hispanic Unknown Race": "NHU",
            "Non-Hispanic Black": "NHB",
            "Non-Hispanic Asian/PI": "NHA/PI",
        }
        terms = []
        for term in str(value).split(";"):
            term = term.strip()
            term = term.replace("Metastatic classification: ", "")
            term = term.replace("Race: ", "")
            terms.append(term_labels.get(term, term))
        return ", ".join(terms)

    def displayed_term_p(row):
        values = [f"Era {row['Era-term Schoenfeld p']}"]
        interaction_p = row["Interaction-term Schoenfeld p"]
        if interaction_p:
            values.append(f"Int {interaction_p}")
        return ", ".join(values)

    def final_adjustment_changes(row):
        changes = []
        time_varying = compact_terms(row["Time-varying adjustment terms"])
        omitted = compact_terms(row["Omitted nonestimable adjustment terms"])
        if time_varying:
            changes.append(f"TV: {time_varying}")
        if omitted:
            changes.append(f"Omit: {omitted}")
        return ". ".join(changes) if changes else "None"

    endpoint_labels = {
        "Overall Survival": "OS",
        "Prostate Cancer-Specific Survival": "PCSS",
    }
    configurations["Result"] = configurations["Corresponding result"].str.replace(
        "Supplementary Table ", "S", regex=False
    )

    def compact_analysis(row):
        stratum = str(row["Stratum"])
        stratum = stratum.replace("Overall interaction model", "int")
        stratum = stratum.replace("Non-Hispanic Black", "NHB")
        stratum = stratum.replace("Non-Hispanic White", "NHW")
        stratum = stratum.replace("Overall", "overall")
        stratum = stratum.replace("; ", "/")
        endpoint = endpoint_labels.get(row["Endpoint"], row["Endpoint"])
        return f"{endpoint}/{stratum}"

    configurations["Analysis"] = configurations.apply(compact_analysis, axis=1)
    configurations["PH p"] = configurations.apply(
        displayed_term_p,
        axis=1,
    )
    configurations["Changes"] = configurations.apply(
        final_adjustment_changes,
        axis=1,
    )
    configurations = configurations[
        [
            "Result",
            "Analysis",
            "PH p",
            "Changes",
        ]
    ]

    coefficients["_order"] = coefficients[
        "Corresponding result"
    ].map(output_order)
    coefficients = coefficients.sort_values(
        ["_order", "Endpoint", "Stratum", "Term"]
    )
    coefficients["Adjusted HR at 6 months (95% CI)"] = coefficients.apply(
        lambda row: (
            f"{row['HR at 6 months']:.2f} "
            f"({row['95% CI lower at 6 months']:.2f}-"
            f"{row['95% CI upper at 6 months']:.2f})"
        ),
        axis=1,
    )
    coefficients["p at 6 months"] = coefficients[
        "p at 6 months"
    ].map(fmt_p)
    coefficients["HR multiplier per doubling of follow-up"] = coefficients[
        "HR multiplier per doubling of follow-up"
    ].map(lambda value: f"{value:.3f}")
    coefficients["Time-interaction p"] = coefficients[
        "Time interaction p"
    ].map(fmt_p)

    def inline_p(value):
        value = str(value)
        return f"p{value}" if value.startswith("<") else f"p={value}"

    coefficients["Result"] = coefficients["Corresponding result"].str.replace(
        "Supplementary Table ", "S", regex=False
    )
    coefficients["Analysis"] = coefficients.apply(compact_analysis, axis=1)
    coefficients["TV term"] = coefficients["Term"].map(compact_terms)
    coefficients["6-month aHR (95% CI); p"] = coefficients.apply(
        lambda row: (
            f"{row['Adjusted HR at 6 months (95% CI)']}; "
            f"{inline_p(row['p at 6 months'])}"
        ),
        axis=1,
    )
    coefficients["Per-doubling HR multiplier; p"] = coefficients.apply(
        lambda row: (
            f"{row['HR multiplier per doubling of follow-up']}; "
            f"{inline_p(row['Time-interaction p'])}"
        ),
        axis=1,
    )
    coefficients = coefficients[
        [
            "Result",
            "Analysis",
            "TV term",
            "6-month aHR (95% CI); p",
            "Per-doubling HR multiplier; p",
        ]
    ]
    return configurations, coefficients


def vertical_interaction_table(rows):
    by_endpoint = {row["Endpoint"]: row for row in rows}
    endpoints = ["OS", "PCSS"]
    metrics = [column for column in rows[0] if column != "Endpoint"]
    return pd.DataFrame(
        {
            "Endpoint": metrics,
            **{
                endpoint: [by_endpoint[endpoint][metric] for metric in metrics]
                for endpoint in endpoints
            },
        }
    )


def main():
    df_os = overall_survival_data.load_and_prepare_data()
    df_pcss = prostate_cancer_specific_survival_data.load_and_prepare_data()
    rows = []
    config_rows = []
    coefficient_rows = []
    term_rows = []
    for endpoint, df, time_col, event_col in [
        ("Overall Survival", df_os, "os_months", "os_event"),
        (
            "Prostate Cancer-Specific Survival",
            df_pcss,
            "pca_months",
            "pca_event",
        ),
    ]:
        row, config_row, new_coefficient_rows, new_term_rows = fit_endpoint(
            endpoint,
            df,
            time_col,
            event_col,
        )
        rows.append(row)
        config_rows.append(config_row)
        coefficient_rows.extend(new_coefficient_rows)
        term_rows.extend(new_term_rows)
    vertical_interaction_table(rows).to_csv(
        OUTPUT_DIR / "22_era_race_interaction.csv",
        index=False,
    )
    pd.DataFrame(term_rows).to_csv(
        OUTPUT_DIR / "22_era_race_interaction_ph_terms.csv",
        index=False,
    )
    configurations, coefficients = manuscript_ph_tables(
        config_rows,
        coefficient_rows,
    )
    configurations.to_csv(
        OUTPUT_DIR / "22_supplementary_table_1_ph_configurations.csv",
        index=False,
    )
    coefficients.to_csv(
        OUTPUT_DIR / "22_supplementary_table_2_time_varying_coefficients.csv",
        index=False,
    )


if __name__ == "__main__":
    main()

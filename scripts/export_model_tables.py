import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
TERM_LABELS = {
    "age": "age",
    "age_per10": "age (per year)",
    "radiotherapy": "RT",
    "prostatectomy": "RP",
    "era2": "E2 vs E1",
    "era3": "E3 vs E1",
    "era4": "E4 vs E1",
    "race_Non-Hispanic Asian/PI": "NHA/PI",
    "m_stage_M1c": "M1c",
    "m_stage_M1NOS": "M1NOS",
}


def format_p_value(probability):
    return "<0.001" if probability < 0.001 else f"{probability:.3f}"


def format_inline_p_value(probability):
    prefix = "p" if probability < 0.001 else "p="
    return prefix + format_p_value(probability)


def label(term):
    return TERM_LABELS.get(term, term)


def coefficient_row(
    result,
    analysis,
    term,
    coefficient,
    standard_error,
    time_coefficient,
    time_standard_error,
):
    if term == "age_per10":
        coefficient, standard_error, time_coefficient, time_standard_error = [
            value / 10
            for value in [
                coefficient,
                standard_error,
                time_coefficient,
                time_standard_error,
            ]
        ]
    coefficient_p = 2 * norm.sf(abs(coefficient / standard_error))
    time_p = 2 * norm.sf(abs(time_coefficient / time_standard_error))
    lower = np.exp(coefficient - 1.95996398454 * standard_error)
    upper = np.exp(coefficient + 1.95996398454 * standard_error)
    term_label = "E4 vs E3" if result == "S9" and term == "era4" else label(term)
    return {
        "Result": result,
        "Analysis": analysis,
        "TV term": term_label,
        "6-month aHR (95% CI); p": (
            f"{np.exp(coefficient):.2f} ({lower:.2f}-{upper:.2f}); "
            f"{format_inline_p_value(coefficient_p)}"
        ),
        "Per-doubling HR multiplier; p": (
            f"{np.exp(time_coefficient * np.log(2)):.3f}; {format_inline_p_value(time_p)}"
        ),
    }


def figure3_model_rows(endpoint):
    prefix = TABLE_DIR / f"23_{endpoint.lower()}_four_era"
    configuration = json.loads(Path(f"{prefix}_model.json").read_text())
    screening = pd.read_csv(f"{prefix}_ph_screen.csv")
    full_screen = screening[screening["Model"] == "Full"].set_index("Term")[
        "Schoenfeld p"
    ]
    ph_text = "; ".join(
        f"E{era} {format_p_value(full_screen[f'era{era}'])} / "
        f"Int {format_p_value(full_screen[f'era{era}_x_black'])}"
        for era in [2, 3, 4]
    )
    time_terms = configuration["Time-varying terms"]
    model_row = {
        "Result": "Fig 3",
        "Analysis": endpoint + "/four eras",
        "PH p": ph_text,
        "Changes": (
            "TV: "
            + ", ".join(label(term) for term in time_terms)
            + ". No M subclassification adjustment."
        ),
    }
    coefficients = pd.read_csv(f"{prefix}_coefficients.csv").set_index("Term")
    rows = []
    for term in time_terms:
        base_row = coefficients.loc[term]
        time_row = coefficients.loc["tv__" + term]
        rows.append(
            coefficient_row(
                "Fig 3",
                endpoint + "/four eras",
                term,
                base_row["Coefficient"],
                base_row["Standard error"],
                time_row["Coefficient"],
                time_row["Standard error"],
            )
        )
    return model_row, rows


def sensitivity_model_rows(endpoint):
    configuration = json.loads(
        (TABLE_DIR / f"24_{endpoint.lower()}_model.json").read_text()
    )
    time_terms = configuration["Time-varying terms"]
    model_row = {
        "Result": "S9",
        "Analysis": endpoint + "/no RP or RT",
        "PH p": f"Era {configuration['Era-term Schoenfeld p']:.4f}",
        "Changes": "TV: "
        + ", ".join(
            "E4 vs E3" if term == "era4" else label(term) for term in time_terms
        ),
    }
    coefficients = pd.read_csv(
        TABLE_DIR / f"24_{endpoint.lower()}_coefficients.csv",
    ).set_index("covariate")
    rows = []
    for term in time_terms:
        base_row = coefficients.loc[term]
        time_row = coefficients.loc["tv__" + term]
        rows.append(
            coefficient_row(
                "S9",
                endpoint + "/no RP or RT",
                term,
                base_row["coef"],
                base_row["se(coef)"],
                time_row["coef"],
                time_row["se(coef)"],
            )
        )
    return model_row, rows


def replace_model_rows(rows, filename):
    additions = pd.DataFrame(rows)
    existing = pd.read_csv(TABLE_DIR / filename).fillna("")
    retained = existing[~existing["Result"].isin(["Fig 3", "S9"])]
    pd.concat([retained, additions], ignore_index=True).to_csv(
        TABLE_DIR / filename, index=False
    )


def export_sensitivity_table():
    source = pd.read_csv(TABLE_DIR / "24_local_treatment_sensitivity.csv").set_index(
        "Endpoint"
    )
    metrics = [
        "n Early ARPI",
        "n Modern ARPI",
        "Events",
        "HR (95% CI); p",
        "aHR at 6 months (95% CI); p",
        "aHR at 12 months (95% CI); p",
        "aHR at 24 months (95% CI); p",
        "Era effect",
        "Era-term Schoenfeld p",
        "Final model",
        "Time-varying adjustment terms",
    ]
    table = pd.DataFrame(
        {
            "Measure": metrics,
            **{
                endpoint: [source.loc[endpoint, metric] for metric in metrics]
                for endpoint in ["OS", "PCSS"]
            },
        }
    )
    for endpoint in ["OS", "PCSS"]:
        table.loc[table["Measure"] == "Final model", endpoint] = "Adjusted Cox"
        for row_index in range(3):
            table.loc[row_index, endpoint] = f"{int(table.loc[row_index, endpoint]):,}"
        configuration = json.loads(
            (TABLE_DIR / f"24_{endpoint.lower()}_model.json").read_text()
        )
        sensitivity_labels = {"age": "Age", "era4": "era"}
        time_term_labels = [
            sensitivity_labels.get(term, label(term))
            for term in configuration["Time-varying terms"]
        ]
        table.loc[table["Measure"] == "Time-varying adjustment terms", endpoint] = (
            "; ".join(time_term_labels)
        )
        table.loc[table["Measure"] == "Era-term Schoenfeld p", endpoint] = (
            f"{configuration['Era-term Schoenfeld p']:.4f}"
        )
    table.to_csv(TABLE_DIR / "24_supplementary_table_9.csv", index=False)
    table.to_csv(TABLE_DIR / "24_supplementary_table_9.tsv", index=False, sep="\t")
    (TABLE_DIR / "24_supplementary_table_9.md").write_text(
        table.to_markdown(index=False)
    )


def export_display_estimates():
    estimates = pd.read_csv(TABLE_DIR / "23_adjusted_race_era_survival.csv")
    estimates["Survival (95% CI), %"] = estimates.apply(
        lambda row: (
            f"{100 * row['Standardized 24-month survival']:.1f} "
            f"({100 * row['95% CI lower']:.1f}–{100 * row['95% CI upper']:.1f})"
        ),
        axis=1,
    )
    columns = ["Endpoint", "Race", "Era", "n", "Survival (95% CI), %"]
    estimates[columns].to_csv(
        TABLE_DIR / "23_figure3_display_estimates.csv", index=False
    )


def main():
    model_rows = []
    coefficient_rows = []
    for endpoint in ["OS", "PCSS"]:
        for build_rows in [figure3_model_rows, sensitivity_model_rows]:
            model_row, rows = build_rows(endpoint)
            model_rows.append(model_row)
            coefficient_rows.extend(rows)
    replace_model_rows(model_rows, "22_supplementary_table_1_ph_configurations.csv")
    replace_model_rows(
        coefficient_rows, "22_supplementary_table_2_time_varying_coefficients.csv"
    )
    export_sensitivity_table()
    export_display_estimates()


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import norm

import overall_survival_data
import ph_aware_cox
import prostate_cancer_specific_survival_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
ANALYSIS_MONTHS = 24
REFERENCE_MONTH = 6.0
PH_P_THRESHOLD = 0.05
M_STAGE_REF = "M1b"


def prepare_cox_data(data, time_column, event_column):
    frame = (
        data[[time_column, event_column, "era", "age", "race", "m_stage"]]
        .copy()
        .dropna()
    )
    beyond_horizon = frame[time_column] > ANALYSIS_MONTHS
    frame.loc[beyond_horizon, event_column] = 0
    frame.loc[beyond_horizon, time_column] = ANALYSIS_MONTHS
    frame["era4"] = (frame["era"] == 4).astype(int)
    race_dummies = pd.get_dummies(
        frame["race"], prefix="race", drop_first=False, dtype=float
    ).drop(columns=["race_Non-Hispanic White"], errors="ignore")
    stage_dummies = pd.get_dummies(
        frame["m_stage"], prefix="m_stage", drop_first=False, dtype=float
    ).drop(columns=[f"m_stage_{M_STAGE_REF}"], errors="ignore")
    base_columns = [time_column, event_column, "era4", "age"]
    return pd.concat(
        [frame[base_columns].astype(float), race_dummies, stage_dummies], axis=1
    )


def estimate(model, term):
    row = model.summary.loc[term]
    return {
        "HR": float(np.exp(row["coef"])),
        "lower": float(np.exp(row["coef lower 95%"])),
        "upper": float(np.exp(row["coef upper 95%"])),
        "p": float(row["p"]),
    }


def format_estimate(value):
    p_text = "p<0.001" if value["p"] < 0.001 else f"p={value['p']:.3f}"
    return f"{value['HR']:.2f} ({value['lower']:.2f}-{value['upper']:.2f}); {p_text}"


def fit_endpoint(endpoint, data, time_column, event_column):
    exclusion_label = "No recorded radical prostatectomy or radiotherapy"
    subset = data[
        data["era"].isin([3, 4])
        & (data["prostatectomy"] == 0)
        & (data["radiotherapy"] == 0)
    ].copy()
    frame = prepare_cox_data(subset, time_column, event_column)
    unadjusted = CoxPHFitter()
    unadjusted.fit(
        frame[[time_column, event_column, "era4"]],
        duration_col=time_column,
        event_col=event_column,
        show_progress=False,
    )
    conventional = ph_aware_cox.fit_cox_model(frame, time_column, event_column)
    screen = ph_aware_cox.test_proportional_hazards(conventional, frame)
    flagged = [
        term for (term, row) in screen.iterrows() if float(row["p"]) < PH_P_THRESHOLD
    ]
    model = (
        ph_aware_cox.fit_time_varying_cox_model(
            frame, time_column, event_column, flagged, "log"
        )
        if flagged
        else conventional
    )
    adjusted = {
        "estimate": ph_aware_cox.target_estimate(model, "era4"),
        "target_ph_p": float(screen.loc["era4", "p"]),
        "model_type": "Extended Cox" if flagged else "Standard Cox",
        "flagged": flagged,
    }
    screen.to_csv(TABLE_DIR / f"24_{endpoint.lower()}_ph_screen.csv")
    model.summary.to_csv(TABLE_DIR / f"24_{endpoint.lower()}_coefficients.csv")
    model.variance_matrix_.to_csv(TABLE_DIR / f"24_{endpoint.lower()}_covariance.csv")

    def era_at(month):
        terms = list(model.params_.index)
        weights = np.zeros(len(terms))
        weights[terms.index("era4")] = 1
        if "tv__era4" in terms:
            weights[terms.index("tv__era4")] = np.log(month / REFERENCE_MONTH)
        beta = float(weights @ model.params_.to_numpy())
        se = np.sqrt(float(weights @ model.variance_matrix_.to_numpy() @ weights))
        return {
            "HR": np.exp(beta),
            "lower": np.exp(beta - 1.959963984540054 * se),
            "upper": np.exp(beta + 1.959963984540054 * se),
            "p": float(2 * norm.sf(abs(beta / se))),
        }

    estimates = {str(m): era_at(m) for m in [6, 12, 24]}
    config = {
        "Endpoint": endpoint,
        "N": len(frame),
        "Events": int(frame[event_column].sum()),
        "Era-term Schoenfeld p": adjusted["target_ph_p"],
        "Time-varying terms": flagged,
        "Era estimates by month": estimates,
    }
    (TABLE_DIR / f"24_{endpoint.lower()}_model.json").write_text(
        json.dumps(config, indent=2)
    )
    return {
        "Endpoint": endpoint,
        "Sensitivity analysis": exclusion_label,
        "n Early ARPI": int((subset["era"] == 3).sum()),
        "n Modern ARPI": int((subset["era"] == 4).sum()),
        "Events": int(frame[event_column].sum()),
        "HR (95% CI); p": format_estimate(estimate(unadjusted, "era4")),
        "aHR at 6 months (95% CI); p": format_estimate(estimates["6"]),
        "aHR at 12 months (95% CI); p": format_estimate(estimates["12"]),
        "aHR at 24 months (95% CI); p": format_estimate(estimates["24"]),
        "Era effect": "Log-time-varying" if "era4" in flagged else "Time-constant",
        "Era-term Schoenfeld p": ph_aware_cox.format_p_value(adjusted["target_ph_p"]),
        "Final model": adjusted["model_type"],
        "Time-varying adjustment terms": "; ".join(
            (ph_aware_cox.pretty_term(term) for term in adjusted["flagged"])
        ),
    }


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    endpoints = [
        ("OS", overall_survival_data.load_and_prepare_data(), "os_months", "os_event"),
        (
            "PCSS",
            prostate_cancer_specific_survival_data.load_and_prepare_data(),
            "pca_months",
            "pca_event",
        ),
    ]
    for endpoint, data, time_column, event_column in endpoints:
        rows.append(fit_endpoint(endpoint, data, time_column, event_column))
    pd.DataFrame(rows).to_csv(
        TABLE_DIR / "24_local_treatment_sensitivity.csv", index=False
    )


if __name__ == "__main__":
    main()

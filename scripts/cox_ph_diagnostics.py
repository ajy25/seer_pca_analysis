from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

import cox_by_race_models
import cox_by_stage_and_race_models
import cox_by_stage_models
import cox_os_models
import cox_pcss_models
import overall_survival_data
import prostate_cancer_specific_survival_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
PH_P_THRESHOLD = 0.05


def fit_model(frame, time_column, event_column):
    model = CoxPHFitter()
    model.fit(
        frame,
        duration_col=time_column,
        event_col=event_column,
        show_progress=False,
    )
    return model


def diagnostic_rows(
    output, endpoint, stratum, model_name, frame, time_column, event_column
):
    model_row = {
        "Manuscript output": output,
        "Endpoint": endpoint,
        "Stratum": stratum,
        "Model": model_name,
        "N": len(frame),
        "Events": int(frame[event_column].sum()),
        "Terms tested": len(frame.columns) - 2,
        "Minimum Schoenfeld p": np.nan,
        "Terms with p < 0.05": "",
        "PH screen": "Fit failed",
        "Error": "",
    }
    try:
        model = fit_model(frame, time_column, event_column)
        result = proportional_hazard_test(
            model,
            frame,
            time_transform="rank",
        ).summary.sort_values("p")
    except Exception as exc:
        model_row["Error"] = str(exc)
        return model_row, []

    minimum_p = float(result["p"].min())
    failed = result[result["p"] < PH_P_THRESHOLD]
    model_row["Minimum Schoenfeld p"] = minimum_p
    model_row["Terms with p < 0.05"] = "; ".join(
        f"{term} (p={row['p']:.3g})" for term, row in failed.iterrows()
    )
    model_row["PH screen"] = (
        "Potential violation" if len(failed) else "No violation detected"
    )

    term_rows = []
    for term, row in result.iterrows():
        term_rows.append(
            {
                "Manuscript output": output,
                "Endpoint": endpoint,
                "Stratum": stratum,
                "Model": model_name,
                "N": len(frame),
                "Events": int(frame[event_column].sum()),
                "Term": term,
                "Schoenfeld test statistic": float(row["test_statistic"]),
                "Schoenfeld p": float(row["p"]),
                "PH screen": (
                    "Potential violation"
                    if float(row["p"]) < PH_P_THRESHOLD
                    else "No violation detected"
                ),
            }
        )
    return model_row, term_rows


def add_diagnostic(
    model_rows,
    term_rows,
    output,
    endpoint,
    stratum,
    model_name,
    frame,
    time_column,
    event_column,
):
    model_row, new_term_rows = diagnostic_rows(
        output,
        endpoint,
        stratum,
        model_name,
        frame,
        time_column,
        event_column,
    )
    model_rows.append(model_row)
    term_rows.extend(new_term_rows)


def collect_primary_rows(model_rows, term_rows, module, data, endpoint, output):
    data = data[data["era"].isin(module.ERAS_INCLUDED)].copy()
    frame = module.prepare_cox_data(data)
    time_column, event_column = frame.columns[:2]
    covariates = [
        col for col in frame.columns if col not in [time_column, event_column]
    ]
    for covariate in covariates:
        add_diagnostic(
            model_rows,
            term_rows,
            output,
            endpoint,
            "Overall",
            f"Univariate: {covariate}",
            frame[[time_column, event_column, covariate]],
            time_column,
            event_column,
        )
    add_diagnostic(
        model_rows,
        term_rows,
        output,
        endpoint,
        "Overall",
        "Adjusted full model",
        frame,
        time_column,
        event_column,
    )


def collect_stage_rows(model_rows, term_rows, os_data, pcss_data):
    configs = [
        ("Overall Survival", os_data, "os_months", "os_event"),
        ("Prostate Cancer-Specific Survival", pcss_data, "pca_months", "pca_event"),
    ]
    for endpoint, data, time_column, event_column in configs:
        for stage in cox_by_stage_models.STAGES:
            subset = data[
                (data["m_stage"] == stage)
                & data["era"].isin(cox_by_stage_models.ERAS_INCLUDED)
            ].copy()
            frame = cox_by_stage_models.prepare_cox_data(
                subset,
                time_column,
                event_column,
            )
            add_diagnostic(
                model_rows,
                term_rows,
                "Supplementary Table 4",
                endpoint,
                stage,
                "Unadjusted era",
                frame[[time_column, event_column, "era4"]],
                time_column,
                event_column,
            )
            add_diagnostic(
                model_rows,
                term_rows,
                "Supplementary Table 4",
                endpoint,
                stage,
                "Adjusted full model",
                frame,
                time_column,
                event_column,
            )


def collect_race_rows(model_rows, term_rows, os_data, pcss_data):
    configs = [
        ("Overall Survival", os_data, "os_months", "os_event"),
        ("Prostate Cancer-Specific Survival", pcss_data, "pca_months", "pca_event"),
    ]
    for endpoint, data, time_column, event_column in configs:
        for race in cox_by_race_models.RACES:
            subset = data[
                (data["race"] == race)
                & data["era"].isin(cox_by_race_models.ERAS_INCLUDED)
            ].copy()
            frame = cox_by_race_models.prepare_cox_data(
                subset,
                time_column,
                event_column,
            )
            add_diagnostic(
                model_rows,
                term_rows,
                "Supplementary Table 6",
                endpoint,
                race,
                "Unadjusted era",
                frame[[time_column, event_column, "era4"]],
                time_column,
                event_column,
            )
            add_diagnostic(
                model_rows,
                term_rows,
                "Supplementary Table 6",
                endpoint,
                race,
                "Adjusted full model",
                frame,
                time_column,
                event_column,
            )


def collect_stage_race_rows(model_rows, term_rows, os_data, pcss_data):
    configs = [
        ("Overall Survival", os_data, "os_months", "os_event"),
        ("Prostate Cancer-Specific Survival", pcss_data, "pca_months", "pca_event"),
    ]
    for endpoint, data, time_column, event_column in configs:
        for race in cox_by_stage_and_race_models.RACES:
            for stage in cox_by_stage_and_race_models.STAGES:
                subset = data[
                    (data["race"] == race)
                    & (data["m_stage"] == stage)
                    & data["era"].isin(cox_by_stage_and_race_models.ERAS_INCLUDED)
                ].copy()
                frame = cox_by_stage_and_race_models.prepare_cox_data(
                    subset,
                    time_column,
                    event_column,
                )
                stratum = f"{race}; {stage}"
                add_diagnostic(
                    model_rows,
                    term_rows,
                    "Supplementary Table 8",
                    endpoint,
                    stratum,
                    "Unadjusted era",
                    frame[[time_column, event_column, "era4"]],
                    time_column,
                    event_column,
                )
                add_diagnostic(
                    model_rows,
                    term_rows,
                    "Supplementary Table 8",
                    endpoint,
                    stratum,
                    "Adjusted full model",
                    frame,
                    time_column,
                    event_column,
                )


def collect_rows():
    os_data = overall_survival_data.load_and_prepare_data()
    pcss_data = prostate_cancer_specific_survival_data.load_and_prepare_data()
    model_rows = []
    term_rows = []

    collect_primary_rows(
        model_rows,
        term_rows,
        cox_os_models,
        os_data,
        "Overall Survival",
        "Table 3",
    )
    collect_primary_rows(
        model_rows,
        term_rows,
        cox_pcss_models,
        pcss_data,
        "Prostate Cancer-Specific Survival",
        "Supplementary Table 3",
    )
    collect_stage_rows(model_rows, term_rows, os_data, pcss_data)
    collect_race_rows(model_rows, term_rows, os_data, pcss_data)
    collect_stage_race_rows(model_rows, term_rows, os_data, pcss_data)
    return pd.DataFrame(model_rows), pd.DataFrame(term_rows)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    model_table, term_table = collect_rows()
    model_path = TABLE_DIR / "20_cox_ph_diagnostics.csv"
    term_path = TABLE_DIR / "20_cox_ph_terms.csv"
    model_table.to_csv(model_path, index=False)
    term_table.to_csv(term_path, index=False)


if __name__ == "__main__":
    main()

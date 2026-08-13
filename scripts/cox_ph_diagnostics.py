
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
OUTPUT_DIR = PROJECT_ROOT / "output" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05


def fit_model(frame, time_col, event_col):
    model = CoxPHFitter()
    model.fit(
        frame,
        duration_col=time_col,
        event_col=event_col,
        show_progress=False,
    )
    return model


def diagnostic_rows(output, endpoint, stratum, model_name, frame, time_col, event_col):
    model_row = {
        "Manuscript output": output,
        "Endpoint": endpoint,
        "Stratum": stratum,
        "Model": model_name,
        "N": len(frame),
        "Events": int(frame[event_col].sum()),
        "Terms tested": len(frame.columns) - 2,
        "Minimum Schoenfeld p": np.nan,
        "Terms with p < 0.05": "",
        "PH screen": "Fit failed",
        "Error": "",
    }
    try:
        model = fit_model(frame, time_col, event_col)
        result = proportional_hazard_test(
            model,
            frame,
            time_transform="rank",
        ).summary.sort_values("p")
    except Exception as exc:
        model_row["Error"] = str(exc)
        return model_row, []

    minimum_p = float(result["p"].min())
    failed = result[result["p"] < ALPHA]
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
                "Events": int(frame[event_col].sum()),
                "Term": term,
                "Schoenfeld test statistic": float(row["test_statistic"]),
                "Schoenfeld p": float(row["p"]),
                "PH screen": (
                    "Potential violation"
                    if float(row["p"]) < ALPHA
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
    time_col,
    event_col,
):
    model_row, new_term_rows = diagnostic_rows(
        output,
        endpoint,
        stratum,
        model_name,
        frame,
        time_col,
        event_col,
    )
    model_rows.append(model_row)
    term_rows.extend(new_term_rows)


def collect_primary_rows(model_rows, term_rows, module, df, endpoint, output):
    df = df[df["era"].isin(module.ERAS_INCLUDED)].copy()
    frame = module.prepare_cox_df(df)
    time_col, event_col = frame.columns[:2]
    covariates = [col for col in frame.columns if col not in [time_col, event_col]]
    for covariate in covariates:
        add_diagnostic(
            model_rows,
            term_rows,
            output,
            endpoint,
            "Overall",
            f"Univariate: {covariate}",
            frame[[time_col, event_col, covariate]],
            time_col,
            event_col,
        )
    add_diagnostic(
        model_rows,
        term_rows,
        output,
        endpoint,
        "Overall",
        "Adjusted full model",
        frame,
        time_col,
        event_col,
    )


def collect_stage_rows(model_rows, term_rows, df_os, df_pca):
    configs = [
        ("Overall Survival", df_os, "os_months", "os_event"),
        ("Prostate Cancer-Specific Survival", df_pca, "pca_months", "pca_event"),
    ]
    for endpoint, df, time_col, event_col in configs:
        for stage in cox_by_stage_models.STAGES:
            subset = df[
                (df["m_stage"] == stage)
                & df["era"].isin(cox_by_stage_models.ERAS_INCLUDED)
            ].copy()
            frame = cox_by_stage_models.prepare_cox_df(
                subset,
                time_col,
                event_col,
            )
            add_diagnostic(
                model_rows,
                term_rows,
                "Supplementary Table 4",
                endpoint,
                stage,
                "Unadjusted era",
                frame[[time_col, event_col, "era4"]],
                time_col,
                event_col,
            )
            add_diagnostic(
                model_rows,
                term_rows,
                "Supplementary Table 4",
                endpoint,
                stage,
                "Adjusted full model",
                frame,
                time_col,
                event_col,
            )


def collect_race_rows(model_rows, term_rows, df_os, df_pca):
    configs = [
        ("Overall Survival", df_os, "os_months", "os_event"),
        ("Prostate Cancer-Specific Survival", df_pca, "pca_months", "pca_event"),
    ]
    for endpoint, df, time_col, event_col in configs:
        for race in cox_by_race_models.RACES:
            subset = df[
                (df["race"] == race)
                & df["era"].isin(cox_by_race_models.ERAS_INCLUDED)
            ].copy()
            frame = cox_by_race_models.prepare_cox_df(
                subset,
                time_col,
                event_col,
            )
            add_diagnostic(
                model_rows,
                term_rows,
                "Supplementary Table 6",
                endpoint,
                race,
                "Unadjusted era",
                frame[[time_col, event_col, "era4"]],
                time_col,
                event_col,
            )
            add_diagnostic(
                model_rows,
                term_rows,
                "Supplementary Table 6",
                endpoint,
                race,
                "Adjusted full model",
                frame,
                time_col,
                event_col,
            )


def collect_stage_race_rows(model_rows, term_rows, df_os, df_pca):
    configs = [
        ("Overall Survival", df_os, "os_months", "os_event"),
        ("Prostate Cancer-Specific Survival", df_pca, "pca_months", "pca_event"),
    ]
    for endpoint, df, time_col, event_col in configs:
        for race in cox_by_stage_and_race_models.RACES:
            for stage in cox_by_stage_and_race_models.STAGES:
                subset = df[
                    (df["race"] == race)
                    & (df["m_stage"] == stage)
                    & df["era"].isin(
                        cox_by_stage_and_race_models.ERAS_INCLUDED
                    )
                ].copy()
                frame = cox_by_stage_and_race_models.prepare_cox_df(
                    subset,
                    time_col,
                    event_col,
                )
                stratum = f"{race}; {stage}"
                add_diagnostic(
                    model_rows,
                    term_rows,
                    "Supplementary Table 8",
                    endpoint,
                    stratum,
                    "Unadjusted era",
                    frame[[time_col, event_col, "era4"]],
                    time_col,
                    event_col,
                )
                add_diagnostic(
                    model_rows,
                    term_rows,
                    "Supplementary Table 8",
                    endpoint,
                    stratum,
                    "Adjusted full model",
                    frame,
                    time_col,
                    event_col,
                )


def collect_rows():
    df_os = overall_survival_data.load_and_prepare_data()
    df_pca = prostate_cancer_specific_survival_data.load_and_prepare_data()
    model_rows = []
    term_rows = []

    collect_primary_rows(
        model_rows,
        term_rows,
        cox_os_models,
        df_os,
        "Overall Survival",
        "Table 3",
    )
    collect_primary_rows(
        model_rows,
        term_rows,
        cox_pcss_models,
        df_pca,
        "Prostate Cancer-Specific Survival",
        "Supplementary Table 3",
    )
    collect_stage_rows(model_rows, term_rows, df_os, df_pca)
    collect_race_rows(model_rows, term_rows, df_os, df_pca)
    collect_stage_race_rows(model_rows, term_rows, df_os, df_pca)
    return pd.DataFrame(model_rows), pd.DataFrame(term_rows)


def main():
    model_table, term_table = collect_rows()
    model_path = OUTPUT_DIR / "20_cox_ph_diagnostics.csv"
    term_path = OUTPUT_DIR / "20_cox_ph_terms.csv"
    model_table.to_csv(model_path, index=False)
    term_table.to_csv(term_path, index=False)


if __name__ == "__main__":
    main()

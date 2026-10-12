import re
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.statistics import proportional_hazard_test

import cox_by_race_models as cox_race
import cox_by_stage_and_race_models as cox_stage_race
import cox_by_stage_models as cox_stage
import cox_os_models as cox_os
import cox_pcss_models as cox_pcss
import overall_survival_data
import prostate_cancer_specific_survival_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
PH_P_THRESHOLD = 0.05
REFERENCE_MONTH = 6.0
TIME_FUNCTIONS = ["log", "linear", "sqrt"]
Z_975 = 1.959963984540054


def time_function(name, time):
    values = np.asarray(time, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Follow-up time must be positive")
    if name == "log":
        return np.log(values / REFERENCE_MONTH)
    if name == "linear":
        return (values - REFERENCE_MONTH) / REFERENCE_MONTH
    if name == "sqrt":
        return np.sqrt(values) - sqrt(REFERENCE_MONTH)
    raise ValueError(f"Unknown time function: {name}")


def fit_cox_model(frame, time_column, event_column):
    model = CoxPHFitter()
    model.fit(
        frame,
        duration_col=time_column,
        event_col=event_column,
        show_progress=False,
    )
    return model


def test_proportional_hazards(model, frame):
    return proportional_hazard_test(
        model,
        frame,
        time_transform="rank",
    ).summary.sort_values("p")


def to_counting_process(
    frame,
    time_column,
    event_column,
    flagged_terms,
    function_name,
):
    base = frame.reset_index(drop=True).copy()
    base.insert(0, "id", np.arange(len(base), dtype=int))
    durations = base[time_column].astype(int).to_numpy()
    if not np.allclose(durations, base[time_column].to_numpy()):
        raise ValueError("Expected integer SEER survival months")
    if np.any(durations <= 0):
        raise ValueError("Expected strictly positive follow-up")

    long = base.loc[base.index.repeat(durations)].copy()
    long["stop"] = long.groupby("id").cumcount() + 1
    long["start"] = long["stop"] - 1
    long["event_long"] = (
        (long[event_column] == 1) & (long["stop"] == long[time_column])
    ).astype(int)

    transform = time_function(function_name, long["stop"].to_numpy())
    for term in flagged_terms:
        long[f"tv__{term}"] = long[term].to_numpy() * transform

    return long.drop(columns=[time_column, event_column])


def fit_time_varying_cox_model(
    frame,
    time_column,
    event_column,
    flagged_terms,
    function_name,
):
    long = to_counting_process(
        frame,
        time_column,
        event_column,
        flagged_terms,
        function_name,
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


def target_estimate(model, target):
    row = model.summary.loc[target]
    coef = float(row["coef"])
    se = float(row["se(coef)"])
    return {
        "HR": float(np.exp(coef)),
        "CI lower": float(np.exp(coef - Z_975 * se)),
        "CI upper": float(np.exp(coef + Z_975 * se)),
        "p": float(row["p"]),
    }


def pretty_term(term):
    replacements = {
        "age": "Age",
        "prostatectomy": "Prostatectomy",
        "radiotherapy": "Radiotherapy",
    }
    if term in replacements:
        return replacements[term]
    if term.startswith("race_"):
        return term.replace("race_", "Race: ", 1)
    if term.startswith("m_stage_"):
        return term.replace("m_stage_", "Metastatic classification: ", 1)
    return term


def format_hazard_ratio(estimate):
    return (
        f"{estimate['HR']:.4f} "
        f"({estimate['CI lower']:.4f}-{estimate['CI upper']:.4f})"
    )


def format_p_value(p):
    return "< 0.001" if p < 0.001 else f"{p:.4f}"


def primary_frame(module, data):
    subset = data[data["era"].isin(module.ERAS_INCLUDED)].copy()
    frame = module.prepare_cox_data(subset)
    time_column, event_column = frame.columns[:2]
    target = f"era_{module.ERA_LABELS[4]}"
    return subset, frame, time_column, event_column, target


def collect_model_specs(os_data, pcss_data):
    specs = []
    for key, output, endpoint, module, data in [
        (
            "primary_os",
            "Table 3",
            "Overall Survival",
            cox_os,
            os_data,
        ),
        (
            "primary_pcss",
            "Supplementary Table 3",
            "Prostate Cancer-Specific Survival",
            cox_pcss,
            pcss_data,
        ),
    ]:
        _, frame, time_column, event_column, target = primary_frame(module, data)
        specs.append(
            {
                "key": key,
                "output": output,
                "endpoint": endpoint,
                "stratum": "Overall",
                "frame": frame,
                "time_column": time_column,
                "event_column": event_column,
                "target": target,
            }
        )

    endpoint_configs = [
        ("Overall Survival", os_data, "os_months", "os_event"),
        (
            "Prostate Cancer-Specific Survival",
            pcss_data,
            "pca_months",
            "pca_event",
        ),
    ]
    for endpoint, data, time_column, event_column in endpoint_configs:
        for stage in cox_stage.STAGES:
            subset = data[
                (data["m_stage"] == stage) & data["era"].isin(cox_stage.ERAS_INCLUDED)
            ].copy()
            specs.append(
                {
                    "key": f"stage|{endpoint}|{stage}",
                    "output": "Supplementary Table 4",
                    "endpoint": endpoint,
                    "stratum": stage,
                    "frame": cox_stage.prepare_cox_data(
                        subset,
                        time_column,
                        event_column,
                    ),
                    "time_column": time_column,
                    "event_column": event_column,
                    "target": "era4",
                }
            )

        for race in cox_race.RACES:
            subset = data[
                (data["race"] == race) & data["era"].isin(cox_race.ERAS_INCLUDED)
            ].copy()
            specs.append(
                {
                    "key": f"race|{endpoint}|{race}",
                    "output": "Supplementary Table 6",
                    "endpoint": endpoint,
                    "stratum": race,
                    "frame": cox_race.prepare_cox_data(
                        subset,
                        time_column,
                        event_column,
                    ),
                    "time_column": time_column,
                    "event_column": event_column,
                    "target": "era4",
                }
            )

        for race in cox_stage_race.RACES:
            for stage in cox_stage_race.STAGES:
                subset = data[
                    (data["race"] == race)
                    & (data["m_stage"] == stage)
                    & data["era"].isin(cox_stage_race.ERAS_INCLUDED)
                ].copy()
                frame = cox_stage_race.prepare_cox_data(
                    subset,
                    time_column,
                    event_column,
                )
                specs.append(
                    {
                        "key": f"stage_race|{endpoint}|{race}|{stage}",
                        "output": "Supplementary Table 8",
                        "endpoint": endpoint,
                        "stratum": f"{race}; {stage}",
                        "frame": frame,
                        "time_column": time_column,
                        "event_column": event_column,
                        "target": "era4",
                        "omitted_terms": [
                            term
                            for term in ["prostatectomy", "radiotherapy"]
                            if term not in frame.columns
                        ],
                    }
                )
    return specs


def fit_screened_model(spec, function_name, cached=None):
    if cached is None:
        conventional = fit_cox_model(
            spec["frame"],
            spec["time_column"],
            spec["event_column"],
        )
        screen = test_proportional_hazards(conventional, spec["frame"])
    else:
        conventional = cached["conventional"]
        screen = cached["screen"]

    target_ph_p = float(screen.loc[spec["target"], "p"])
    if target_ph_p < PH_P_THRESHOLD:
        raise RuntimeError(
            f"{spec['key']}: displayed era term failed PH screening "
            f"(p={target_ph_p:.6g})"
        )
    flagged = [
        term
        for term, row in screen.iterrows()
        if float(row["p"]) < PH_P_THRESHOLD and term != spec["target"]
    ]
    if flagged:
        if cached is not None and cached.get("model") is not None:
            model = cached["model"]
        else:
            model = fit_time_varying_cox_model(
                spec["frame"],
                spec["time_column"],
                spec["event_column"],
                flagged,
                function_name,
            )
        model_type = "Extended Cox"
    else:
        model = conventional
        model_type = "Standard Cox"
    return {
        "conventional": conventional,
        "screen": screen,
        "flagged": flagged,
        "model": model,
        "model_type": model_type,
        "estimate": target_estimate(model, spec["target"]),
        "target_ph_p": target_ph_p,
    }


def select_time_function(primary_specs):
    comparison_rows = []
    cached = {}
    for spec in primary_specs:
        conventional = fit_cox_model(
            spec["frame"],
            spec["time_column"],
            spec["event_column"],
        )
        screen = test_proportional_hazards(conventional, spec["frame"])
        flagged = [
            term
            for term, row in screen.iterrows()
            if float(row["p"]) < PH_P_THRESHOLD and term != spec["target"]
        ]
        if float(screen.loc[spec["target"], "p"]) < PH_P_THRESHOLD:
            raise RuntimeError(f"{spec['key']}: displayed era term failed PH screening")
        for function_name in TIME_FUNCTIONS:
            model = fit_time_varying_cox_model(
                spec["frame"],
                spec["time_column"],
                spec["event_column"],
                flagged,
                function_name,
            )
            aic = float(model.AIC_partial_)
            comparison_rows.append(
                {
                    "Model": spec["key"],
                    "Endpoint": spec["endpoint"],
                    "Time function": function_name,
                    "Partial AIC": aic,
                    "Time-varying terms": "; ".join(
                        pretty_term(term) for term in flagged
                    ),
                }
            )
            cached[(spec["key"], function_name)] = {
                "conventional": conventional,
                "screen": screen,
                "model": model,
            }

    comparison = pd.DataFrame(comparison_rows)
    totals = (
        comparison.groupby("Time function", as_index=False)["Partial AIC"]
        .sum()
        .rename(columns={"Partial AIC": "Summed partial AIC"})
    )
    selected = totals.sort_values("Summed partial AIC").iloc[0]["Time function"]
    comparison = comparison.merge(totals, on="Time function", how="left")
    comparison["Selected common function"] = comparison["Time function"].eq(selected)
    return selected, comparison, cached


def coefficient_rows(spec, result, function_name):
    rows = []
    summary = result["model"].summary
    for term in result["flagged"]:
        tv_term = f"tv__{term}"
        row = summary.loc[term]
        tv_row = summary.loc[tv_term]
        item = {
            "Corresponding result": spec["output"],
            "Endpoint": spec["endpoint"],
            "Stratum": spec["stratum"],
            "Term": pretty_term(term),
            "HR at 6 months": float(np.exp(row["coef"])),
            "95% CI lower at 6 months": float(
                np.exp(row["coef"] - Z_975 * row["se(coef)"])
            ),
            "95% CI upper at 6 months": float(
                np.exp(row["coef"] + Z_975 * row["se(coef)"])
            ),
            "p at 6 months": float(row["p"]),
            "Time interaction coefficient": float(tv_row["coef"]),
            "Time interaction p": float(tv_row["p"]),
            "Time function": function_name,
            "HR multiplier per doubling of follow-up": np.nan,
        }
        if function_name == "log":
            item["HR multiplier per doubling of follow-up"] = float(
                np.exp(tv_row["coef"] * np.log(2.0))
            )
        rows.append(item)
    return rows


def configuration_row(spec, result, function_name):
    failed = [
        term
        for term, row in result["screen"].iterrows()
        if float(row["p"]) < PH_P_THRESHOLD
    ]
    return {
        "Corresponding result": spec["output"],
        "Endpoint": spec["endpoint"],
        "Stratum": spec["stratum"],
        "N": len(spec["frame"]),
        "Events": int(spec["frame"][spec["event_column"]].sum()),
        "Era-term Schoenfeld p": result["target_ph_p"],
        "Final model": result["model_type"],
        "Time function": (
            function_name if result["model_type"] == "Extended Cox" else ""
        ),
        "Time-varying adjustment terms": "; ".join(
            pretty_term(term) for term in result["flagged"]
        ),
        "All terms flagged by initial PH screen": "; ".join(
            pretty_term(term) for term in failed
        ),
        "Omitted nonestimable adjustment terms": "; ".join(
            pretty_term(term) for term in spec.get("omitted_terms", [])
        ),
    }


def estimate_row(spec, result):
    estimate = result["estimate"]
    return {
        "Corresponding result": spec["output"],
        "Endpoint": spec["endpoint"],
        "Stratum": spec["stratum"],
        "Adjusted era HR": estimate["HR"],
        "95% CI lower": estimate["CI lower"],
        "95% CI upper": estimate["CI upper"],
        "p": estimate["p"],
        "Era-term Schoenfeld p": result["target_ph_p"],
        "Final model": result["model_type"],
        "Time-varying adjustment terms": "; ".join(
            pretty_term(term) for term in result["flagged"]
        ),
        "Omitted nonestimable adjustment terms": "; ".join(
            pretty_term(term) for term in spec.get("omitted_terms", [])
        ),
    }


def replace_target(table, mask, estimate):
    if int(mask.sum()) != 1:
        raise RuntimeError(f"Expected one table row, found {int(mask.sum())}")
    table.loc[mask, "Adjusted HR (95% CI)"] = format_hazard_ratio(estimate)
    table.loc[mask, "Adj p-value"] = format_p_value(estimate["p"])


def primary_term_row(table, term, module):
    if term == f"era_{module.ERA_LABELS[4]}":
        return table["Variable"].eq(f"  {module.ERA_LABELS[4]}")
    if term == "age":
        return table["Variable"].eq("Age (continuous, per year)")
    if term.startswith("race_"):
        return table["Variable"].eq(f"  {term.replace('race_', '', 1)}")
    if term.startswith("m_stage_"):
        return table["Variable"].eq(f"  {term.replace('m_stage_', '', 1)}")
    if term == "prostatectomy":
        section = table.index[table["Variable"].eq("Radical Prostatectomy")][0]
        mask = pd.Series(False, index=table.index)
        mask.loc[section + 2] = True
        return mask
    if term == "radiotherapy":
        section = table.index[table["Variable"].eq("Radiotherapy")][0]
        mask = pd.Series(False, index=table.index)
        mask.loc[section + 2] = True
        return mask
    raise KeyError(f"No primary-table row for {term}")


def replace_primary_adjusted_estimates(table, result, module):
    for term in result["model"].params_.index:
        if term.startswith("tv__"):
            continue
        replace_target(
            table,
            primary_term_row(table, term, module),
            target_estimate(result["model"], term),
        )
    return table.rename(
        columns={
            "Adjusted HR (95% CI)": "Adjusted HR at 6 months (95% CI)",
            "Adj p-value": "Adj p-value at 6 months",
        }
    )


def format_inline_p_value(value):
    value = str(value).strip().replace(" ", "")
    if value.startswith("<"):
        return f"p{value}"
    try:
        value = f"{float(value):.3f}"
    except ValueError:
        pass
    return f"p={value}"


def compact_hr(estimate):
    estimate = str(estimate)
    match = re.fullmatch(
        r"([0-9.]+) \(([0-9.]+)-([0-9.]+)\)",
        estimate,
    )
    if not match:
        return estimate
    point, lower, upper = (float(value) for value in match.groups())
    return f"{point:.2f} ({lower:.2f}-{upper:.2f})"


def estimate_with_p(estimate, p_value):
    if pd.isna(estimate) or not str(estimate).strip():
        return ""
    if pd.isna(p_value) or not str(p_value).strip():
        return str(estimate)
    return f"{compact_hr(estimate)}; {format_inline_p_value(p_value)}"


def compact_primary_supplement(table):
    compact = table[["Variable", "Sample size (%)"]].copy()
    compact["6-month aHR (95% CI); p"] = table.apply(
        lambda row: estimate_with_p(
            row["Adjusted HR at 6 months (95% CI)"],
            row["Adj p-value at 6 months"],
        ),
        axis=1,
    )
    return compact.rename(columns={"Sample size (%)": "n (%)"})


def compact_stratified_table(table, include_race=False, include_stage=False):
    endpoint_labels = {
        "Overall Survival": "OS",
        "Prostate Cancer-Specific Survival": "PCSS",
    }
    race_labels = {
        "Non-Hispanic White": "NHW",
        "Non-Hispanic Black": "NHB",
        "Non-Hispanic Asian/PI": "NHA/PI",
    }

    def analysis_label(row):
        parts = [endpoint_labels.get(row["Survival"], row["Survival"])]
        if include_race:
            parts.append(race_labels.get(row["Race"], row["Race"]))
        if include_stage:
            parts.append(row["Stage"])
        return "/".join(parts)

    era_3_label = cox_stage.ERA_LABELS[3]
    era_4_label = cox_stage.ERA_LABELS[4]
    era_3_col = next(
        column for column in table if column.startswith(f"n {era_3_label}")
    )
    era_4_col = next(
        column for column in table if column.startswith(f"n {era_4_label}")
    )
    era_3_short = era_3_label.rsplit(" (", 1)[0]
    era_4_short = era_4_label.rsplit(" (", 1)[0]
    compact = pd.DataFrame(
        {
            "Analysis": table.apply(analysis_label, axis=1),
            f"n {era_3_short}": table[era_3_col],
            f"n {era_4_short}": table[era_4_col],
            "HR (95% CI); p": table.apply(
                lambda row: estimate_with_p(row["HR (95% CI)"], row["p-value"]),
                axis=1,
            ),
            "aHR (95% CI); p": table.apply(
                lambda row: estimate_with_p(
                    row["Adjusted HR (95% CI)"],
                    row["Adj p-value"],
                ),
                axis=1,
            ),
        }
    )
    return compact


def build_updated_tables(os_data, pcss_data, result_by_key):
    os_subset, os_frame, _, _, _ = primary_frame(cox_os, os_data)
    os_adjusted_results = cox_os.fit_multivariable(os_frame)
    os_table = cox_os.build_table(os_subset, os_frame, os_adjusted_results)
    os_table = replace_primary_adjusted_estimates(
        os_table,
        result_by_key["primary_os"],
        cox_os,
    )

    pcss_subset, pcss_frame, _, _, _ = primary_frame(cox_pcss, pcss_data)
    pcss_adjusted_results = cox_pcss.fit_multivariable(pcss_frame)
    pcss_table = cox_pcss.build_table(pcss_subset, pcss_frame, pcss_adjusted_results)
    pcss_table = replace_primary_adjusted_estimates(
        pcss_table,
        result_by_key["primary_pcss"],
        cox_pcss,
    )

    stage_table = cox_stage.build_table(os_data, pcss_data)
    race_table = cox_race.build_table(os_data, pcss_data)
    stage_race_table = cox_stage_race.build_table(os_data, pcss_data)

    for key, result in result_by_key.items():
        parts = key.split("|")
        if parts[0] == "stage":
            _, endpoint, stage = parts
            replace_target(
                stage_table,
                stage_table["Survival"].eq(endpoint) & stage_table["Stage"].eq(stage),
                result["estimate"],
            )
        elif parts[0] == "race":
            _, endpoint, race = parts
            replace_target(
                race_table,
                race_table["Survival"].eq(endpoint) & race_table["Race"].eq(race),
                result["estimate"],
            )
        elif parts[0] == "stage_race":
            _, endpoint, race, stage = parts
            replace_target(
                stage_race_table,
                stage_race_table["Survival"].eq(endpoint)
                & stage_race_table["Race"].eq(race)
                & stage_race_table["Stage"].eq(stage),
                result["estimate"],
            )

    os_table = compact_primary_supplement(os_table)
    pcss_table = compact_primary_supplement(pcss_table)
    stage_table = compact_stratified_table(stage_table, include_stage=True)
    race_table = compact_stratified_table(race_table, include_race=True)
    stage_race_table = compact_stratified_table(
        stage_race_table,
        include_race=True,
        include_stage=True,
    )

    outputs = [
        ("14_cox_os.csv", os_table),
        ("15_cox_pcss.csv", pcss_table),
        ("16_cox_by_stage.csv", stage_table),
        ("17_cox_by_race.csv", race_table),
        ("19_cox_by_stage_race.csv", stage_race_table),
    ]
    for filename, table in outputs:
        table.to_csv(TABLE_DIR / filename, index=False)


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    os_data = overall_survival_data.load_and_prepare_data()
    pcss_data = prostate_cancer_specific_survival_data.load_and_prepare_data()
    specs = collect_model_specs(os_data, pcss_data)
    primary_specs = [spec for spec in specs if spec["key"].startswith("primary_")]
    selected, comparison, cached = select_time_function(primary_specs)

    configuration_rows = []
    estimates = []
    coefficients = []
    result_by_key = {}
    for spec in specs:
        cached_fit = cached.get((spec["key"], selected))
        result = fit_screened_model(spec, selected, cached=cached_fit)
        result_by_key[spec["key"]] = result
        configuration_rows.append(configuration_row(spec, result, selected))
        estimates.append(estimate_row(spec, result))
        coefficients.extend(coefficient_rows(spec, result, selected))

    comparison.to_csv(
        TABLE_DIR / "21_ph_time_function_comparison.csv",
        index=False,
    )
    pd.DataFrame(configuration_rows).to_csv(
        TABLE_DIR / "21_ph_aware_model_configurations.csv",
        index=False,
    )
    pd.DataFrame(estimates).to_csv(
        TABLE_DIR / "21_ph_aware_era_estimates.csv",
        index=False,
    )
    pd.DataFrame(coefficients).to_csv(
        TABLE_DIR / "21_ph_aware_coefficients.csv",
        index=False,
    )
    build_updated_tables(os_data, pcss_data, result_by_key)


if __name__ == "__main__":
    main()

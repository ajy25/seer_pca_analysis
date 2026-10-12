import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from scipy.stats import chi2

import era_race_interaction
import grouped_efron_cox as grouped
import overall_survival_data
import prostate_cancer_specific_survival_data
from export_manuscript_figures import export_figure3
from figure3_plot import plot_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "output" / "tables"
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
RACES = ["Non-Hispanic White", "Non-Hispanic Black"]
ERAS = ["Pre-Docetaxel", "Post-Docetaxel", "Early ARPI", "Modern ARPI"]
YEARS = ["2000–2003", "2004–2009", "2010–2016", "2017–2020"]
ERA_CODES = [1, 2, 3, 4]
RACE_ERA_GROUPS = [(era, black) for black in [0, 1] for era in ERA_CODES]
ERA_TERMS = ["era2", "era3", "era4"]
INTERACTION_TERMS = ["era2_x_black", "era3_x_black", "era4_x_black"]
BASE_TERMS = [
    *ERA_TERMS,
    "black",
    "age_per10",
    "prostatectomy",
    "radiotherapy",
    *INTERACTION_TERMS,
]
COVARIATE_COLUMNS = ["era", "black", "age", "prostatectomy", "radiotherapy"]
ANALYSIS_MONTHS = 24
PH_P_THRESHOLD = 0.05
BOOTSTRAP_SEED = 20260923
N_BOOTSTRAP = 500
ENDPOINTS = [
    ("OS", overall_survival_data, "os_months", "os_event"),
    ("PCSS", prostate_cancer_specific_survival_data, "pca_months", "pca_event"),
]


def build_design_matrix(covariates):
    era_indicators = np.column_stack(
        [(covariates["era"].to_numpy() == era).astype(float) for era in ERA_CODES[1:]]
    )
    black_indicator = covariates["black"].to_numpy(dtype=float)
    age_per_decade = (covariates["age"].to_numpy() - 70) / 10
    race_era_interactions = era_indicators * black_indicator[:, None]
    return np.column_stack(
        [
            era_indicators,
            black_indicator,
            age_per_decade,
            covariates["prostatectomy"],
            covariates["radiotherapy"],
            race_era_interactions,
        ]
    ).astype(float)


def prepare_model_data(data, time_column, event_column):
    patients = data[data["race"].isin(RACES)].copy()
    patients["black"] = (patients["race"] == RACES[1]).astype(int)
    patients["event"] = (
        (patients[event_column] == 1) & (patients[time_column] <= ANALYSIS_MONTHS)
    ).astype(int)
    patients["duration"] = np.minimum(patients[time_column], ANALYSIS_MONTHS)
    patients = patients[COVARIATE_COLUMNS + ["duration", "event"]].dropna()

    durations = patients["duration"]
    if (durations <= 0).any() or not np.allclose(durations, durations.astype(int)):
        raise ValueError("Expected positive integer SEER survival months")

    covariate_keys = pd.MultiIndex.from_frame(patients[COVARIATE_COLUMNS])
    group_ids, unique_covariates = pd.factorize(covariate_keys, sort=True)
    covariates = unique_covariates.to_frame(index=False)
    covariates.columns = COVARIATE_COLUMNS
    counts = np.zeros((len(covariates), ANALYSIS_MONTHS, 2), dtype=np.int64)
    count_indices = (
        group_ids,
        durations.to_numpy(dtype=int) - 1,
        patients["event"].to_numpy(dtype=int),
    )
    np.add.at(counts, count_indices, 1)

    frame = pd.DataFrame(build_design_matrix(patients), columns=BASE_TERMS)
    frame["duration"] = durations.to_numpy()
    frame["event"] = patients["event"].to_numpy()
    return patients, covariates, counts, frame


def select_time_varying_terms(frame):
    full_model = CoxPHFitter().fit(frame, "duration", "event")
    reduced_frame = frame.drop(columns=INTERACTION_TERMS)
    reduced_model = CoxPHFitter().fit(reduced_frame, "duration", "event")
    screening_rows = []

    for label, model, model_frame in [
        ("Full", full_model, frame),
        ("Reduced", reduced_model, reduced_frame),
    ]:
        screening = proportional_hazard_test(
            model,
            model_frame,
            time_transform="rank",
        ).summary
        for term, row in screening.iterrows():
            screening_rows.append(
                {
                    "Model": label,
                    "Term": term,
                    "Schoenfeld p": float(row["p"]),
                }
            )

    flagged_terms = {
        row["Term"] for row in screening_rows if row["Schoenfeld p"] < PH_P_THRESHOLD
    }
    if flagged_terms.intersection(ERA_TERMS):
        flagged_terms.update(ERA_TERMS)
    if flagged_terms.intersection(INTERACTION_TERMS):
        flagged_terms.update(INTERACTION_TERMS + ERA_TERMS + ["black"])
    time_terms = [term for term in BASE_TERMS if term in flagged_terms]
    return time_terms, screening_rows


def build_counterfactuals(covariates, time_terms):
    time_indices = [BASE_TERMS.index(term) for term in time_terms]
    counterfactual_designs = []
    for era, black in RACE_ERA_GROUPS:
        counterfactual = covariates.copy()
        counterfactual["era"] = era
        counterfactual["black"] = black
        counterfactual_designs.append(
            grouped.make_designs(build_design_matrix(counterfactual), time_indices)
        )
    return counterfactual_designs


@dataclass
class BootstrapContext:
    counts: np.ndarray
    designs: np.ndarray
    counterfactuals: list
    initial_coefficients: np.ndarray
    sampling_groups: list


_BOOTSTRAP_CONTEXT = None


def initialize_bootstrap(covariates, counts, designs, counterfactuals, coefficients):
    global _BOOTSTRAP_CONTEXT
    sampling_groups = []
    for era, black in RACE_ERA_GROUPS:
        group_ids = np.flatnonzero(
            (covariates["era"] == era) & (covariates["black"] == black)
        )
        cells = counts[group_ids].ravel()
        sample_size = int(cells.sum())
        sampling_groups.append((group_ids, sample_size, cells / sample_size))
    _BOOTSTRAP_CONTEXT = BootstrapContext(
        counts,
        designs,
        counterfactuals,
        coefficients,
        sampling_groups,
    )


def bootstrap_one(seed):
    context = _BOOTSTRAP_CONTEXT
    if context is None:
        raise RuntimeError("Bootstrap worker has not been initialized")
    generator = np.random.default_rng(seed)
    sampled_counts = np.zeros_like(context.counts)
    for group_ids, sample_size, probabilities in context.sampling_groups:
        sampled_counts[group_ids] = generator.multinomial(
            sample_size,
            probabilities,
        ).reshape(len(group_ids), ANALYSIS_MONTHS, 2)
    try:
        model = grouped.fit(
            context.designs,
            sampled_counts,
            initial=context.initial_coefficients,
        )
        estimates = grouped.standardize(
            model,
            context.counterfactuals,
            sampled_counts.sum(axis=(1, 2)),
        )
        return {"seed": int(seed), "values": estimates.tolist(), "error": None}
    except Exception as error:
        return {"seed": int(seed), "values": None, "error": str(error)}


def run_bootstrap(
    covariates, counts, designs, counterfactuals, model, n_bootstrap, workers, endpoint
):
    seeds = np.random.SeedSequence(BOOTSTRAP_SEED).generate_state(n_bootstrap)
    successful = []
    failures = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_bootstrap,
        initargs=(covariates, counts, designs, counterfactuals, model.coefficients),
    ) as pool:
        futures = [pool.submit(bootstrap_one, int(seed)) for seed in seeds]
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["error"]:
                failures.append(result)
            else:
                successful.append(result)
            if completed % 25 == 0:
                print(endpoint, f"bootstrap {completed}/{n_bootstrap}", flush=True)
    successful.sort(key=lambda result: result["seed"])
    if len(successful) < n_bootstrap * 0.95:
        raise RuntimeError(
            f"{endpoint}: insufficient bootstrap convergence: {failures}"
        )
    return successful, failures


def validate_against_lifelines(frame, time_terms, model, counts, counterfactuals):
    started = time.monotonic()
    reference = era_race_interaction.fit_time_varying_cox_model(
        frame,
        "duration",
        "event",
        time_terms,
    )
    terms = BASE_TERMS + ["tv__" + term for term in time_terms]
    reference_coefficients = reference.params_.loc[terms].to_numpy()
    coefficient_error = float(
        np.max(np.abs(model.coefficients - reference_coefficients))
    )
    likelihood_error = abs(model.log_likelihood - float(reference.log_likelihood_))

    cumulative_baseline = reference.baseline_cumulative_hazard_.iloc[:, 0]
    baseline_increments = cumulative_baseline.diff()
    baseline_increments.iloc[0] = cumulative_baseline.iloc[0]
    weights = counts.sum(axis=(1, 2))
    centers = reference._norm_mean.loc[terms].to_numpy()
    reference_survival = []
    for designs in counterfactuals:
        cumulative_hazard = sum(
            float(increment)
            * np.exp((designs[int(month) - 1] - centers) @ reference_coefficients)
            for month, increment in baseline_increments.items()
        )
        reference_survival.append(
            np.average(np.exp(-cumulative_hazard), weights=weights)
        )
    estimates = grouped.standardize(model, counterfactuals, weights)
    survival_error = float(np.max(np.abs(estimates - reference_survival)))
    validation = {
        "max_coefficient_error": coefficient_error,
        "log_likelihood_error": likelihood_error,
        "max_survival_error": survival_error,
        "lifelines_seconds": time.monotonic() - started,
    }
    if coefficient_error > 2e-5 or likelihood_error > 1e-5 or survival_error > 2e-6:
        raise AssertionError(validation)
    print("Independent lifelines validation:", validation, flush=True)
    return validation


def summarize_survival(endpoint, patients, estimates, bootstrap_values):
    rows = []
    for index, (era, black) in enumerate(RACE_ERA_GROUPS):
        lower, upper = np.percentile(bootstrap_values[:, index], [2.5, 97.5])
        rows.append(
            {
                "Endpoint": endpoint,
                "Race": RACES[black],
                "Era": ERAS[era - 1],
                "Years": YEARS[era - 1],
                "n": int(
                    ((patients["era"] == era) & (patients["black"] == black)).sum()
                ),
                "Standardized 24-month survival": estimates[index],
                "95% CI lower": lower,
                "95% CI upper": upper,
            }
        )
    return rows


def summarize_contrasts(endpoint, estimates, bootstrap_values):
    rows = []
    for black in [0, 1]:
        for earlier, later in [(1, 2), (2, 3), (3, 4), (1, 4)]:
            earlier_index = RACE_ERA_GROUPS.index((earlier, black))
            later_index = RACE_ERA_GROUPS.index((later, black))
            bootstrap_change = (
                bootstrap_values[:, later_index] - bootstrap_values[:, earlier_index]
            )
            lower, upper = np.percentile(bootstrap_change, [2.5, 97.5])
            rows.append(
                {
                    "Endpoint": endpoint,
                    "Race": RACES[black],
                    "Comparison": ERAS[later - 1] + " vs " + ERAS[earlier - 1],
                    "Absolute change": estimates[later_index]
                    - estimates[earlier_index],
                    "95% CI lower": lower,
                    "95% CI upper": upper,
                }
            )

    for earlier, later in [(3, 4), (1, 4)]:
        black_later = RACE_ERA_GROUPS.index((later, 1))
        black_earlier = RACE_ERA_GROUPS.index((earlier, 1))
        white_later = RACE_ERA_GROUPS.index((later, 0))
        white_earlier = RACE_ERA_GROUPS.index((earlier, 0))

        def difference_in_gains(values):
            return (
                values[..., black_later]
                - values[..., black_earlier]
                - values[..., white_later]
                + values[..., white_earlier]
            )

        lower, upper = np.percentile(difference_in_gains(bootstrap_values), [2.5, 97.5])
        rows.append(
            {
                "Endpoint": endpoint,
                "Race": "NHB change minus NHW change",
                "Comparison": ERAS[later - 1] + " vs " + ERAS[earlier - 1],
                "Absolute change": float(difference_in_gains(estimates)),
                "95% CI lower": lower,
                "95% CI upper": upper,
            }
        )
    return rows


def fit_endpoint(
    endpoint, data, time_column, event_column, n_bootstrap, workers, validate=True
):
    started = time.monotonic()
    patients, covariates, counts, frame = prepare_model_data(
        data, time_column, event_column
    )
    time_terms, screening_rows = select_time_varying_terms(frame)
    print(endpoint, "N", len(patients), "time-varying terms", time_terms, flush=True)
    time_indices = [BASE_TERMS.index(term) for term in time_terms]
    designs = grouped.make_designs(build_design_matrix(covariates), time_indices)
    counterfactuals = build_counterfactuals(covariates, time_terms)
    model = grouped.fit(designs, counts)

    terms = BASE_TERMS + ["tv__" + term for term in time_terms]
    interaction_columns = INTERACTION_TERMS + [
        "tv__" + term for term in INTERACTION_TERMS
    ]
    reduced_indices = [
        index for index, term in enumerate(terms) if term not in interaction_columns
    ]
    reduced_model = grouped.fit(designs[:, :, reduced_indices], counts)
    likelihood_ratio = 2 * (model.log_likelihood - reduced_model.log_likelihood)
    interaction_df = len(model.coefficients) - len(reduced_model.coefficients)
    interaction_p = float(chi2.sf(likelihood_ratio, interaction_df))
    estimates = grouped.standardize(model, counterfactuals, counts.sum(axis=(1, 2)))
    validation = None
    if validate:
        validation = validate_against_lifelines(
            frame,
            time_terms,
            model,
            counts,
            counterfactuals,
        )
    print(
        endpoint,
        "point estimates",
        estimates,
        "overall interaction",
        interaction_p,
        "df",
        interaction_df,
        flush=True,
    )

    successful, failures = run_bootstrap(
        covariates,
        counts,
        designs,
        counterfactuals,
        model,
        n_bootstrap,
        workers,
        endpoint,
    )
    bootstrap_values = np.array([result["values"] for result in successful])
    np.savez_compressed(
        TABLE_DIR / f"23_{endpoint.lower()}_bootstrap_draws.npz",
        values=bootstrap_values,
        seeds=[result["seed"] for result in successful],
    )
    survival_rows = summarize_survival(endpoint, patients, estimates, bootstrap_values)
    contrast_rows = summarize_contrasts(endpoint, estimates, bootstrap_values)
    coefficient_rows = [
        {
            "Endpoint": endpoint,
            "Term": term,
            "Coefficient": float(coefficient),
            "Standard error": float(np.sqrt(model.covariance[index, index])),
        }
        for index, (term, coefficient) in enumerate(zip(terms, model.coefficients))
    ]
    configuration = {
        "Endpoint": endpoint,
        "N": len(patients),
        "Events": int(patients["event"].sum()),
        "Time-varying terms": time_terms,
        "Interaction LR statistic": likelihood_ratio,
        "Interaction df": interaction_df,
        "Overall four-era interaction p": interaction_p,
        "Bootstrap attempted": n_bootstrap,
        "Bootstrap converged": len(successful),
        "Failures": failures,
        "Seed": BOOTSTRAP_SEED,
        "Stage included": False,
        "Validation": validation,
        "Run seconds": time.monotonic() - started,
        "Max score": model.max_score,
    }
    for row in screening_rows:
        row["Endpoint"] = endpoint
    prefix = TABLE_DIR / f"23_{endpoint.lower()}_four_era"
    Path(f"{prefix}_model.json").write_text(json.dumps(configuration, indent=2))
    for suffix, rows in [
        ("survival", survival_rows),
        ("contrasts", contrast_rows),
        ("ph_screen", screening_rows),
        ("coefficients", coefficient_rows),
    ]:
        pd.DataFrame(rows).to_csv(f"{prefix}_{suffix}.csv", index=False)
    return survival_rows, contrast_rows, configuration


def redraw_figure(summary):
    plot_results(summary, output_dir=FIGURE_DIR)
    export_figure3(figure_dir=FIGURE_DIR)


def main(n_bootstrap=N_BOOTSTRAP, workers=4, validate=True, endpoint_filter=None):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    survival_rows = []
    contrast_rows = []
    for endpoint, data_module, time_column, event_column in ENDPOINTS:
        if endpoint_filter and endpoint != endpoint_filter:
            continue
        endpoint_survival, endpoint_contrasts, _ = fit_endpoint(
            endpoint,
            data_module.load_and_prepare_data(),
            time_column,
            event_column,
            n_bootstrap,
            workers,
            validate,
        )
        survival_rows.extend(endpoint_survival)
        contrast_rows.extend(endpoint_contrasts)
    if endpoint_filter:
        return
    summary = pd.DataFrame(survival_rows)
    summary.to_csv(TABLE_DIR / "23_adjusted_race_era_survival.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(
        TABLE_DIR / "23_adjusted_race_era_contrasts.csv",
        index=False,
    )
    redraw_figure(summary)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate the final four-era Figure 3 and its analysis outputs.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Redraw and export the final figure using saved estimates.",
    )
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Development only; final analyses require validation.",
    )
    parser.add_argument("--endpoint", choices=["OS", "PCSS"])
    arguments = parser.parse_args()
    if arguments.bootstrap < 1 or arguments.workers < 1:
        parser.error("--bootstrap and --workers must be positive integers")
    return arguments


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.plot_only:
        redraw_figure(pd.read_csv(TABLE_DIR / "23_adjusted_race_era_survival.csv"))
    else:
        main(
            arguments.bootstrap,
            arguments.workers,
            not arguments.skip_validation,
            arguments.endpoint,
        )

from dataclasses import dataclass

import numpy as np

ANALYSIS_MONTHS = 24
REFERENCE_MONTH = 6.0
MAX_ITERATIONS = 60
MAX_LINE_SEARCH_STEPS = 30


@dataclass
class GroupedCoxFit:
    coefficients: np.ndarray
    log_likelihood: float
    covariance: np.ndarray
    baseline_increments: np.ndarray
    iterations: int
    max_score: float


def moments(counts):
    exits = counts.sum(axis=2)
    patients_at_risk = np.cumsum(exits[:, ::-1], axis=1)[:, ::-1]
    deaths = counts[:, :, 1]
    return patients_at_risk, deaths


def likelihood(coefficients, designs, risk, deaths, hessian=True):
    n_terms = len(coefficients)
    log_likelihood = 0.0
    score = np.zeros(n_terms)
    information = np.zeros((n_terms, n_terms))

    for month_index, covariates in enumerate(designs):
        n_deaths = int(deaths[:, month_index].sum())
        if n_deaths == 0:
            continue

        linear_predictor = covariates @ coefficients
        numerical_shift = np.max(linear_predictor[risk[:, month_index] > 0])
        relative_hazards = np.exp(linear_predictor - numerical_shift)
        risk_weights = risk[:, month_index] * relative_hazards
        death_weights = deaths[:, month_index] * relative_hazards
        risk_total = risk_weights.sum()
        death_total = death_weights.sum()
        risk_first_moment = covariates.T @ risk_weights
        death_first_moment = covariates.T @ death_weights
        observed_covariates = covariates.T @ deaths[:, month_index]

        efron_fractions = np.arange(n_deaths) / n_deaths
        denominators = risk_total - efron_fractions * death_total
        inverse_denominators = 1.0 / denominators
        inverse_sum = inverse_denominators.sum()
        fraction_inverse_sum = np.dot(efron_fractions, inverse_denominators)

        log_likelihood += (
            observed_covariates @ coefficients
            - np.log(denominators).sum()
            - n_deaths * numerical_shift
        )
        score += (
            observed_covariates
            - risk_first_moment * inverse_sum
            + death_first_moment * fraction_inverse_sum
        )
        if hessian:
            risk_second_moment = covariates.T @ (risk_weights[:, None] * covariates)
            death_second_moment = covariates.T @ (death_weights[:, None] * covariates)
            squared_inverses = inverse_denominators * inverse_denominators
            squared_inverse_sum = squared_inverses.sum()
            fraction_squared_inverse_sum = np.dot(efron_fractions, squared_inverses)
            squared_fraction_inverse_sum = np.dot(
                efron_fractions * efron_fractions,
                squared_inverses,
            )
            information += (
                risk_second_moment * inverse_sum
                - death_second_moment * fraction_inverse_sum
                - np.outer(risk_first_moment, risk_first_moment) * squared_inverse_sum
                + (
                    np.outer(risk_first_moment, death_first_moment)
                    + np.outer(death_first_moment, risk_first_moment)
                )
                * fraction_squared_inverse_sum
                - np.outer(death_first_moment, death_first_moment)
                * squared_fraction_inverse_sum
            )
    return float(log_likelihood), score, information


def fit(designs, counts, initial=None):
    risk, deaths = moments(counts)
    coefficients = (
        np.zeros(designs.shape[2]) if initial is None else np.array(initial, copy=True)
    )
    for iteration in range(1, MAX_ITERATIONS + 1):
        log_likelihood, score, information = likelihood(
            coefficients,
            designs,
            risk,
            deaths,
        )
        newton_step = np.linalg.solve(information, score)
        if np.max(np.abs(score)) < 1e-6 or np.max(np.abs(newton_step)) < 1e-9:
            break

        step_size = 1.0
        for _ in range(MAX_LINE_SEARCH_STEPS):
            candidate = coefficients + step_size * newton_step
            candidate_likelihood = likelihood(
                candidate,
                designs,
                risk,
                deaths,
                hessian=False,
            )[0]
            if (
                np.isfinite(candidate_likelihood)
                and candidate_likelihood >= log_likelihood - 1e-8
            ):
                coefficients = candidate
                break
            step_size *= 0.5
        else:
            raise RuntimeError("Cox Newton line search failed")
    else:
        raise RuntimeError("Cox Newton solver did not converge")

    log_likelihood, score, information = likelihood(coefficients, designs, risk, deaths)
    max_score = float(np.max(np.abs(score)))
    if max_score > 1e-4:
        raise RuntimeError(f"Unresolved Cox score {max_score}")
    baseline_increments = np.array(
        [
            deaths[:, month_index].sum()
            / np.dot(risk[:, month_index], np.exp(covariates @ coefficients))
            for month_index, covariates in enumerate(designs)
        ]
    )
    return GroupedCoxFit(
        coefficients=coefficients,
        log_likelihood=log_likelihood,
        covariance=np.linalg.inv(information),
        baseline_increments=baseline_increments,
        iterations=iteration,
        max_score=max_score,
    )


def make_designs(base, time_indices):
    monthly_designs = []
    for month in range(1, ANALYSIS_MONTHS + 1):
        if len(time_indices):
            time_varying_covariates = base[:, time_indices] * np.log(
                month / REFERENCE_MONTH
            )
            monthly_designs.append(np.column_stack([base, time_varying_covariates]))
        else:
            monthly_designs.append(base.copy())
    return np.stack(monthly_designs)


def standardize(model, counterfactual_designs, weights):
    survival_estimates = []
    for designs in counterfactual_designs:
        cumulative_hazard = sum(
            increment * np.exp(covariates @ model.coefficients)
            for increment, covariates in zip(model.baseline_increments, designs)
        )
        survival_estimates.append(
            np.average(np.exp(-cumulative_hazard), weights=weights)
        )
    return np.array(survival_estimates)

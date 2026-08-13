
import cox_ph_diagnostics
import era_race_interaction
import median_os_by_year_white_black
import median_survival_by_era_and_race
import median_survival_by_stage_race_and_era
import os_km_by_era_and_stage
import os_km_by_race
import os_km_by_race_and_era
import os_km_by_stage_race_and_era
import os_km_white_black_by_era
import patient_characteristics
import pcss_km_by_era_and_stage
import pcss_km_by_race
import pcss_km_by_race_and_era
import pcss_km_by_stage_race_and_era
import pcss_km_white_black_by_era
import ph_aware_cox


ANALYSES = [
    ("Overall-survival curves by era and stage", os_km_by_era_and_stage.main),
    (
        "Cancer-specific survival curves by era and stage",
        pcss_km_by_era_and_stage.main,
    ),
    ("Overall-survival curves by race", os_km_by_race.main),
    ("Cancer-specific survival curves by race", pcss_km_by_race.main),
    ("Overall-survival curves by race and era", os_km_by_race_and_era.main),
    (
        "Cancer-specific survival curves by race and era",
        pcss_km_by_race_and_era.main,
    ),
    (
        "Overall-survival curves by stage, race, and era",
        os_km_by_stage_race_and_era.main,
    ),
    (
        "Cancer-specific survival curves by stage, race, and era",
        pcss_km_by_stage_race_and_era.main,
    ),
    ("White-Black overall-survival curves by era", os_km_white_black_by_era.main),
    (
        "White-Black cancer-specific survival curves by era",
        pcss_km_white_black_by_era.main,
    ),
    ("Annual median overall survival", median_os_by_year_white_black.main),
    ("Patient characteristics", patient_characteristics.main),
    ("Median survival by era and race", median_survival_by_era_and_race.main),
    (
        "Median survival by stage, race, and era",
        median_survival_by_stage_race_and_era.main,
    ),
    ("Proportional-hazards diagnostics", cox_ph_diagnostics.main),
    ("Final PH-aware Cox tables", ph_aware_cox.main),
    ("Era-by-race interaction models", era_race_interaction.main),
]


def main():
    for description, run_analysis in ANALYSES:
        print(f"Running: {description}", flush=True)
        run_analysis()


if __name__ == "__main__":
    main()

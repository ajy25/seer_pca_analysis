from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "pca-survival" / "data" / "export_causespecific.csv"
COLUMN_NAMES = {
    "Age recode with <1 year olds and 90+": "age_recode",
    "Year of diagnosis": "year_dx",
    "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)": "race_raw",
    "RX Summ--Surg/Rad Seq": "surg_rad_seq",
    "RX Summ--Surg Prim Site (1998+)": "surg_code",
    "Radiation recode": "rad_recode",
    "Derived EOD 2018 M Recode (2018+)": "m_eod2018",
    "Derived SEER Combined M (2016-2017)": "m_seer_combined",
    "Derived AJCC M, 7th ed (2010-2015)": "m_ajcc7",
    "Derived AJCC M, 6th ed (2004-2015)": "m_ajcc6",
    "M value - based on AJCC 3rd (1988-2003)": "m_ajcc3",
    "Age recode with single ages and 90+": "age_single",
    "Sex": "sex",
    "Vital status recode (study cutoff used)": "vital_status",
    "Year of follow-up recode": "year_followup",
    "End Calc Vital Status (Adjusted)": "end_calc_status",
    "Number of Intervals (Calculated)": "surv_months",
}
ERA_LABELS = {
    1: "Pre-Docetaxel (2000-2003)",
    2: "Post-Docetaxel (2004-2009)",
    3: "Early ARPI (2010-2016)",
    4: "Modern ARPI (2017-2020)",
}
ERA_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
}
NO_RADIATION_TREATMENT = {
    "None/Unknown",
    "Refused (1988+)",
    "Recommended, unknown if administered",
}
PROSTATECTOMY_CODES = {50, 70}
RACE_MAP = {
    "Non-Hispanic White": "Non-Hispanic White",
    "Non-Hispanic Black": "Non-Hispanic Black",
    "Hispanic (All Races)": "Hispanic",
    "Non-Hispanic Asian or Pacific Islander": "Non-Hispanic Asian/PI",
    "Non-Hispanic American Indian/Alaska Native": "Non-Hispanic AI/AN",
    "Non-Hispanic Unknown Race": "Non-Hispanic Unknown Race",
}
MAIN_RACES = [
    "Non-Hispanic White",
    "Non-Hispanic Black",
    "Hispanic",
    "Non-Hispanic Asian/PI",
]


def derive_cause_specific_death_event(end_calc_status: pd.Series) -> pd.Series:
    status_str = end_calc_status.astype(str).str.strip()
    is_dead_text = status_str.str.lower().eq("dead")
    status_num = pd.to_numeric(status_str, errors="coerce")
    is_dead_numeric = status_num.eq(2)
    return (is_dead_text | is_dead_numeric).astype(int)


def harmonize_m_stage(data: pd.DataFrame) -> pd.Series:
    stage = pd.Series(index=data.index, dtype="object")
    yr = data["year_dx"]

    m3 = data["m_ajcc3"].astype(str)
    mask = yr <= 2003
    stage[
        mask & m3.str.contains("M1", na=False) & ~m3.str.contains("Blank", na=False)
    ] = "M1"

    m6 = data["m_ajcc6"].astype(str)
    mask = (yr >= 2004) & (yr <= 2009)
    stage[mask & m6.str.contains("M1a", na=False)] = "M1a"
    stage[mask & m6.str.contains("M1b", na=False)] = "M1b"
    stage[mask & m6.str.contains("M1c", na=False)] = "M1c"
    stage[mask & m6.str.contains("M1NOS", na=False)] = "M1NOS"
    stage[mask & m6.str.match(r"^M1$", na=False)] = "M1NOS"

    m7 = data["m_ajcc7"].astype(str)
    mask = (yr >= 2010) & (yr <= 2015)
    stage[mask & m7.str.contains("M1a", na=False)] = "M1a"
    stage[mask & m7.str.contains("M1b", na=False)] = "M1b"
    stage[mask & m7.str.contains("M1c", na=False)] = "M1c"
    stage[mask & m7.str.contains("M1NOS", na=False)] = "M1NOS"
    stage[mask & m7.str.match(r"^M1$", na=False)] = "M1NOS"

    mc = data["m_seer_combined"].astype(str).str.upper()
    mask = (yr >= 2016) & (yr <= 2017)
    stage[mask & mc.str.contains("1A", na=False)] = "M1a"
    stage[mask & mc.str.contains("1B", na=False)] = "M1b"
    stage[mask & mc.str.contains("1C", na=False)] = "M1c"
    nos_mask = (
        mask
        & ~mc.str.contains("1A|1B|1C|BLANK", na=False, regex=True)
        & mc.str.contains("^[CP]1$", na=False, regex=True)
    )
    stage[nos_mask] = "M1NOS"

    me = data["m_eod2018"].astype(str)
    mask = yr >= 2018
    stage[mask & me.str.contains("M1a", na=False)] = "M1a"
    stage[mask & me.str.contains("M1b", na=False)] = "M1b"
    stage[mask & me.str.contains("M1c", na=False)] = "M1c"
    stage[mask & me.str.match(r"^M1$", na=False)] = "M1NOS"

    stage[stage.isin(["", "nan", "Blank(s)", "BLANK(S)"])] = np.nan
    return stage


def load_and_prepare_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data = data.rename(columns=COLUMN_NAMES)

    data = data[(data["year_dx"] >= 2000) & (data["year_dx"] <= 2020)].copy()
    data["age"] = data["age_single"].astype(str).str.extract(r"(\d+)").astype(float)
    data = data[data["age"] >= 18].copy()

    era_bins = [1999, 2003, 2009, 2016, 2020]
    era_labels = [1, 2, 3, 4]
    data["era"] = (
        pd.cut(
            data["year_dx"],
            bins=era_bins,
            labels=era_labels,
        )
        .astype(float)
        .astype("Int64")
    )
    data["era_label"] = data["era"].map(ERA_LABELS)

    data["m_stage"] = harmonize_m_stage(data)
    data = data.dropna(subset=["m_stage"]).copy()

    data["race"] = data["race_raw"].map(RACE_MAP).fillna("Unknown")

    surg_code_num = pd.to_numeric(data["surg_code"], errors="coerce")
    had_rad = ~data["rad_recode"].isin(NO_RADIATION_TREATMENT)

    data["prostatectomy"] = surg_code_num.isin(PROSTATECTOMY_CODES).astype(int)
    data["radiotherapy"] = had_rad.astype(int)

    data["pca_event"] = derive_cause_specific_death_event(data["end_calc_status"])
    data["pca_months"] = data["surv_months"].astype(float)
    data = data.dropna(subset=["pca_months"]).copy()

    return data.reset_index(drop=True)


if __name__ == "__main__":
    data = load_and_prepare_data()
    print(f"n = {len(data):,}")
    print(data["era"].value_counts().sort_index().to_string())
    print(data["m_stage"].value_counts().to_string())
    print(data["race"].value_counts().to_string())
    print(f"PCA event rate: {data['pca_event'].mean():.3f}")

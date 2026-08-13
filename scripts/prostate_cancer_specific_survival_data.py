from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "pca-survival" / "data" / "export_causespecific.csv"

COL_RENAMING_MAP = {
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

NO_SURGICAL_CODES = {0, 99}
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


def harmonize_m_stage(df: pd.DataFrame) -> pd.Series:
    stage = pd.Series(index=df.index, dtype="object")
    yr = df["year_dx"]

    m3 = df["m_ajcc3"].astype(str)
    mask = yr <= 2003
    stage[
        mask & m3.str.contains("M1", na=False) & ~m3.str.contains("Blank", na=False)
    ] = "M1"

    m6 = df["m_ajcc6"].astype(str)
    mask = (yr >= 2004) & (yr <= 2009)
    stage[mask & m6.str.contains("M1a", na=False)] = "M1a"
    stage[mask & m6.str.contains("M1b", na=False)] = "M1b"
    stage[mask & m6.str.contains("M1c", na=False)] = "M1c"
    stage[mask & m6.str.contains("M1NOS", na=False)] = "M1NOS"
    stage[mask & m6.str.match(r"^M1$", na=False)] = "M1NOS"

    m7 = df["m_ajcc7"].astype(str)
    mask = (yr >= 2010) & (yr <= 2015)
    stage[mask & m7.str.contains("M1a", na=False)] = "M1a"
    stage[mask & m7.str.contains("M1b", na=False)] = "M1b"
    stage[mask & m7.str.contains("M1c", na=False)] = "M1c"
    stage[mask & m7.str.contains("M1NOS", na=False)] = "M1NOS"
    stage[mask & m7.str.match(r"^M1$", na=False)] = "M1NOS"

    mc = df["m_seer_combined"].astype(str).str.upper()
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

    me = df["m_eod2018"].astype(str)
    mask = yr >= 2018
    stage[mask & me.str.contains("M1a", na=False)] = "M1a"
    stage[mask & me.str.contains("M1b", na=False)] = "M1b"
    stage[mask & me.str.contains("M1c", na=False)] = "M1c"
    stage[mask & me.str.match(r"^M1$", na=False)] = "M1NOS"

    stage[stage.isin(["", "nan", "Blank(s)", "BLANK(S)"])] = np.nan
    return stage


def load_and_prepare_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=COL_RENAMING_MAP)

    df = df[(df["year_dx"] >= 2000) & (df["year_dx"] <= 2020)].copy()
    df["age"] = df["age_single"].astype(str).str.extract(r"(\d+)").astype(float)
    df = df[df["age"] >= 18].copy()

    era_bins = [1999, 2003, 2009, 2016, 2020]
    era_labels = [1, 2, 3, 4]
    df["era"] = (
        pd.cut(
            df["year_dx"],
            bins=era_bins,
            labels=era_labels,
        )
        .astype(float)
        .astype("Int64")
    )
    df["era_label"] = df["era"].map(ERA_LABELS)

    df["m_stage"] = harmonize_m_stage(df)
    df = df.dropna(subset=["m_stage"]).copy()

    df["race"] = df["race_raw"].map(RACE_MAP).fillna("Unknown")

    surg_code_num = pd.to_numeric(df["surg_code"], errors="coerce")
    had_surg = ~surg_code_num.isin(NO_SURGICAL_CODES)
    had_rad = ~df["rad_recode"].isin(NO_RADIATION_TREATMENT)

    df["prostatectomy"] = surg_code_num.isin(PROSTATECTOMY_CODES).astype(int)
    df["radiotherapy"] = had_rad.astype(int)
    df["prostate_surgery"] = had_surg.astype(int)
    df["local_tx"] = (had_surg | had_rad).astype(int)
    df["local_tx_label"] = df["local_tx"].map(
        {1: "Prostatectomy and/or Radiotherapy", 0: "No local therapy"}
    )

    df["pca_event"] = derive_cause_specific_death_event(df["end_calc_status"])
    df["pca_months"] = df["surv_months"].astype(float)
    df = df.dropna(subset=["pca_months"]).copy()

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_and_prepare_data()
    print(f"n = {len(df):,}")
    print(df["era"].value_counts().sort_index().to_string())
    print(df["m_stage"].value_counts().to_string())
    print(df["race"].value_counts().to_string())
    print(f"PCA event rate: {df['pca_event'].mean():.3f}")

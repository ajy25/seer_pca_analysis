
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

from overall_survival_data import load_and_prepare_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2000, 2021))
RACE_ORDER = ["Non-Hispanic White", "Non-Hispanic Black"]
RACE_COLORS = {
    "Non-Hispanic White": "#1f77b4",
    "Non-Hispanic Black": "#d62728",
}
RACE_SHORT = {
    "Non-Hispanic White": "NH White",
    "Non-Hispanic Black": "NH Black",
}


def km_median_ci(times, events):
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float)
    if len(times) == 0 or events.sum() == 0:
        return np.nan, np.nan, np.nan
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed=events)
    med = kmf.median_survival_time_
    if med == np.inf:
        return np.nan, np.nan, np.nan
    ci = median_survival_times(kmf.confidence_interval_)
    lo, hi = ci.iloc[0, 0], ci.iloc[0, 1]
    if lo == np.inf:
        lo = np.nan
    if hi == np.inf:
        hi = np.nan
    return med, lo, hi


def compute_year_race_medians(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for race in RACE_ORDER:
        for year in YEARS:
            sub = df[(df["race"] == race) & (df["year_dx"] == year)]
            sub = sub.dropna(subset=["os_months", "os_event"])
            n = len(sub)
            events = int(sub["os_event"].sum()) if n > 0 else 0
            med, lo, hi = km_median_ci(sub["os_months"], sub["os_event"])
            rows.append(
                {
                    "race": race,
                    "year": year,
                    "n": n,
                    "events": events,
                    "median_os": med,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    return pd.DataFrame(rows)


def main():
    df = load_and_prepare_data()
    df = df[df["race"].isin(RACE_ORDER)].copy()
    summary = compute_year_race_medians(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    for race in RACE_ORDER:
        sub = summary[summary["race"] == race].sort_values("year")
        reached = sub.dropna(subset=["median_os"])
        ax.plot(
            reached["year"],
            reached["median_os"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=RACE_COLORS[race],
            label=RACE_SHORT[race],
        )

    ax.set_xlabel("Year of Diagnosis", fontsize=13)
    ax.set_ylabel("Median Overall Survival (months)", fontsize=13)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlim(min(YEARS) - 0.5, max(YEARS) + 0.5)
    ax.set_ylim(18, 38)
    ax.set_yticks([18, 24, 30, 36])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y)}"))

    fig.tight_layout()
    out_path = OUTPUT_DIR / "11_median_os_by_year_white_black.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

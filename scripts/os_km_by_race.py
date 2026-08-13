
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid")
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from overall_survival_data import (
    MAIN_RACES,
    load_and_prepare_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_LIMIT = 24

RACE_COLORS = {
    "Non-Hispanic White": "#1f77b4",
    "Non-Hispanic Black": "#d62728",
    "Hispanic": "#2ca02c",
    "Non-Hispanic Asian/PI": "#ff7f0e",
}
RACE_SHORT = {
    "Non-Hispanic White": "NH White",
    "Non-Hispanic Black": "NH Black",
    "Hispanic": "Hispanic",
    "Non-Hispanic Asian/PI": "NH Asian/PI",
}


def plot_race_panel(ax, df, time_col, event_col, title):
    kmf_list = []
    for race in MAIN_RACES:
        sub = df[df["race"] == race].dropna(subset=[time_col, event_col]).copy()
        if len(sub) == 0:
            continue
        mask = sub[time_col] > X_LIMIT
        sub.loc[mask, event_col] = 0
        sub.loc[mask, time_col] = X_LIMIT
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], event_observed=sub[event_col], label=RACE_SHORT[race])
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=RACE_COLORS[race],
            linewidth=2,
        )
        kmf_list.append(kmf)

    sub_all = (
        df[df["race"].isin(MAIN_RACES)].dropna(subset=[time_col, event_col]).copy()
    )
    mask = sub_all[time_col] > X_LIMIT
    sub_all.loc[mask, event_col] = 0
    sub_all.loc[mask, time_col] = X_LIMIT
    if len(sub_all) > 0 and sub_all["race"].nunique() > 1:
        lr = multivariate_logrank_test(
            sub_all[time_col], sub_all["race"], sub_all[event_col]
        )
        p_val = lr.p_value
        p_txt = "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
    else:
        p_txt = "N/A"

    ax.text(
        0.98,
        0.98,
        f"Log-rank\n{p_txt}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )

    ax.set_xlim(0, X_LIMIT)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (Years)", fontsize=13)
    ax.set_ylabel("Proportion Surviving", fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x // 12)}"))
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(True, linestyle="--", alpha=0.4)

    return kmf_list


def main():
    df = load_and_prepare_data()
    df_modern = df[df["era"].isin([3, 4])].copy()
    total_n = len(df_modern[df_modern["race"].isin(MAIN_RACES)])
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plot_race_panel(ax, df_modern, "os_months", "os_event", "Overall Survival")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "03_os_km_by_race.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()


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
    ERA_COLORS,
    ERA_LABELS,
    MAIN_RACES,
    load_and_prepare_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_LIMIT = 24
ERA_ORDER = [1, 2, 3, 4]

PANEL_LETTERS = {
    "Non-Hispanic White": "A",
    "Non-Hispanic Black": "B",
    "Hispanic": "C",
    "Non-Hispanic Asian/PI": "D",
}
PANEL_SUBTITLES = {
    "Non-Hispanic White": "Non-Hispanic White",
    "Non-Hispanic Black": "Non-Hispanic Black",
    "Hispanic": "Hispanic",
    "Non-Hispanic Asian/PI": "Non-Hispanic Asian/PI",
}


def plot_race_era_panel(ax, df_race, race_label):
    kmf_list = []
    for e in ERA_ORDER:
        sub = (
            df_race[df_race["era"] == e].dropna(subset=["os_months", "os_event"]).copy()
        )
        if len(sub) == 0:
            continue
        mask = sub["os_months"] > X_LIMIT
        sub.loc[mask, "os_event"] = 0
        sub.loc[mask, "os_months"] = X_LIMIT
        kmf = KaplanMeierFitter()
        kmf.fit(sub["os_months"], event_observed=sub["os_event"], label=ERA_LABELS[e])
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=ERA_COLORS[e],
            linewidth=2,
        )
        kmf_list.append(kmf)

    sub_all = (
        df_race[df_race["era"].isin(ERA_ORDER)]
        .dropna(subset=["os_months", "os_event"])
        .copy()
    )
    mask = sub_all["os_months"] > X_LIMIT
    sub_all.loc[mask, "os_event"] = 0
    sub_all.loc[mask, "os_months"] = X_LIMIT
    if len(sub_all) > 0 and sub_all["era"].nunique() > 1:
        lr = multivariate_logrank_test(
            sub_all["os_months"], sub_all["era"], sub_all["os_event"]
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

    n_total = len(df_race)
    ax.set_xlim(0, X_LIMIT)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (Years)", fontsize=13)
    ax.set_ylabel("Proportion Surviving", fontsize=13)
    ax.text(
        -0.12,
        1.05,
        PANEL_LETTERS[race_label],
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
    )
    ax.set_title(
        f"{PANEL_SUBTITLES[race_label]} (n = {n_total:,})",
        fontsize=11,
        fontweight="bold",
        loc="center",
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x // 12)}"))
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(True, linestyle="--", alpha=0.4)

    return kmf_list


def main():
    df = load_and_prepare_data()
    scale = 0.7
    fig, axes = plt.subplots(2, 2, figsize=(18 * scale, 14 * scale))
    axes_flat = axes.flatten()
    for ax, race in zip(axes_flat, MAIN_RACES):
        df_race = df[df["race"] == race].copy()
        plot_race_era_panel(ax, df_race, race)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "05_os_km_race_era.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

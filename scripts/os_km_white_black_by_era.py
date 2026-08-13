
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid")
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

from overall_survival_data import (
    ERA_LABELS,
    load_and_prepare_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERAS = [1, 2, 3, 4]
RACE_ORDER = ["Non-Hispanic White", "Non-Hispanic Black"]
RACE_COLORS = {
    "Non-Hispanic White": "#1f77b4",
    "Non-Hispanic Black": "#d62728",
}
RACE_SHORT = {
    "Non-Hispanic White": "NH White",
    "Non-Hispanic Black": "NH Black",
}

X_LIMIT = 24
PANEL_LETTERS = {1: "A", 2: "B", 3: "C", 4: "D"}


def format_p_value(p_val):
    if p_val is None:
        return "N/A"
    return "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"


def panel_title(era: int, n_total: int):
    return f"{ERA_LABELS[era]} (n = {n_total:,})"


def plot_panel(ax, df_era: pd.DataFrame, era: int):
    for race in RACE_ORDER:
        sub = (
            df_era[df_era["race"] == race]
            .dropna(subset=["os_months", "os_event"])
            .copy()
        )
        if len(sub) == 0:
            continue
        mask = sub["os_months"] > X_LIMIT
        sub.loc[mask, "os_event"] = 0
        sub.loc[mask, "os_months"] = X_LIMIT

        kmf = KaplanMeierFitter()
        kmf.fit(
            sub["os_months"], event_observed=sub["os_event"], label=RACE_SHORT[race]
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=RACE_COLORS[race],
            linewidth=2,
        )

    white = (
        df_era[df_era["race"] == "Non-Hispanic White"]
        .dropna(subset=["os_months", "os_event"])
        .copy()
    )
    mask = white["os_months"] > X_LIMIT
    white.loc[mask, "os_event"] = 0
    white.loc[mask, "os_months"] = X_LIMIT
    black = (
        df_era[df_era["race"] == "Non-Hispanic Black"]
        .dropna(subset=["os_months", "os_event"])
        .copy()
    )
    mask = black["os_months"] > X_LIMIT
    black.loc[mask, "os_event"] = 0
    black.loc[mask, "os_months"] = X_LIMIT

    if len(white) > 0 and len(black) > 0:
        lr = logrank_test(
            white["os_months"],
            black["os_months"],
            event_observed_A=white["os_event"],
            event_observed_B=black["os_event"],
        )
        p_text = format_p_value(lr.p_value)
    else:
        p_text = "N/A"

    ax.text(
        0.98,
        0.98,
        f"Log-rank\n{p_text}",
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
    ax.text(
        -0.12,
        1.05,
        PANEL_LETTERS[era],
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
    )
    ax.set_title(
        panel_title(era, len(df_era)), fontsize=13, fontweight="bold", loc="center"
    )
    ax.set_xlabel("Time (Years)", fontsize=13)
    ax.set_ylabel("Proportion Surviving", fontsize=13)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x // 12)}"))
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)


def main():
    df = load_and_prepare_data()
    df = df[df["race"].isin(RACE_ORDER) & df["era"].isin(ERAS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), sharex=False, sharey=False)
    axes_flat = axes.flatten()
    for ax, era in zip(axes_flat, ERAS):
        df_era = df[df["era"] == era].copy()
        plot_panel(ax, df_era, era)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "09_os_km_white_black_by_era.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

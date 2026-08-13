
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid")
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from prostate_cancer_specific_survival_data import (
    ERA_COLORS,
    ERA_LABELS,
    load_and_prepare_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STAGES = ["M1a", "M1b", "M1c"]
ERAS = [2, 3, 4]

ROW_STRATA = [
    ("Overall", None),
    ("Non-Hispanic White", "Non-Hispanic White"),
    ("Non-Hispanic Black", "Non-Hispanic Black"),
]

PANEL_LETTERS = {
    (0, 0): "A",
    (0, 1): "B",
    (0, 2): "C",
    (1, 0): "D",
    (1, 1): "E",
    (1, 2): "F",
    (2, 0): "G",
    (2, 1): "H",
    (2, 2): "I",
}

X_LIMIT = 24


def format_p_value(p_val):
    if p_val is None:
        return "N/A"
    return "p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"


def plot_panel(ax, df_sub: pd.DataFrame, panel_title: str, panel_letter: str = None):
    for era in ERAS:
        sub_era = (
            df_sub[df_sub["era"] == era]
            .dropna(subset=["pca_months", "pca_event"])
            .copy()
        )
        if len(sub_era) == 0:
            continue
        mask = sub_era["pca_months"] > X_LIMIT
        sub_era.loc[mask, "pca_event"] = 0
        sub_era.loc[mask, "pca_months"] = X_LIMIT

        kmf = KaplanMeierFitter()
        kmf.fit(
            sub_era["pca_months"],
            event_observed=sub_era["pca_event"],
            label=ERA_LABELS[era],
        )
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=ERA_COLORS[era],
            linewidth=2,
        )

    sub_all = (
        df_sub[df_sub["era"].isin(ERAS)]
        .dropna(subset=["pca_months", "pca_event"])
        .copy()
    )
    mask = sub_all["pca_months"] > X_LIMIT
    sub_all.loc[mask, "pca_event"] = 0
    sub_all.loc[mask, "pca_months"] = X_LIMIT
    if len(sub_all) > 0 and sub_all["era"].nunique() > 1:
        lr = multivariate_logrank_test(
            sub_all["pca_months"],
            sub_all["era"],
            sub_all["pca_event"],
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
    if panel_letter:
        ax.text(
            -0.12,
            1.05,
            panel_letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
        )
    ax.set_title(panel_title, fontsize=12, fontweight="bold", loc="center")
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
    df = df[df["era"].isin(ERAS) & df["m_stage"].isin(STAGES)].copy()
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=False, sharey=False)
    for row_idx, (stratum_label, race_value) in enumerate(ROW_STRATA):
        for col_idx, stage in enumerate(STAGES):
            ax = axes[row_idx, col_idx]
            sub = df[df["m_stage"] == stage]
            if race_value is not None:
                sub = sub[sub["race"] == race_value]
            n_val = len(sub)
            letter = PANEL_LETTERS[(row_idx, col_idx)]
            title = f"{stage}, {stratum_label} (n = {n_val:,})"
            plot_panel(ax, sub, title, panel_letter=letter)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "08_pcss_km_stage_race_era.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

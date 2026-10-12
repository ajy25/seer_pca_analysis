from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from overall_survival_data import ERA_COLORS, ERA_LABELS, load_and_prepare_data

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
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
ANALYSIS_MONTHS = 24


def format_p_value(p_value):
    if p_value is None:
        return "N/A"
    return "p < 0.0001" if p_value < 0.0001 else f"p = {p_value:.4f}"


def plot_panel(
    ax, data_subset: pd.DataFrame, panel_title: str, panel_letter: str = None
):
    for era in ERAS:
        era_subset = (
            data_subset[data_subset["era"] == era]
            .dropna(subset=["os_months", "os_event"])
            .copy()
        )
        if len(era_subset) == 0:
            continue
        mask = era_subset["os_months"] > ANALYSIS_MONTHS
        era_subset.loc[mask, "os_event"] = 0
        era_subset.loc[mask, "os_months"] = ANALYSIS_MONTHS

        km_model = KaplanMeierFitter()
        km_model.fit(
            era_subset["os_months"],
            event_observed=era_subset["os_event"],
            label=ERA_LABELS[era],
        )
        km_model.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=ERA_COLORS[era],
            linewidth=2,
        )

    all_patients = (
        data_subset[data_subset["era"].isin(ERAS)]
        .dropna(subset=["os_months", "os_event"])
        .copy()
    )
    mask = all_patients["os_months"] > ANALYSIS_MONTHS
    all_patients.loc[mask, "os_event"] = 0
    all_patients.loc[mask, "os_months"] = ANALYSIS_MONTHS
    if len(all_patients) > 0 and all_patients["era"].nunique() > 1:
        logrank_result = multivariate_logrank_test(
            all_patients["os_months"],
            all_patients["era"],
            all_patients["os_event"],
        )
        p_text = format_p_value(logrank_result.p_value)
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

    ax.set_xlim(0, ANALYSIS_MONTHS)
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
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_prepare_data()
    data = data[data["era"].isin(ERAS) & data["m_stage"].isin(STAGES)].copy()
    fig, axes = plt.subplots(3, 3, figsize=(18, 14), sharex=False, sharey=False)
    for row_idx, (stratum_label, race_value) in enumerate(ROW_STRATA):
        for col_idx, stage in enumerate(STAGES):
            ax = axes[row_idx, col_idx]
            subset = data[data["m_stage"] == stage]
            if race_value is not None:
                subset = subset[subset["race"] == race_value]
            n_val = len(subset)
            letter = PANEL_LETTERS[(row_idx, col_idx)]
            title = f"{stage}, {stratum_label} (n = {n_val:,})"
            plot_panel(ax, subset, title, panel_letter=letter)
    plt.tight_layout()
    output_path = FIGURE_DIR / "07_os_km_stage_race_era.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

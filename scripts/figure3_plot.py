from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
FIGURE_STEM = "23_adjusted_race_era_survival"
SURVIVAL_COLUMN = "Standardized 24-month survival"
ERA_LABELS = ["Pre-Docetaxel", "Post-Docetaxel", "Early ARPI", "Modern ARPI"]
ERA_TICK_LABELS = [
    "Pre-\nDocetaxel\n(2000-2003)",
    "Post-\nDocetaxel\n(2004-2009)",
    "Early ARPI\n(2010-2016)",
    "Modern ARPI\n(2017-2020)",
]
RACE_STYLES = [
    ("Non-Hispanic White", "NH White", "#1f77b4"),
    ("Non-Hispanic Black", "NH Black", "#d62728"),
]
PANELS = [
    ("OS", "Overall Survival"),
    ("PCSS", "Prostate Cancer-Specific Survival"),
]
FONT_SETTINGS = {
    "font.family": "Arial",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
}


def plot_results(summary: pd.DataFrame, output_dir: Path = FIGURE_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context():
        sns.set_theme(style="whitegrid")
        plt.rcParams.update(FONT_SETTINGS)
        figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)

        for panel_index, (axis, (endpoint, title)) in enumerate(zip(axes, PANELS)):
            for race_index, (race, label, color) in enumerate(RACE_STYLES):
                estimates = (
                    summary[
                        (summary["Endpoint"] == endpoint) & (summary["Race"] == race)
                    ]
                    .set_index("Era")
                    .loc[ERA_LABELS]
                )
                survival = estimates[SURVIVAL_COLUMN].to_numpy() * 100
                lower = estimates["95% CI lower"].to_numpy() * 100
                upper = estimates["95% CI upper"].to_numpy() * 100

                positions = np.arange(4) + (race_index - 0.5) * 0.045
                axis.plot(
                    positions,
                    survival,
                    color=color,
                    marker="o",
                    markersize=4.5,
                    linewidth=2,
                    label=label,
                )
                axis.vlines(positions, lower, upper, color=color, linewidth=1.1)
                axis.hlines(
                    lower,
                    positions - 0.035,
                    positions + 0.035,
                    color=color,
                    linewidth=1.1,
                )
                axis.hlines(
                    upper,
                    positions - 0.035,
                    positions + 0.035,
                    color=color,
                    linewidth=1.1,
                )

            axis.set_xticks(range(4), ERA_TICK_LABELS)
            axis.set_xlim(-0.3, 3.3)
            axis.set_ylim(35, 75)
            axis.set_yticks(np.arange(35, 76, 5))
            axis.set_title(title, fontsize=13, fontweight="bold", loc="center")
            axis.text(
                -0.12,
                1.05,
                chr(65 + panel_index),
                transform=axis.transAxes,
                fontsize=18,
                fontweight="bold",
                va="bottom",
            )
            axis.grid(True, linestyle="--", alpha=0.4)
            for border in axis.spines.values():
                border.set_visible(True)
                border.set_color(".8")
                border.set_linewidth(1.25)
            axis.set_xlabel("Diagnosis Era", fontsize=13)
            axis.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.9)

        axes[0].set_ylabel("Adjusted 24-Month Survival (%)")
        figure.subplots_adjust(
            left=0.075,
            right=0.99,
            top=0.87,
            bottom=0.23,
            wspace=0.15,
        )
        for extension in ("png", "pdf"):
            figure.savefig(
                output_dir / f"{FIGURE_STEM}.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(figure)

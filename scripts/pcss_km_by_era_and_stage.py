from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from prostate_cancer_specific_survival_data import (
    ERA_COLORS,
    ERA_LABELS,
    load_and_prepare_data,
)

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
ANALYSIS_MONTHS = 24
ERAS = [1, 2, 3, 4]
STAGES = ["M1a", "M1b", "M1c"]
PANEL_LETTERS = {"overall": "A", "M1a": "B", "M1b": "C", "M1c": "D"}
PANEL_SUBTITLES = {"overall": "Overall", "M1a": "M1a", "M1b": "M1b", "M1c": "M1c"}


def plot_panel(ax, panel_data, panel_key):
    time_column = "pca_months"
    event_column = "pca_event"
    for e in ERAS:
        subset = (
            panel_data[panel_data["era"] == e]
            .dropna(subset=[time_column, event_column])
            .copy()
        )
        if len(subset) == 0:
            continue
        mask = subset[time_column] > ANALYSIS_MONTHS
        subset.loc[mask, event_column] = 0
        subset.loc[mask, time_column] = ANALYSIS_MONTHS
        km_model = KaplanMeierFitter()
        km_model.fit(
            subset[time_column],
            event_observed=subset[event_column],
            label=ERA_LABELS[e],
        )
        km_model.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=ERA_COLORS[e],
            linewidth=2,
        )
    all_patients = (
        panel_data[panel_data["era"].isin(ERAS)]
        .dropna(subset=[time_column, event_column])
        .copy()
    )
    mask = all_patients[time_column] > ANALYSIS_MONTHS
    all_patients.loc[mask, event_column] = 0
    all_patients.loc[mask, time_column] = ANALYSIS_MONTHS
    if len(all_patients) > 0 and all_patients["era"].nunique() > 1:
        logrank_result = multivariate_logrank_test(
            all_patients[time_column], all_patients["era"], all_patients[event_column]
        )
        p_value = logrank_result.p_value
        p_text = "p < 0.0001" if p_value < 0.0001 else f"p = {p_value:.4f}"
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
    ax.set_xlabel("Time (Years)", fontsize=13)
    ax.set_ylabel("Proportion Surviving", fontsize=13)
    ax.text(
        -0.12,
        1.05,
        PANEL_LETTERS[panel_key],
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="bottom",
    )
    ax.set_title(
        PANEL_SUBTITLES[panel_key], fontsize=13, fontweight="bold", loc="center"
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x // 12)}"))
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(True, linestyle="--", alpha=0.4)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_prepare_data()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_panel(axes[0, 0], data, "overall")
    staged_data = data[data["m_stage"].isin(STAGES)]
    for ax, stage in zip([axes[0, 1], axes[1, 0], axes[1, 1]], STAGES):
        stage_data = staged_data[staged_data["m_stage"] == stage]
        plot_panel(ax, stage_data, stage)
    plt.tight_layout()
    output_path = FIGURE_DIR / "02_pcss_km.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

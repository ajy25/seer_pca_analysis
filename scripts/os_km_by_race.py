from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from overall_survival_data import MAIN_RACES, load_and_prepare_data

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "output" / "figures"
ANALYSIS_MONTHS = 24
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


def plot_race_panel(ax, data, time_column, event_column):
    kmf_list = []
    for race in MAIN_RACES:
        subset = (
            data[data["race"] == race].dropna(subset=[time_column, event_column]).copy()
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
            label=RACE_SHORT[race],
        )
        km_model.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=RACE_COLORS[race],
            linewidth=2,
        )
        kmf_list.append(km_model)

    all_patients = (
        data[data["race"].isin(MAIN_RACES)]
        .dropna(subset=[time_column, event_column])
        .copy()
    )
    mask = all_patients[time_column] > ANALYSIS_MONTHS
    all_patients.loc[mask, event_column] = 0
    all_patients.loc[mask, time_column] = ANALYSIS_MONTHS
    if len(all_patients) > 0 and all_patients["race"].nunique() > 1:
        logrank_result = multivariate_logrank_test(
            all_patients[time_column], all_patients["race"], all_patients[event_column]
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
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x // 12)}"))
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(True, linestyle="--", alpha=0.4)

    return kmf_list


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_and_prepare_data()
    df_modern = data[data["era"].isin([3, 4])].copy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plot_race_panel(ax, df_modern, "os_months", "os_event")
    plt.tight_layout()
    output_path = FIGURE_DIR / "03_os_km_by_race.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

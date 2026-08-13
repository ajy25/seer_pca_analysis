
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="whitegrid")
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from overall_survival_data import (
    ERA_COLORS,
    ERA_LABELS,
    load_and_prepare_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_LIMIT = 24
ERAS = [1, 2, 3, 4]
STAGES = ["M1a", "M1b", "M1c"]
PANEL_LETTERS = {"overall": "A", "M1a": "B", "M1b": "C", "M1c": "D"}
PANEL_SUBTITLES = {"overall": "Overall", "M1a": "M1a", "M1b": "M1b", "M1c": "M1c"}


def plot_panel(ax, df_panel, panel_key):
    time_col = "os_months"
    event_col = "os_event"
    for e in ERAS:
        sub = df_panel[df_panel["era"] == e].dropna(subset=[time_col, event_col]).copy()
        if len(sub) == 0:
            continue
        mask = sub[time_col] > X_LIMIT
        sub.loc[mask, event_col] = 0
        sub.loc[mask, time_col] = X_LIMIT
        kmf = KaplanMeierFitter()
        kmf.fit(sub[time_col], event_observed=sub[event_col], label=ERA_LABELS[e])
        kmf.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=ERA_COLORS[e],
            linewidth=2,
        )
    sub_all = (
        df_panel[df_panel["era"].isin(ERAS)].dropna(subset=[time_col, event_col]).copy()
    )
    mask = sub_all[time_col] > X_LIMIT
    sub_all.loc[mask, event_col] = 0
    sub_all.loc[mask, time_col] = X_LIMIT
    if len(sub_all) > 0 and sub_all["era"].nunique() > 1:
        lr = multivariate_logrank_test(
            sub_all[time_col], sub_all["era"], sub_all[event_col]
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
    df = load_and_prepare_data()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    plot_panel(axes[0, 0], df, "overall")
    df_staged = df[df["m_stage"].isin(STAGES)]
    for ax, stage in zip([axes[0, 1], axes[1, 0], axes[1, 1]], STAGES):
        df_s = df_staged[df_staged["m_stage"] == stage]
        plot_panel(ax, df_s, stage)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "01_os_km.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

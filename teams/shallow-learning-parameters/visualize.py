"""Generate the figures for parameters.md from the real Design Storm data.

Run from this directory: `python visualize.py`. Writes PNGs to figures/.
Matplotlib only (no seaborn), per the task's "favor sklearn" steer -- the
"advanced" pieces are PCA/KMeans/RandomForest from sklearn, not a plotting
library.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import (
    build_dataset,
    feature_columns,
    load_dwr_telemetry,
    load_snowpack,
    with_calendar_year_and_doy,
)
from models import (
    cluster_hydrologic_regimes,
    fit_linear_baseline,
    fit_random_forest_importance,
    fit_threshold_classifier,
)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

TOC_FEATURES = [
    "turb_flow",
    "roll_swe_7",
    "Specific_Cond_Mean",
    "roll_flow_7",
    "roll_precip_7",
    "roll_turb_3",
    "flow_delta",
    "month_sin",
    "month_cos",
    "Temp_C_Mean",
]
ALK_FEATURES = [
    "Specific_Cond_Mean",
    "pH_Median",
    "roll_swe_7",
    "roll_flow_7",
    "Flow_CFS",
    "roll_precip_7",
    "Temp_C_Mean",
    "month_sin",
]


def plot_targets_timeseries(df_toc: pd.DataFrame, out_path: Path) -> None:
    """Fig 1: TOC and alkalinity over time, raw and 7-day smoothed, on twin
    axes. Shows the spring-runoff TOC spikes and the winter data gap
    (guide.md section 4: no Jan-Mar rows)."""
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(df_toc.index, df_toc["TOC_mg_L"], color="tab:green", alpha=0.3, linewidth=0.8)
    ax1.plot(
        df_toc.index,
        df_toc["TOC_mg_L"].rolling(7).mean(),
        color="tab:green",
        linewidth=1.6,
        label="TOC (7-day mean)",
    )
    ax2.plot(df_toc.index, df_toc["Alk_mg_L"], color="tab:blue", alpha=0.25, linewidth=0.8)
    ax2.plot(
        df_toc.index,
        df_toc["Alk_mg_L"].rolling(7).mean(),
        color="tab:blue",
        linewidth=1.6,
        label="Alkalinity (7-day mean)",
    )

    ax1.set_ylabel("TOC (mg/L)", color="tab:green")
    ax2.set_ylabel("Alkalinity (mg/L)", color="tab:blue")
    ax1.set_title("Foothills influent: TOC and alkalinity, 2022-2026")
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.88))
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_correlation_heatmap(df_toc: pd.DataFrame, df_alk: pd.DataFrame, all_features: list[str], out_path: Path) -> None:
    """Fig 2: how strongly every engineered predictor correlates with each
    target, at the lag actually used for that target."""
    toc_corr = df_toc[all_features].corrwith(df_toc["TOC_mg_L"])
    alk_corr = df_alk[all_features].corrwith(df_alk["Alk_mg_L"])
    matrix = np.vstack([toc_corr.values, alk_corr.values])

    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks([0, 1], labels=["TOC_mg_L (lag=2d)", "Alk_mg_L (lag=4d)"])
    ax.set_xticks(range(len(all_features)), labels=all_features, rotation=60, ha="right")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Predictor correlation with each target")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_turb_flow_scatter(df_toc: pd.DataFrame, out_path: Path) -> None:
    """Fig 3: the single strongest engineered TOC predictor (guide.md
    section 7) against TOC, with its linear fit and held-out R^2."""
    model, r2 = fit_linear_baseline(df_toc, "turb_flow", "TOC_mg_L")
    clean = df_toc.dropna(subset=["turb_flow", "TOC_mg_L"])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(clean["turb_flow"], clean["TOC_mg_L"], s=10, alpha=0.35, color="tab:green")
    x_line = np.linspace(clean["turb_flow"].min(), clean["turb_flow"].max(), 100)
    y_line = model.predict(pd.DataFrame({"turb_flow": x_line}))
    ax.plot(x_line, y_line, color="black", linewidth=1.5, label=f"linear fit (held-out R^2={r2:.2f})")
    ax.set_xlabel("turb_flow = turbidity x flow, 2 days earlier")
    ax.set_ylabel("TOC (mg/L)")
    ax.set_title("Turbidity-weighted flow vs. TOC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_feature_importance(df_toc: pd.DataFrame, df_alk: pd.DataFrame, out_path: Path) -> None:
    """Fig 4: RandomForestRegressor feature importances, TOC vs. alkalinity,
    side by side."""
    toc_result = fit_random_forest_importance(df_toc, TOC_FEATURES, "TOC_mg_L")
    alk_result = fit_random_forest_importance(df_alk, ALK_FEATURES, "Alk_mg_L")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, result, title in [
        (axes[0], toc_result, f"TOC_mg_L (held-out R^2={toc_result.r2:.2f})"),
        (axes[1], alk_result, f"Alk_mg_L (held-out R^2={alk_result.r2:.2f})"),
    ]:
        ordered = result.importances.sort_values()
        ax.barh(ordered.index, ordered.values, color="tab:purple")
        ax.set_title(title)
        ax.set_xlabel("importance")
    fig.suptitle("Random forest feature importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_regime_clusters(df_toc: pd.DataFrame, all_features: list[str], out_path: Path) -> None:
    """Fig 5: unsupervised discovery. Standardize every engineered feature,
    project to 2D with PCA, color by KMeans cluster -- a first pass at
    Scenario 3's "hydrologic regimes" without using either target."""
    clean, clusters = cluster_hydrologic_regimes(df_toc, all_features, n_clusters=3)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    scatter = ax.scatter(
        clusters.coordinates[:, 0],
        clusters.coordinates[:, 1],
        c=clusters.labels,
        cmap="viridis",
        s=14,
        alpha=0.7,
    )
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_title("Hydrologic-regime clusters (KMeans on standardized features)")
    legend = ax.legend(*scatter.legend_elements(), title="cluster")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_alkalinity_classifier_curve(df_alk: pd.DataFrame, out_path: Path) -> None:
    """Fig 6: precision/recall tradeoff for "alkalinity below 60", the
    classifier guide.md section 11 covers only in prose."""
    result = fit_threshold_classifier(df_alk, ALK_FEATURES, "Alk_mg_L", threshold=60.0)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(result.recall, result.precision, color="tab:red", linewidth=1.8)
    ax.set_xlabel("recall (share of true lows caught)")
    ax.set_ylabel("precision (share of low-alarms correct)")
    ax.set_title(f"Alkalinity < 60 mg/L classifier (ROC-AUC={result.roc_auc:.2f})")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_snowpack_and_flow_by_year(data_dir: Path, out_path: Path) -> None:
    """Fig 7 (Scenario 3): Hoosier Pass SWE and South Platte flow overlaid by
    calendar year on a shared day-of-year axis, so wetter and drier years are
    directly comparable -- the "drought vs. wet years" framing from the deck's
    Scenario 3."""
    snow = with_calendar_year_and_doy(load_snowpack(data_dir))
    flow = with_calendar_year_and_doy(load_dwr_telemetry(data_dir))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for year, group in snow.groupby("year"):
        axes[0].plot(group["day_of_year"], group["SWE"], label=str(year), linewidth=1.3)
    axes[0].set_title("Hoosier Pass SWE by day of year")
    axes[0].set_xlabel("day of year")
    axes[0].set_ylabel("SWE (in)")
    axes[0].legend(fontsize=8)

    for year, group in flow.groupby("year"):
        axes[1].plot(group["day_of_year"], group["Flow_CFS"], label=str(year), linewidth=1.3)
    axes[1].set_title("South Platte flow by day of year")
    axes[1].set_xlabel("day of year")
    axes[1].set_ylabel("Flow (CFS)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Year-over-year comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    df_toc = build_dataset(DATA_DIR, lag_days=2)
    df_alk = build_dataset(DATA_DIR, lag_days=4)
    all_features = feature_columns(df_toc)

    plot_targets_timeseries(df_toc, FIGURES_DIR / "01_targets_timeseries.png")
    plot_correlation_heatmap(df_toc, df_alk, all_features, FIGURES_DIR / "02_correlation_heatmap.png")
    plot_turb_flow_scatter(df_toc, FIGURES_DIR / "03_turb_flow_vs_toc.png")
    plot_feature_importance(df_toc, df_alk, FIGURES_DIR / "04_feature_importance.png")
    plot_regime_clusters(df_toc, all_features, FIGURES_DIR / "05_hydrologic_regimes.png")
    plot_alkalinity_classifier_curve(df_alk, FIGURES_DIR / "06_alkalinity_classifier_pr_curve.png")
    plot_snowpack_and_flow_by_year(DATA_DIR, FIGURES_DIR / "07_snowpack_streamflow_by_year.png")
    print(f"Wrote 7 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

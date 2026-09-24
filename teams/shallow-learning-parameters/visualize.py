"""Generate the figures for parameters.md from the real Design Storm data.

Run from this directory: `python visualize.py`. Writes PNGs to figures/.
Matplotlib only (no seaborn), per the task's "favor sklearn" steer -- the
"advanced" pieces are PCA/KMeans/RandomForest from sklearn, not a plotting
library. Figures 01-06 and 11-12 (Scenario 1) live here; 07-10 (Scenario 3
transit-time tracing) are in visualize_transit.py and 13-16 (Strontia
sonde) are in visualize_sonde.py -- split out to stay under this repo's
file-length gate. This file's `main()` is still the single entry point.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import build_dataset, feature_columns, peak_loading_date
from models import (
    cluster_hydrologic_regimes,
    fit_gradient_boosting_importance,
    fit_linear_baseline,
    fit_logistic_baseline,
    fit_random_forest_importance,
    fit_svr_baseline,
    fit_threshold_classifier,
)
from sonde_loader import cast_summary, daily_surface_features, load_sonde_readings
from visualize_advanced import (
    plot_flow_forecast,
    plot_gaussian_process,
    plot_multioutput_comparison,
    plot_quantile_bands,
    plot_sarimax,
    plot_tuning_comparison,
)
from visualize_anomaly import plot_anomaly_detection
from visualize_sonde import (
    plot_depth_profiles,
    plot_sonde_vs_gage_comparison,
    plot_storm_profile_comparison,
    plot_stratification_timeline,
)
from visualize_transit import (
    plot_lag_correlation_scan,
    plot_snowpack_and_flow_by_year,
    plot_transit_animation,
    plot_transit_event_trace,
    select_storm_event,
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


def plot_model_family_comparison(df_toc: pd.DataFrame, df_alk: pd.DataFrame, out_path: Path) -> None:
    """Fig 11 (Scenario 1): held-out R^2 across model families -- linear,
    random forest, SVR, and gradient boosting (deck slide 9: "try different
    models like support-vector machines"; gradient boosting because
    guide.md's own comparison shows Jake's CatBoost beating his random
    forest, and this catalog otherwise never tested a boosted-tree family)."""
    _, toc_linear_r2 = fit_linear_baseline(df_toc, "turb_flow", "TOC_mg_L")
    toc_rf_r2 = fit_random_forest_importance(df_toc, TOC_FEATURES, "TOC_mg_L").r2
    toc_svr_r2 = fit_svr_baseline(df_toc, TOC_FEATURES, "TOC_mg_L").r2
    toc_gbr_r2 = fit_gradient_boosting_importance(df_toc, TOC_FEATURES, "TOC_mg_L").r2

    alk_rf_r2 = fit_random_forest_importance(df_alk, ALK_FEATURES, "Alk_mg_L").r2
    alk_svr_r2 = fit_svr_baseline(df_alk, ALK_FEATURES, "Alk_mg_L").r2
    alk_gbr_r2 = fit_gradient_boosting_importance(df_alk, ALK_FEATURES, "Alk_mg_L").r2

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    axes[0].bar(
        ["Linear\n(turb_flow)", "Random\nforest", "SVR\n(RBF)", "Gradient\nboosting"],
        [toc_linear_r2, toc_rf_r2, toc_svr_r2, toc_gbr_r2],
        color=["tab:gray", "tab:purple", "tab:red", "tab:orange"],
    )
    axes[0].set_title("TOC_mg_L")
    axes[0].set_ylabel("held-out R^2")
    axes[0].axhline(0, color="black", linewidth=0.8)

    axes[1].bar(
        ["Random\nforest", "SVR\n(RBF)", "Gradient\nboosting"],
        [alk_rf_r2, alk_svr_r2, alk_gbr_r2],
        color=["tab:purple", "tab:red", "tab:orange"],
    )
    axes[1].set_title("Alk_mg_L")
    axes[1].axhline(0, color="black", linewidth=0.8)

    fig.suptitle("Model family comparison (held-out R^2, same split/features)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_classifier_comparison(df_alk: pd.DataFrame, out_path: Path) -> None:
    """Fig 17 (Scenario 1): the alkalinity-below-60 classifier, random
    forest vs. a logistic regression baseline (`LogisticRegression` was
    imported in models.py but never actually used before this comparison).
    Same held-out split, precision/recall curves overlaid."""
    rf_result = fit_threshold_classifier(df_alk, ALK_FEATURES, "Alk_mg_L", threshold=60.0)
    logistic_result = fit_logistic_baseline(df_alk, ALK_FEATURES, "Alk_mg_L", threshold=60.0)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(
        rf_result.recall, rf_result.precision, color="tab:purple",
        label=f"Random forest (ROC-AUC={rf_result.roc_auc:.2f})",
    )
    ax.plot(
        logistic_result.recall, logistic_result.precision, color="tab:orange",
        label=f"Logistic regression (ROC-AUC={logistic_result.roc_auc:.2f})",
    )
    ax.set_xlabel("recall (share of true lows caught)")
    ax.set_ylabel("precision (share of low-alarms correct)")
    ax.set_title("Alkalinity < 60 mg/L: random forest vs. logistic regression")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_alkalinity_roc_curve(df_alk: pd.DataFrame, out_path: Path) -> None:
    """Fig 25 (Scenario 1): the actual ROC curve (true positive rate vs.
    false positive rate) for the alkalinity-below-60 classifier, random
    forest vs. logistic regression. Figs 06/17 plot precision/recall with
    the ROC-AUC number in the title/legend; this is the literal ROC
    diagram behind that number, including the y=x random-classifier
    reference line."""
    rf_result = fit_threshold_classifier(df_alk, ALK_FEATURES, "Alk_mg_L", threshold=60.0)
    logistic_result = fit_logistic_baseline(df_alk, ALK_FEATURES, "Alk_mg_L", threshold=60.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(
        rf_result.fpr, rf_result.tpr, color="tab:purple",
        label=f"Random forest (ROC-AUC={rf_result.roc_auc:.3f})",
    )
    ax.plot(
        logistic_result.fpr, logistic_result.tpr, color="tab:orange",
        label=f"Logistic regression (ROC-AUC={logistic_result.roc_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, label="Random classifier")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Alkalinity < 60 mg/L classifier: ROC curve (held-out split)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_lag_day_grid_search(data_dir: Path, out_path: Path, lag_days_grid: list[int] | None = None) -> None:
    """Fig 12 (Scenario 1): held-out R^2 vs. lag_days, same random forest and
    feature list rebuilt at each lag (deck slide 9: "try different...
    lag-times")."""
    if lag_days_grid is None:
        lag_days_grid = [0, 1, 2, 3, 4, 5, 7, 10]

    toc_scores = []
    alk_scores = []
    for lag in lag_days_grid:
        df = build_dataset(data_dir, lag_days=lag)
        toc_scores.append(fit_random_forest_importance(df, TOC_FEATURES, "TOC_mg_L").r2)
        alk_scores.append(fit_random_forest_importance(df, ALK_FEATURES, "Alk_mg_L").r2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(lag_days_grid, toc_scores, marker="o", color="tab:green", label="TOC_mg_L")
    ax.plot(lag_days_grid, alk_scores, marker="o", color="tab:blue", label="Alk_mg_L")
    ax.axvline(2, color="tab:green", linestyle=":", alpha=0.5)
    ax.axvline(4, color="tab:blue", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("lag_days used to shift every predictor")
    ax.set_ylabel("held-out R^2 (random forest)")
    ax.set_title("Lag-day grid search (dotted lines: lags used elsewhere in this catalog)")
    ax.legend()
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
    plot_lag_correlation_scan(DATA_DIR, FIGURES_DIR / "08_lag_correlation_scan.png")

    event = select_storm_event(DATA_DIR)
    plot_transit_event_trace(event, FIGURES_DIR / "09_storm_event_trace.png")
    plot_transit_animation(event, FIGURES_DIR / "10_transit_animation.gif")

    plot_model_family_comparison(df_toc, df_alk, FIGURES_DIR / "11_model_family_comparison.png")
    plot_lag_day_grid_search(DATA_DIR, FIGURES_DIR / "12_lag_day_grid_search.png")

    sonde_readings = load_sonde_readings(DATA_DIR)
    sonde_casts = cast_summary(sonde_readings)
    sonde_daily = daily_surface_features(sonde_readings)

    plot_stratification_timeline(sonde_casts, FIGURES_DIR / "13_stratification_timeline.png")
    plot_depth_profiles(sonde_readings, sonde_casts, FIGURES_DIR / "14_depth_profiles.png")

    sonde_storm_date = peak_loading_date(
        DATA_DIR, sonde_readings["timestamp"].min().normalize(), sonde_readings["timestamp"].max().normalize()
    )
    plot_storm_profile_comparison(
        sonde_readings, sonde_casts, sonde_storm_date, FIGURES_DIR / "15_storm_profile_comparison.png"
    )
    plot_sonde_vs_gage_comparison(
        DATA_DIR, sonde_readings, sonde_daily, FIGURES_DIR / "16_sonde_vs_gage_comparison.png"
    )

    plot_classifier_comparison(df_alk, FIGURES_DIR / "17_classifier_comparison.png")
    plot_anomaly_detection(DATA_DIR, FIGURES_DIR / "18_anomaly_detection.png")

    plot_tuning_comparison(df_toc, TOC_FEATURES, df_alk, ALK_FEATURES, FIGURES_DIR / "19_tuning_comparison.png")
    plot_flow_forecast(DATA_DIR, FIGURES_DIR / "20_flow_forecast.png")
    plot_quantile_bands(df_toc, TOC_FEATURES, df_alk, ALK_FEATURES, FIGURES_DIR / "21_quantile_bands.png")
    plot_multioutput_comparison(
        df_toc, TOC_FEATURES, ALK_FEATURES, all_features, FIGURES_DIR / "22_multioutput_comparison.png"
    )
    plot_gaussian_process(df_toc, TOC_FEATURES, df_alk, ALK_FEATURES, FIGURES_DIR / "23_gaussian_process.png")
    plot_sarimax(df_toc, df_alk, FIGURES_DIR / "24_sarimax.png")
    plot_alkalinity_roc_curve(df_alk, FIGURES_DIR / "25_alkalinity_roc_curve.png")

    print(f"Wrote 24 figures + 1 animation to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

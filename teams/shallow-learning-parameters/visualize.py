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
    load_target,
    load_usgs_gage,
    with_calendar_year_and_doy,
)
from models import (
    best_lag,
    cluster_hydrologic_regimes,
    fit_linear_baseline,
    fit_random_forest_importance,
    fit_svr_baseline,
    fit_threshold_classifier,
    lag_correlation_scan,
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


def plot_lag_correlation_scan(data_dir: Path, out_path: Path, max_lag_days: int = 10) -> None:
    """Fig 8 (Scenario 3): correlation vs. lag for the predictors behind each
    target -- the empirical version of "how long does a signal take to
    arrive", in place of assuming Jake's fixed 2/4-day lags. Dotted vertical
    lines mark each predictor's best-correlated lag."""
    gage = load_usgs_gage(data_dir)
    telemetry = load_dwr_telemetry(data_dir)
    target = load_target(data_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    toc_pairs = [
        ("Turbidity_Median", gage["Turbidity_Median"], "tab:brown"),
        ("Flow_CFS", telemetry["Flow_CFS"], "tab:blue"),
    ]
    for name, series, color in toc_pairs:
        scan = lag_correlation_scan(series, target["TOC_mg_L"], max_lag_days)
        lag = best_lag(scan)
        axes[0].plot(scan.index, scan.values, marker="o", color=color, label=f"{name} (best={lag}d)")
        axes[0].axvline(lag, color=color, linestyle=":", alpha=0.6)
    axes[0].set_title("TOC_mg_L")
    axes[0].set_xlabel("lag (days)")
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(fontsize=8)

    alk_pairs = [
        ("Specific_Cond_Mean", gage["Specific_Cond_Mean"], "tab:purple"),
        ("pH_Median", gage["pH_Median"], "tab:orange"),
    ]
    for name, series, color in alk_pairs:
        scan = lag_correlation_scan(series, target["Alk_mg_L"], max_lag_days)
        lag = best_lag(scan)
        axes[1].plot(scan.index, scan.values, marker="o", color=color, label=f"{name} (best={lag}d)")
        axes[1].axvline(lag, color=color, linestyle=":", alpha=0.6)
    axes[1].set_title("Alk_mg_L")
    axes[1].set_xlabel("lag (days)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Empirical transit-time lag scan (statistical fit, not a measured travel time)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


class StormEvent:
    """One real storm window, used by both the static event-trace figure and
    the transit animation so they tell the same story."""

    def __init__(
        self,
        event_date: pd.Timestamp,
        lag_days: int,
        turbidity: pd.Series,
        flow: pd.Series,
        toc: pd.Series,
    ) -> None:
        self.event_date = event_date
        self.lag_days = lag_days
        self.turbidity = turbidity
        self.flow = flow
        self.toc = toc


def select_storm_event(data_dir: Path, window_before: int = 5, window_after: int = 20) -> StormEvent:
    """Pick the real day with the highest raw turbidity x flow ("loading")
    above Strontia Springs, and window the raw upstream/downstream series
    around it -- one concrete, real event to trace end to end, rather than
    an aggregate correlation."""
    gage = load_usgs_gage(data_dir)
    telemetry = load_dwr_telemetry(data_dir)
    target = load_target(data_dir)

    raw_loading = (gage["Turbidity_Median"] * telemetry["Flow_CFS"]).dropna()
    event_date = raw_loading.idxmax()
    lag = best_lag(lag_correlation_scan(gage["Turbidity_Median"], target["TOC_mg_L"], max_lag_days=10))

    start = event_date - pd.Timedelta(days=window_before)
    end = event_date + pd.Timedelta(days=window_after)
    return StormEvent(
        event_date=event_date,
        lag_days=lag,
        turbidity=gage["Turbidity_Median"].loc[start:end],
        flow=telemetry["Flow_CFS"].loc[start:end],
        toc=target["TOC_mg_L"].loc[start:end],
    )


def plot_transit_event_trace(event: StormEvent, out_path: Path) -> None:
    """Fig 9 (Scenario 3): one real storm, traced end to end -- the upstream
    turbidity/flow spike above Strontia, and the TOC response at Foothills
    the empirically-fit lag_days later."""
    fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    ax_up.plot(event.turbidity.index, event.turbidity.values, color="tab:brown", label="Turbidity_Median")
    ax_up_flow = ax_up.twinx()
    ax_up_flow.plot(event.flow.index, event.flow.values, color="tab:blue", alpha=0.5, label="Flow_CFS")
    ax_up.axvline(event.event_date, color="black", linestyle="--", linewidth=1)
    ax_up.set_ylabel("Turbidity (NTU)", color="tab:brown")
    ax_up_flow.set_ylabel("Flow (CFS)", color="tab:blue")
    ax_up.set_title(f"Upstream gage above Strontia Springs -- peak loading {event.event_date.date()}")

    response_date = event.event_date + pd.Timedelta(days=event.lag_days)
    ax_down.plot(event.toc.index, event.toc.values, color="tab:green", label="TOC_mg_L")
    ax_down.axvline(response_date, color="black", linestyle="--", linewidth=1)
    ax_down.set_ylabel("TOC (mg/L)", color="tab:green")
    ax_down.set_title(f"Foothills influent -- empirical {event.lag_days}-day response ({response_date.date()})")
    ax_down.set_xlabel("date")

    fig.suptitle("Tracing one real storm from the upstream gage to the treatment plant")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_transit_animation(event: StormEvent, out_path: Path) -> None:
    """Animated GIF: a cursor sweeps day by day across the same storm window,
    synchronized between the upstream gage (top) and the Foothills lab
    result (bottom) -- a single unified view of the signal moving through
    the system, in place of literally animating a water parcel (which this
    data cannot ground -- see guide.md section 1 on lag vs. travel time)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    dates = event.turbidity.index.union(event.toc.index).sort_values().unique()
    if len(dates) == 0:
        return

    fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax_up.plot(event.turbidity.index, event.turbidity.values, color="tab:brown")
    ax_up.set_ylabel("Turbidity (NTU)")
    ax_up.set_title("Upstream gage above Strontia Springs")
    ax_down.plot(event.toc.index, event.toc.values, color="tab:green")
    ax_down.set_ylabel("TOC (mg/L)")
    ax_down.set_title(f"Foothills influent (~{event.lag_days}-day empirical lag)")
    ax_down.set_xlabel("date")

    cursor_up = ax_up.axvline(dates[0], color="black", linewidth=1.5)
    cursor_down = ax_down.axvline(dates[0], color="black", linewidth=1.5)
    fig.tight_layout()

    def update(frame_date: pd.Timestamp):
        cursor_up.set_xdata([frame_date, frame_date])
        cursor_down.set_xdata([frame_date, frame_date])
        return cursor_up, cursor_down

    anim = FuncAnimation(fig, update, frames=list(dates), interval=150, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=6))
    plt.close(fig)


def plot_model_family_comparison(df_toc: pd.DataFrame, df_alk: pd.DataFrame, out_path: Path) -> None:
    """Fig 11 (Scenario 1): held-out R^2 across model families -- linear,
    random forest, and SVR (deck slide 9: "try different models like
    support-vector machines")."""
    _, toc_linear_r2 = fit_linear_baseline(df_toc, "turb_flow", "TOC_mg_L")
    toc_rf_r2 = fit_random_forest_importance(df_toc, TOC_FEATURES, "TOC_mg_L").r2
    toc_svr_r2 = fit_svr_baseline(df_toc, TOC_FEATURES, "TOC_mg_L").r2

    alk_rf_r2 = fit_random_forest_importance(df_alk, ALK_FEATURES, "Alk_mg_L").r2
    alk_svr_r2 = fit_svr_baseline(df_alk, ALK_FEATURES, "Alk_mg_L").r2

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].bar(
        ["Linear\n(turb_flow)", "Random\nforest", "SVR\n(RBF)"],
        [toc_linear_r2, toc_rf_r2, toc_svr_r2],
        color=["tab:gray", "tab:purple", "tab:red"],
    )
    axes[0].set_title("TOC_mg_L")
    axes[0].set_ylabel("held-out R^2")
    axes[0].axhline(0, color="black", linewidth=0.8)

    axes[1].bar(["Random\nforest", "SVR\n(RBF)"], [alk_rf_r2, alk_svr_r2], color=["tab:purple", "tab:red"])
    axes[1].set_title("Alk_mg_L")
    axes[1].axhline(0, color="black", linewidth=0.8)

    fig.suptitle("Model family comparison (held-out R^2, same split/features)")
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

    print(f"Wrote 11 figures + 1 animation to {FIGURES_DIR}")


if __name__ == "__main__":
    main()

"""Figures 13-16: the Strontia profiling sonde (stratification, depth
profiles, storm impact, sonde-vs-gage comparison). Split out of
visualize.py to keep files under the repo's file-length gate; see
visualize.py for the entry point (`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import load_target, load_usgs_gage
from models import best_lag, lag_correlation_scan
from sonde_loader import assign_cast_ids, full_depth_casts


def plot_stratification_timeline(casts: pd.DataFrame, out_path: Path) -> None:
    """Fig 13 (Scenario 3): surface-minus-bottom temperature per cast across
    the sonde's season -- large values mean a stratified water column,
    values near zero mean fully mixed (the "lake turnover" bullet)."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(casts["date"], casts["temp_diff_c"], color="tab:cyan", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)

    most_mixed = casts.loc[casts["temp_diff_c"].idxmin()]
    most_stratified = casts.loc[casts["temp_diff_c"].idxmax()]
    ax.scatter([most_mixed["date"]], [most_mixed["temp_diff_c"]], color="tab:blue", zorder=5, label="most mixed")
    ax.scatter(
        [most_stratified["date"]], [most_stratified["temp_diff_c"]], color="tab:red", zorder=5, label="most stratified"
    )

    ax.set_ylabel("surface - bottom temperature (C)")
    ax.set_title("Strontia Springs Reservoir stratification, by cast")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_depth_profiles(readings: pd.DataFrame, casts: pd.DataFrame, out_path: Path, n_profiles: int = 4) -> None:
    """Fig 14 (Scenario 2): a handful of real depth profiles spread across
    the season, showing how temperature and turbidity vary with depth and
    how that shape changes as the season progresses. Restricted to
    full-depth casts so a short/aborted cast doesn't appear as a shallow
    reservoir."""
    readings = readings.assign(cast_id=assign_cast_ids(readings))
    casts = full_depth_casts(casts)
    positions = np.linspace(0, len(casts) - 1, n_profiles).astype(int)
    chosen_ids = casts.index[positions]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    cmap = plt.get_cmap("viridis")
    for i, cast_id in enumerate(chosen_ids):
        cast_readings = readings[readings["cast_id"] == cast_id].sort_values("Depth_m")
        color = cmap(i / max(len(chosen_ids) - 1, 1))
        label = cast_readings["timestamp"].iloc[0].strftime("%Y-%m-%d")
        axes[0].plot(cast_readings["Temp_C"], cast_readings["Depth_m"], color=color, marker="o", markersize=3, label=label)
        axes[1].plot(cast_readings["Turbidity_NTU"], cast_readings["Depth_m"], color=color, marker="o", markersize=3)

    axes[0].invert_yaxis()
    axes[0].set_xlabel("Temperature (C)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("Temperature")
    axes[0].legend(fontsize=7, title="cast date")
    axes[1].set_xlabel("Turbidity (NTU)")
    axes[1].set_title("Turbidity")

    fig.suptitle("Strontia Springs Reservoir: depth profiles across the season")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_storm_profile_comparison(
    readings: pd.DataFrame, casts: pd.DataFrame, storm_date: pd.Timestamp, out_path: Path
) -> None:
    """Fig 15 (Scenario 2): full depth profiles immediately before and after
    a real storm, not just the surface/bottom summary numbers -- does the
    storm's signal reach every depth, or only the surface? Restricted to
    full-depth casts (see `full_depth_casts`) for a fair before/after
    comparison."""
    readings = readings.assign(cast_id=assign_cast_ids(readings))
    casts = full_depth_casts(casts)
    before_mask = casts["date"] < storm_date
    after_mask = casts["date"] > storm_date
    before_id = casts.index[before_mask][-1]
    after_id = casts.index[after_mask][0]
    before_start = casts.loc[before_id, "start"]
    after_start = casts.loc[after_id, "start"]

    before_readings = readings[readings["cast_id"] == before_id].sort_values("Depth_m")
    after_readings = readings[readings["cast_id"] == after_id].sort_values("Depth_m")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    axes[0].plot(
        before_readings["Turbidity_NTU"], before_readings["Depth_m"], color="tab:blue", marker="o", markersize=3,
        label=f"before ({before_start.date()})",
    )
    axes[0].plot(
        after_readings["Turbidity_NTU"], after_readings["Depth_m"], color="tab:red", marker="o", markersize=3,
        label=f"after ({after_start.date()})",
    )
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Turbidity (NTU)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_title("Turbidity")
    axes[0].legend(fontsize=8)

    axes[1].plot(before_readings["Temp_C"], before_readings["Depth_m"], color="tab:blue", marker="o", markersize=3)
    axes[1].plot(after_readings["Temp_C"], after_readings["Depth_m"], color="tab:red", marker="o", markersize=3)
    axes[1].set_xlabel("Temperature (C)")
    axes[1].set_title("Temperature")

    fig.suptitle(f"Depth profile before/after the {storm_date.date()} storm")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sonde_vs_gage_comparison(
    data_dir: Path, sonde_readings: pd.DataFrame, daily_sonde: pd.DataFrame, out_path: Path
) -> None:
    """Fig 16 (Scenario 1): does the closer sonde predict better than the
    upstream gage, on the same window? Correlation-at-best-lag for both
    sensors, both targets."""
    start = sonde_readings["timestamp"].min().normalize()
    end = sonde_readings["timestamp"].max().normalize()
    gage = load_usgs_gage(data_dir).loc[start:end]
    target = load_target(data_dir).loc[start:end]

    toc_sonde = lag_correlation_scan(daily_sonde["Turbidity_NTU"], target["TOC_mg_L"], 10)
    toc_gage = lag_correlation_scan(gage["Turbidity_Median"], target["TOC_mg_L"], 10)
    alk_sonde = lag_correlation_scan(daily_sonde["Conductivity"], target["Alk_mg_L"], 10)
    alk_gage = lag_correlation_scan(gage["Specific_Cond_Mean"], target["Alk_mg_L"], 10)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].bar(
        ["sonde\n(turbidity)", "gage\n(turbidity)"],
        [toc_sonde[best_lag(toc_sonde)], toc_gage[best_lag(toc_gage)]],
        color=["tab:orange", "tab:blue"],
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("TOC_mg_L")
    axes[0].set_ylabel("correlation at best lag")

    axes[1].bar(
        ["sonde\n(corr. w/ conductivity)", "gage\n(corr. w/ conductivity)"],
        [alk_sonde[best_lag(alk_sonde)], alk_gage[best_lag(alk_gage)]],
        color=["tab:orange", "tab:blue"],
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Alk_mg_L")
    axes[1].set_ylabel("correlation at best lag")  # bars are r, not raw conductivity -- a negative bar is a valid r, not a sensor error

    fig.suptitle(f"Sonde vs. upstream gage, same window ({start.date()} to {end.date()})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

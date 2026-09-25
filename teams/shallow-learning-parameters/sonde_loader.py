"""Load the Strontia Springs Reservoir profiling sonde (`Strontia 0407_0819.xlsx`).

Unlike the daily CSVs in data_loader.py, this instrument takes a depth
profile ("cast") every ~6 hours: it lowers through the water column,
recording a reading roughly every meter. There is no `DATE` column to
join on directly -- a cast has a timestamp and ~48 rows, one per depth.

This module turns that into two things build_dataset-style code can use:
- `cast_summary`: one row per cast (surface/bottom values, a stratification
  metric), for time-series analysis (guide.md never uses this data at all).
- `daily_surface_features`: one row per calendar day with a cast, averaging
  every cast that day's near-surface reading -- a daily series alignable
  with the Foothills lab results, the way the upstream USGS gage already is.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_target

COLUMN_RENAME = {
    "Time stamp": "timestamp",
    "Temp C": "Temp_C",
    "Conductivity ": "Conductivity",
    "Vertical Position ": "Depth_m",
    "pH": "pH",
    "ORP mV": "ORP_mV",
    "Turbidity NTU": "Turbidity_NTU",
    "Chl ug/L": "Chl_ugL",
    "Phycocyanin ": "Phycocyanin",
    "ODO & sat": "ODO_sat",
    "ODO mg/L": "ODO_mgL",
}

SURFACE_DEPTH_M = 3.0  # readings shallower than this count as "near-surface"


def load_sonde_readings(data_dir: Path) -> pd.DataFrame:
    """Every raw depth reading, sorted by time. Column names cleaned up
    (the source file has trailing spaces on several headers)."""
    path = Path(data_dir) / "Strontia 0407_0819.xlsx"
    df = pd.read_excel(path)
    df = df.rename(columns=COLUMN_RENAME)
    return df.sort_values("timestamp").reset_index(drop=True)


def assign_cast_ids(df: pd.DataFrame, gap_minutes: float = 60.0) -> pd.Series:
    """Group consecutive readings into casts: a new cast starts whenever the
    gap to the previous reading exceeds `gap_minutes` (a single cast's ~48
    readings are seconds to minutes apart; casts themselves are hours apart).
    Returns a 0-indexed cast id per row, same order as `df`."""
    gap = df["timestamp"].diff().dt.total_seconds().div(60)
    return (gap.isna() | (gap > gap_minutes)).cumsum() - 1


def cast_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cast: start time, calendar date, reading count, depth
    range, and surface/bottom temperature with their difference -- a simple,
    standard stratification indicator (large difference = stratified water
    column; near zero = mixed, i.e. turned over)."""
    working = df.assign(cast_id=assign_cast_ids(df))

    def _summarize(group: pd.DataFrame) -> pd.Series:
        ordered = group.sort_values("Depth_m")
        surface = ordered.iloc[0]
        bottom = ordered.iloc[-1]
        return pd.Series(
            {
                "start": group["timestamp"].min(),
                "date": group["timestamp"].min().normalize(),
                "n_readings": len(group),
                "depth_min_m": ordered["Depth_m"].min(),
                "depth_max_m": ordered["Depth_m"].max(),
                "surface_temp_c": surface["Temp_C"],
                "bottom_temp_c": bottom["Temp_C"],
                "temp_diff_c": surface["Temp_C"] - bottom["Temp_C"],
                "surface_turbidity_ntu": surface["Turbidity_NTU"],
                "surface_conductivity": surface["Conductivity"],
                "surface_ph": surface["pH"],
            }
        )

    return working.groupby("cast_id").apply(_summarize, include_groups=False)


def daily_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day that had at least one cast: the mean of
    every near-surface (depth <= SURFACE_DEPTH_M) reading that day, across
    all of that day's casts. This is the sonde's equivalent of the USGS
    gage's daily columns in data_loader.py -- alignable with the Foothills
    lab results by date."""
    near_surface = df[df["Depth_m"] <= SURFACE_DEPTH_M]
    daily = near_surface.groupby(near_surface["timestamp"].dt.normalize()).agg(
        Temp_C=("Temp_C", "mean"),
        Conductivity=("Conductivity", "mean"),
        pH=("pH", "mean"),
        Turbidity_NTU=("Turbidity_NTU", "mean"),
        Chl_ugL=("Chl_ugL", "mean"),
        ODO_mgL=("ODO_mgL", "mean"),
    )
    daily.index.name = "DATE"
    return daily


def full_depth_casts(casts: pd.DataFrame, min_readings: int = 20) -> pd.DataFrame:
    """Casts with at least `min_readings` depth readings -- excludes short
    or aborted casts (e.g. a handful of readings down to only a few metres)
    that would otherwise look like a shallow water column rather than an
    incomplete one. Use this before comparing profiles across casts."""
    return casts[casts["n_readings"] >= min_readings]


def engineer_sonde_features(daily: pd.DataFrame, casts: pd.DataFrame) -> pd.DataFrame:
    """Rolling-window versions of the sonde's daily surface features, plus a
    daily stratification signal from the casts -- the sonde equivalent of
    data_loader.engineer_predictor_features, so a sonde-only feature set can
    be built and compared against the national-dataset feature set on equal
    footing (rolling context, not just one raw daily value per column)."""
    features = daily.copy()
    features["roll_turbidity_3"] = daily["Turbidity_NTU"].rolling(3).mean()
    features["roll_conductivity_3"] = daily["Conductivity"].rolling(3).mean()

    stratification = casts.groupby("date")["temp_diff_c"].mean()
    stratification.index.name = "DATE"
    return features.join(stratification)


def build_sonde_dataset(data_dir: Path, lag_days: int = 2) -> pd.DataFrame:
    """Sonde equivalent of data_loader.build_dataset: Foothills lab results
    left-joined with sonde-derived features shifted `lag_days` forward.

    Feature rows only exist within the sonde's own coverage window
    (2026-04-07 to 2026-08-19, ~4 months) -- callers answering Scenario 1's
    "introduce the sonde" bullet on that limited window should restrict the
    returned frame's rows to that range before fitting anything, the same
    way `analyze_sonde._sonde_predictor_table` already restricts the gage
    comparison to a fair, matching window."""
    target = load_target(data_dir)
    readings = load_sonde_readings(data_dir)
    casts = cast_summary(readings)
    daily = daily_surface_features(readings)
    features = engineer_sonde_features(daily, casts)
    shifted = features.shift(lag_days, freq="D")
    return target.join(shifted, how="left")


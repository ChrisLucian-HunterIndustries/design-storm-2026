"""Report sections built around the Strontia profiling sonde and the
Michigan Creek anomaly check: stratification, sonde-as-predictor, storm
depth profiles, and sensor-fault detection.

Split out of analyze_parameters.py (which was already at this repo's
file-length gate) -- `analyze_parameters.main()` calls these functions
directly and appends the results to its own report lines, so there is still
only one `results/parameter_summary.md` and one command to run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_dwr_telemetry, load_michigan_creek, load_target, load_usgs_gage
from models import best_lag, detect_anomalies, lag_correlation_scan
from sonde_loader import full_depth_casts


def _fmt(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.3f}"


def _stratification_table(casts: pd.DataFrame) -> list[str]:
    """Scenario 3's "lake turnover" bullet: surface-minus-bottom temperature
    per cast is a standard stratification indicator. Report the overall
    range plus the most-mixed and most-stratified casts by date, so "lake
    turnover" names an actual observed date rather than a generic claim."""
    most_mixed = casts.loc[casts["temp_diff_c"].idxmin()]
    most_stratified = casts.loc[casts["temp_diff_c"].idxmax()]
    lines = [
        f"{len(casts)} casts, {casts['date'].min().date()} to {casts['date'].max().date()}.",
        "",
        f"- Surface-bottom temperature difference: min {_fmt(casts['temp_diff_c'].min())}\u00b0C, "
        f"max {_fmt(casts['temp_diff_c'].max())}\u00b0C, mean {_fmt(casts['temp_diff_c'].mean())}\u00b0C.",
        f"- Most mixed cast (smallest difference, closest to a turnover state): "
        f"{most_mixed['date'].date()}, {_fmt(most_mixed['temp_diff_c'])}\u00b0C "
        f"(surface {_fmt(most_mixed['surface_temp_c'])}\u00b0C, bottom {_fmt(most_mixed['bottom_temp_c'])}\u00b0C).",
        f"- Most stratified cast: {most_stratified['date'].date()}, "
        f"{_fmt(most_stratified['temp_diff_c'])}\u00b0C "
        f"(surface {_fmt(most_stratified['surface_temp_c'])}\u00b0C, bottom {_fmt(most_stratified['bottom_temp_c'])}\u00b0C).",
    ]
    return lines


def _sonde_predictor_table(data_dir: Path, sonde_readings: pd.DataFrame, daily_sonde: pd.DataFrame) -> list[str]:
    """Scenario 1's second bullet: does the sonde, sitting closer to the
    plant, predict better (or with shorter lag) than the upstream USGS gage?
    Scored on the sonde's *own* coverage window for both sensors -- the
    sonde covers only that window, so comparing it against the gage's full
    multi-year record would not be a fair test."""
    start = sonde_readings["timestamp"].min().normalize()
    end = sonde_readings["timestamp"].max().normalize()
    gage = load_usgs_gage(data_dir).loc[start:end]
    telemetry = load_dwr_telemetry(data_dir).loc[start:end]
    target = load_target(data_dir).loc[start:end]

    lines = [
        f"Same {start.date()} to {end.date()} window for both sensors ({len(target)} lab results).\n",
        "| sensor | predictor | target | best lag (days) | correlation at best lag |",
        "|---|---|---|---:|---:|",
    ]
    pairs = [
        ("sonde (Strontia)", daily_sonde["Turbidity_NTU"], "TOC_mg_L", target["TOC_mg_L"]),
        ("USGS gage (upstream)", gage["Turbidity_Median"], "TOC_mg_L", target["TOC_mg_L"]),
        ("sonde (Strontia)", daily_sonde["Conductivity"], "Alk_mg_L", target["Alk_mg_L"]),
        ("USGS gage (upstream)", gage["Specific_Cond_Mean"], "Alk_mg_L", target["Alk_mg_L"]),
    ]
    for sensor, predictor, target_name, target_series in pairs:
        scan = lag_correlation_scan(predictor, target_series, max_lag_days=10)
        lag = best_lag(scan)
        lines.append(
            f"| {sensor} | {predictor.name} | {target_name} | {lag} | {_fmt(scan[lag])} |"
        )
    return lines


def _storm_profile_table(casts: pd.DataFrame, storm_date: pd.Timestamp) -> list[str]:
    """Scenario 2's depth-profile bullet: compare the nearest full-depth
    cast before and after a real storm (peak flow AND peak turbidity day
    within the sonde's window) to see whether the storm's signal reaches
    every depth or only the surface. Restricted to full-depth casts
    (`full_depth_casts`) so a short/aborted cast doesn't get compared as if
    it were a complete water-column profile."""
    casts = full_depth_casts(casts)
    before = casts[casts["date"] < storm_date].iloc[-1]
    after = casts[casts["date"] > storm_date].iloc[0]

    lines = [
        f"Storm date (peak flow and turbidity in the sonde's window): {storm_date.date()}.\n",
        "| cast | surface turbidity (NTU) | surface conductivity | bottom temp (C) | surface temp (C) |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, cast in [("before: " + str(before["start"]), before), ("after: " + str(after["start"]), after)]:
        lines.append(
            f"| {label} | {_fmt(cast['surface_turbidity_ntu'])} | {_fmt(cast['surface_conductivity'])} | "
            f"{_fmt(cast['bottom_temp_c'])} | {_fmt(cast['surface_temp_c'])} |"
        )
    return lines


def _anomaly_detection_table(data_dir: Path) -> list[str]:
    """parameters.md's previously-unimplemented sensor-fault framing:
    IsolationForest on MichiganCreek's SWE, validated against the one
    labeled bad patch this repo already knows about (AGENTS.md: SWE 9.0 on
    2026-05-12 to 05-15, bracketed by near-zero readings) -- a real ground
    truth to check the detector against, not just a plausible-looking
    result."""
    creek = load_michigan_creek(data_dir)
    features = pd.DataFrame(index=creek.index)
    features["SWE"] = creek["SWE"]
    features["diff_prev"] = creek["SWE"].diff().abs()
    features["diff_next"] = creek["SWE"].diff(-1).abs()
    features["rolling_std_5"] = creek["SWE"].rolling(5, center=True).std()

    feature_cols = ["SWE", "diff_prev", "diff_next", "rolling_std_5"]
    result = detect_anomalies(features, feature_cols, contamination=0.01)
    flagged_dates = sorted(result.is_anomaly[result.is_anomaly].index)

    known_bad_dates = pd.date_range("2026-05-12", "2026-05-15")
    caught = [d for d in known_bad_dates if d in flagged_dates]

    lines = [
        f"{len(flagged_dates)} of {result.is_anomaly.notna().sum()} days flagged "
        f"(IsolationForest, contamination=0.01, features={feature_cols}).\n",
        f"Known bad patch (AGENTS.md): SWE=9.0 on 2026-05-12 to 05-15. Caught "
        f"{len(caught)}/{len(known_bad_dates)} of those exact days.\n",
        "| flagged date | SWE | diff from previous day |",
        "|---|---:|---:|",
    ]
    for date in flagged_dates:
        if pd.Timestamp("2026-04-01") <= date <= pd.Timestamp("2026-06-30"):
            lines.append(
                f"| {date.date()} | {_fmt(features.loc[date, 'SWE'])} | {_fmt(features.loc[date, 'diff_prev'])} |"
            )
    return lines

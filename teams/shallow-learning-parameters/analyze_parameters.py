"""Compute grounded numbers on the real Design Storm datasets: correlations,
held-out R^2, feature importances, cluster sizes, and classifier scores.

Run from this directory: `python analyze_parameters.py`. Writes
results/parameter_summary.md, which is what parameters.md's numbers are
drawn from -- nothing in that document is invented (see AGENTS.md: "Never
invent numbers").
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_loader import (
    build_dataset,
    feature_columns,
    load_dwr_telemetry,
    load_michigan_creek,
    load_snowpack,
    load_target,
    load_usgs_gage,
    load_weather,
    peak_loading_date,
    with_calendar_year_and_doy,
)
from models import (
    best_lag,
    cluster_hydrologic_regimes,
    detect_anomalies,
    fit_gradient_boosting_importance,
    fit_linear_baseline,
    fit_logistic_baseline,
    fit_random_forest_importance,
    fit_svr_baseline,
    fit_threshold_classifier,
    lag_correlation_scan,
    predict_full_series,
)
from sonde_loader import cast_summary, daily_surface_features, full_depth_casts, load_sonde_readings

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

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


LAG_SCAN_PAIRS = [
    ("Turbidity_Median", "TOC_mg_L"),
    ("Flow_CFS", "TOC_mg_L"),
    ("Specific_Cond_Mean", "Alk_mg_L"),
    ("pH_Median", "Alk_mg_L"),
]
MAX_LAG_DAYS = 10
LAG_DAYS_GRID = [0, 1, 2, 3, 4, 5, 7, 10]


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _missingness_table(df: pd.DataFrame, all_features: list[str]) -> list[str]:
    lines = ["| column | non-null rows | % missing |", "|---|---:|---:|"]
    for col in [*all_features, "TOC_mg_L", "Alk_mg_L"]:
        non_null = df[col].notna().sum()
        pct_missing = 100 * (1 - non_null / len(df))
        lines.append(f"| {col} | {non_null} | {pct_missing:.1f}% |")
    return lines


def _correlation_table(df: pd.DataFrame, all_features: list[str], target: str) -> list[str]:
    corr = df[[*all_features, target]].corr()
    lines = [f"| feature | corr with {target} |", "|---|---:|"]
    for col in all_features:
        lines.append(f"| {col} | {_fmt(corr.loc[col, target])} |")
    return lines


def _yearly_summary_table(data_dir: Path) -> list[str]:
    """Scenario 3 grounding: per-calendar-year peak/mean SWE, flow, and TOC,
    so "drought vs. wet year" in the writeup names actual years and numbers."""
    snow = with_calendar_year_and_doy(load_snowpack(data_dir)).groupby("year")["SWE"].agg(["mean", "max"])
    flow = with_calendar_year_and_doy(load_dwr_telemetry(data_dir)).groupby("year")["Flow_CFS"].agg(
        ["mean", "max"]
    )
    toc = with_calendar_year_and_doy(load_target(data_dir)).groupby("year")["TOC_mg_L"].agg(["mean", "max"])

    lines = [
        "| year | mean SWE (in) | peak SWE (in) | mean flow (CFS) | peak flow (CFS) | mean TOC (mg/L) | peak TOC (mg/L) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year in sorted(snow.index):
        row = [
            str(year),
            _fmt(snow.loc[year, "mean"]) if year in snow.index else "n/a",
            _fmt(snow.loc[year, "max"]) if year in snow.index else "n/a",
            _fmt(flow.loc[year, "mean"]) if year in flow.index else "n/a",
            _fmt(flow.loc[year, "max"]) if year in flow.index else "n/a",
            _fmt(toc.loc[year, "mean"]) if year in toc.index else "n/a",
            _fmt(toc.loc[year, "max"]) if year in toc.index else "n/a",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _lag_scan_table(data_dir: Path) -> list[str]:
    """Scenario 3 grounding: empirically fit upstream-to-plant lag per
    predictor/target pair (see models.lag_correlation_scan) rather than
    reusing Jake's fixed 2/4-day lags uncritically."""
    gage = load_usgs_gage(data_dir)
    telemetry = load_dwr_telemetry(data_dir)
    target = load_target(data_dir)
    sources = {
        "Turbidity_Median": gage["Turbidity_Median"],
        "Specific_Cond_Mean": gage["Specific_Cond_Mean"],
        "pH_Median": gage["pH_Median"],
        "Flow_CFS": telemetry["Flow_CFS"],
    }

    lines = ["| predictor | target | best lag (days) | correlation at best lag |", "|---|---|---:|---:|"]
    for predictor_name, target_name in LAG_SCAN_PAIRS:
        scan = lag_correlation_scan(sources[predictor_name], target[target_name], MAX_LAG_DAYS)
        lag = best_lag(scan)
        lines.append(f"| {predictor_name} | {target_name} | {lag} | {_fmt(scan[lag])} |")
    return lines


def _model_family_table(df_toc: pd.DataFrame, df_alk: pd.DataFrame) -> list[str]:
    """Deck slide 9: "try different models (like support-vector machines)...
    to see if they can improve performance." Same held-out split, same
    feature lists, four model families side by side -- including gradient
    boosting, the family guide.md's own comparison (Jake's CatBoost) won
    with, which earlier versions of this table omitted."""
    lines = ["| target | model | held-out R^2 |", "|---|---|---:|"]

    _, toc_linear_r2 = fit_linear_baseline(df_toc, "turb_flow", "TOC_mg_L")
    toc_rf_r2 = fit_random_forest_importance(df_toc, TOC_FEATURES, "TOC_mg_L").r2
    toc_svr_r2 = fit_svr_baseline(df_toc, TOC_FEATURES, "TOC_mg_L").r2
    toc_gbr_r2 = fit_gradient_boosting_importance(df_toc, TOC_FEATURES, "TOC_mg_L").r2
    lines.append(f"| TOC_mg_L | Linear (turb_flow only) | {_fmt(toc_linear_r2)} |")
    lines.append(f"| TOC_mg_L | Random forest | {_fmt(toc_rf_r2)} |")
    lines.append(f"| TOC_mg_L | SVR (RBF kernel, scaled features) | {_fmt(toc_svr_r2)} |")
    lines.append(f"| TOC_mg_L | Gradient boosting | {_fmt(toc_gbr_r2)} |")

    alk_rf_r2 = fit_random_forest_importance(df_alk, ALK_FEATURES, "Alk_mg_L").r2
    alk_svr_r2 = fit_svr_baseline(df_alk, ALK_FEATURES, "Alk_mg_L").r2
    alk_gbr_r2 = fit_gradient_boosting_importance(df_alk, ALK_FEATURES, "Alk_mg_L").r2
    lines.append(f"| Alk_mg_L | Random forest | {_fmt(alk_rf_r2)} |")
    lines.append(f"| Alk_mg_L | SVR (RBF kernel, scaled features) | {_fmt(alk_svr_r2)} |")
    lines.append(f"| Alk_mg_L | Gradient boosting | {_fmt(alk_gbr_r2)} |")
    return lines


def _classifier_family_table(df_alk: pd.DataFrame) -> list[str]:
    """A second "try different models" comparison, this time for the
    alkalinity-below-60 classifier: random forest vs. a logistic regression
    baseline (`LogisticRegression` was imported in models.py but never
    actually used until this comparison)."""
    rf_result = fit_threshold_classifier(df_alk, ALK_FEATURES, "Alk_mg_L", threshold=60.0)
    logistic_result = fit_logistic_baseline(df_alk, ALK_FEATURES, "Alk_mg_L", threshold=60.0)
    return [
        "| model | ROC-AUC |",
        "|---|---:|",
        f"| Random forest classifier | {_fmt(rf_result.roc_auc)} |",
        f"| Logistic regression (scaled features) | {_fmt(logistic_result.roc_auc)} |",
    ]


def _lag_day_grid_search_table(data_dir: Path) -> list[str]:
    """Deck slide 9: "try different... lag-times... to see if they can
    improve performance." Rebuilds the whole dataset at each candidate lag
    (this also re-times the `turb_flow` engineered feature) and refits the
    same random forest used in the main results table above."""
    lines = ["| lag (days) | TOC_mg_L held-out R^2 | Alk_mg_L held-out R^2 |", "|---:|---:|---:|"]
    for lag in LAG_DAYS_GRID:
        df = build_dataset(data_dir, lag_days=lag)
        toc_r2 = fit_random_forest_importance(df, TOC_FEATURES, "TOC_mg_L").r2
        alk_r2 = fit_random_forest_importance(df, ALK_FEATURES, "Alk_mg_L").r2
        lines.append(f"| {lag} | {_fmt(toc_r2)} | {_fmt(alk_r2)} |")
    return lines


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


def _series_to_points(series: pd.Series) -> list[list]:
    """A pandas Series (DatetimeIndex -> float) as [["YYYY-MM-DD", value], ...]
    JSON, for viewer.html's Chart.js time-scale line charts."""
    return [
        [index.strftime("%Y-%m-%d"), None if pd.isna(value) else float(value)]
        for index, value in series.items()
    ]


def export_viewer_json(
    data_dir: Path, df_toc: pd.DataFrame, df_alk: pd.DataFrame, results_dir: Path
) -> Path:
    """Write results/predictions.json: actual vs. predicted TOC/alkalinity
    (train and held-out test clearly separated) plus the raw context series
    -- streamflow, turbidity, precipitation, snowpack -- viewer.html plots.
    This is the deck's Scenario 1 web-app bullet: "viewing all the relevant
    data for predictions... and projected TOC and/or alkalinity."""
    toc_pred = predict_full_series(df_toc, TOC_FEATURES, "TOC_mg_L")
    alk_pred = predict_full_series(df_alk, ALK_FEATURES, "Alk_mg_L")

    def target_payload(result, lag_days: int) -> dict:
        frame = result.frame
        return {
            "lag_days": lag_days,
            "test_r2": result.test_r2,
            "actual": _series_to_points(frame["actual"]),
            "predicted_train": _series_to_points(frame.loc[frame["split"] == "train", "predicted"]),
            "predicted_test": _series_to_points(frame.loc[frame["split"] == "test", "predicted"]),
        }

    gage = load_usgs_gage(data_dir)
    telemetry = load_dwr_telemetry(data_dir)
    weather = load_weather(data_dir)
    snow = load_snowpack(data_dir)

    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "toc": target_payload(toc_pred, lag_days=2),
        "alk": target_payload(alk_pred, lag_days=4),
        "context": {
            "Flow_CFS": _series_to_points(telemetry["Flow_CFS"]),
            "Turbidity_Median": _series_to_points(gage["Turbidity_Median"]),
            "PRCP": _series_to_points(weather["PRCP"]),
            "SWE": _series_to_points(snow["SWE"]),
        },
    }

    out_path = results_dir / "predictions.json"
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    return out_path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    # Lags match guide.md section 14: TOC shifts predictors 2 days, alkalinity 4.
    df_toc = build_dataset(DATA_DIR, lag_days=2)
    df_alk = build_dataset(DATA_DIR, lag_days=4)
    all_features = feature_columns(df_toc)

    lines: list[str] = []
    lines.append("# Parameter summary (computed, not invented)\n")
    lines.append(
        f"TOC frame (lag_days=2): {len(df_toc)} rows, {df_toc.index.min().date()} to "
        f"{df_toc.index.max().date()}. Alkalinity frame (lag_days=4): {len(df_alk)} rows, "
        f"same range. See data_loader.build_dataset.\n"
    )

    lines.append("## Missingness after the join (TOC frame, lag_days=2)\n")
    lines += _missingness_table(df_toc, all_features)

    lines.append("\n## Correlation with TOC_mg_L (lag_days=2, Pearson, pairwise complete)\n")
    lines += _correlation_table(df_toc, all_features, "TOC_mg_L")

    lines.append("\n## Correlation with Alk_mg_L (lag_days=4, Pearson, pairwise complete)\n")
    lines += _correlation_table(df_alk, all_features, "Alk_mg_L")

    lines.append("\n## Random forest, held-out R^2 and feature importance\n")
    for target, features, df in [
        ("TOC_mg_L", TOC_FEATURES, df_toc),
        ("Alk_mg_L", ALK_FEATURES, df_alk),
    ]:
        result = fit_random_forest_importance(df, features, target)
        lines.append(f"\n**{target}** (lag_days={2 if target == 'TOC_mg_L' else 4}): held-out R^2 = {_fmt(result.r2)}\n")
        lines.append("| feature | importance |")
        lines.append("|---|---:|")
        for feature, importance in result.importances.items():
            lines.append(f"| {feature} | {_fmt(importance)} |")

    lines.append("\n## Linear baseline: turb_flow -> TOC_mg_L (lag_days=2)\n")
    _, linear_r2 = fit_linear_baseline(df_toc, "turb_flow", "TOC_mg_L")
    lines.append(f"Held-out R^2 = {_fmt(linear_r2)} (single feature, no trees).\n")

    lines.append("\n## Alkalinity < 60 mg/L classifier (RandomForestClassifier, lag_days=4)\n")
    clf_result = fit_threshold_classifier(df_alk, ALK_FEATURES, "Alk_mg_L", threshold=60.0)
    lines.append(f"ROC-AUC (held-out) = {_fmt(clf_result.roc_auc)}\n")
    lines.append("| feature | importance |")
    lines.append("|---|---:|")
    for feature, importance in clf_result.importances.items():
        lines.append(f"| {feature} | {_fmt(importance)} |")

    lines.append("\n## Unsupervised hydrologic-regime clusters (k=3, TOC frame lag_days=2)\n")
    clean, clusters = cluster_hydrologic_regimes(df_toc, all_features, n_clusters=3)
    lines.append("| cluster | days | mean TOC_mg_L | mean Alk_mg_L | mean Flow_CFS |")
    lines.append("|---:|---:|---:|---:|---:|")
    clean = clean.assign(_cluster=clusters.labels)
    for cluster_id, group in clean.groupby("_cluster"):
        lines.append(
            f"| {cluster_id} | {len(group)} | {_fmt(group['TOC_mg_L'].mean())} | "
            f"{_fmt(group['Alk_mg_L'].mean())} | {_fmt(group['Flow_CFS'].mean())} |"
        )

    lines.append("\n## Year-over-year summary (Scenario 3: drought vs. wet years)\n")
    lines += _yearly_summary_table(DATA_DIR)

    lines.append(
        "\n## Sensor-fault anomaly detection (previously an unimplemented idea in parameters.md)\n"
    )
    lines += _anomaly_detection_table(DATA_DIR)

    lines.append(
        "\n## Empirical transit-time lag scan (Scenario 3: follow a parameter through the system)\n"
    )
    lines.append(
        f"Correlation between each raw upstream predictor and each target, scanned over lags "
        f"0-{MAX_LAG_DAYS} days (models.lag_correlation_scan). This is a statistical fit, not a "
        "measured travel time -- see guide.md section 1.\n"
    )
    lines += _lag_scan_table(DATA_DIR)

    lines.append(
        "\n## Model family comparison (Scenario 1: \"try different models like support-vector machines\")\n"
    )
    lines += _model_family_table(df_toc, df_alk)

    lines.append("\n## Alkalinity classifier: model family comparison\n")
    lines += _classifier_family_table(df_alk)

    lines.append(
        "\n## Lag-day grid search (Scenario 1: \"try different... lag-times\")\n"
    )
    lines.append(
        "Same random forest and feature list as above, refit at each candidate lag. "
        "Elsewhere in this document TOC uses lag_days=2 and Alk_mg_L uses lag_days=4 "
        "(guide.md section 14); compare those rows below against the rest of the grid.\n"
    )
    lines += _lag_day_grid_search_table(DATA_DIR)

    lines.append(
        "\n## Strontia profiling sonde: stratification (Scenario 3: \"lake turnover\")\n"
    )
    sonde_readings = load_sonde_readings(DATA_DIR)
    sonde_casts = cast_summary(sonde_readings)
    lines += _stratification_table(sonde_casts)

    lines.append(
        "\n## Strontia profiling sonde as a closer predictor (Scenario 1: \"introduce real-time "
        "Strontia profiling sonde data\")\n"
    )
    sonde_daily = daily_surface_features(sonde_readings)
    lines += _sonde_predictor_table(DATA_DIR, sonde_readings, sonde_daily)

    lines.append(
        "\n## Storm impact on the reservoir's depth profile (Scenario 2: \"how do water quality "
        "parameters change and distribute by depth\")\n"
    )
    storm_date = peak_loading_date(
        DATA_DIR, sonde_readings["timestamp"].min().normalize(), sonde_readings["timestamp"].max().normalize()
    )
    lines += _storm_profile_table(sonde_casts, storm_date)

    output_path = RESULTS_DIR / "parameter_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")

    json_path = export_viewer_json(DATA_DIR, df_toc, df_alk, RESULTS_DIR)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

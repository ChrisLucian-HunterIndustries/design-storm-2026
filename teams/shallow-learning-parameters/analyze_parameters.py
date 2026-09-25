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

from analyze_advanced import build_report_sections
from analyze_dosing import build_dosing_alert_section
from analyze_enso import build_enso_experiment_section
from analyze_lag_experiments import build_lag_experiments_section
from analyze_sonde import (
    _anomaly_detection_table,
    _limited_window_model_table,
    _sonde_predictor_table,
    _storm_profile_table,
    _stratification_table,
)
from data_loader import (
    build_dataset,
    feature_columns,
    load_dwr_telemetry,
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
    fit_gradient_boosting_importance,
    fit_linear_baseline,
    fit_logistic_baseline,
    fit_random_forest_importance,
    fit_svr_baseline,
    fit_threshold_classifier,
    lag_correlation_scan,
    predict_full_series,
)
from sonde_loader import cast_summary, daily_surface_features, load_sonde_readings

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

    lines.append(
        "\n## Chemical-dosing alert calendar (when to add chemical after a spike)\n"
    )
    lines += build_dosing_alert_section(df_toc, TOC_FEATURES, df_alk, ALK_FEATURES)

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

    lines += build_lag_experiments_section(DATA_DIR, TOC_FEATURES, ALK_FEATURES)

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
        "\n## Full feature-set comparison on the sonde's limited 4-month window "
        "(Scenario 1 x Scenario 2: does the sonde add predictive value there?)\n"
    )
    lines += _limited_window_model_table(
        DATA_DIR,
        sonde_readings,
        {"TOC_mg_L": TOC_FEATURES, "Alk_mg_L": ALK_FEATURES},
        {"TOC_mg_L": 2, "Alk_mg_L": 4},
    )

    lines.append(
        "\n## Storm impact on the reservoir's depth profile (Scenario 2: \"how do water quality "
        "parameters change and distribute by depth\")\n"
    )
    storm_date = peak_loading_date(
        DATA_DIR, sonde_readings["timestamp"].min().normalize(), sonde_readings["timestamp"].max().normalize()
    )
    lines += _storm_profile_table(sonde_casts, storm_date)

    lines += build_report_sections(DATA_DIR, df_toc, TOC_FEATURES, df_alk, ALK_FEATURES, all_features)

    lines += build_enso_experiment_section(DATA_DIR)

    output_path = RESULTS_DIR / "parameter_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")

    json_path = export_viewer_json(DATA_DIR, df_toc, df_alk, RESULTS_DIR)
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

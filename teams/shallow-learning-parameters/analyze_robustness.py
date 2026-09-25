"""Four follow-up experiments on Scenario 1's limited 4-month sonde window
(analyze_sonde.compute_limited_window_scores, which found every feature set
scoring at or below a mean-only baseline). Each tests a different fix for
"too many parameters, too few rows" besides adding more features:

1. Shrink parameters instead (Ridge/Lasso) on the least-bad feature set.
2. Measure the existing split's stability directly (TimeSeriesSplit CV)
   instead of trusting one 50/50 split.
3. Predict day-over-day deltas instead of absolute levels.
4. Route through the sonde's higher-frequency casts (390 rows) instead of
   its daily-aggregated features (~100-130 rows).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score

from data_loader import build_dataset, load_target, load_usgs_gage
from models import (
    RANDOM_STATE,
    fit_random_forest_importance,
    fit_regularized_linear,
    fit_svr_baseline,
    time_ordered_split,
    time_series_cv_scores,
)
from sonde_loader import cast_summary

CAST_FEATURES = [
    "surface_temp_c",
    "bottom_temp_c",
    "temp_diff_c",
    "surface_turbidity_ntu",
    "surface_conductivity",
    "surface_ph",
]
CHAIN_LAG_DAYS = 4  # matches _sonde_predictor_table's fitted gage-turbidity-to-TOC lag


def _fmt(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{value:.3f}"


def _sonde_window(sonde_readings: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return sonde_readings["timestamp"].min().normalize(), sonde_readings["timestamp"].max().normalize()


def compute_regularization_comparison(
    data_dir: Path,
    sonde_readings: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
) -> pd.DataFrame:
    """Restricted to the sonde window's least-bad feature set (national
    datasets only), does shrinking parameters (Ridge/Lasso) do better than
    random forest/SVR's unregularized overfitting?"""
    start, end = _sonde_window(sonde_readings)
    records = []
    for target in ("TOC_mg_L", "Alk_mg_L"):
        lag = lag_days[target]
        cols = national_features[target]
        df = build_dataset(data_dir, lag_days=lag).loc[start:end]
        n_rows = len(df.dropna(subset=[*cols, target]))
        if n_rows < 10:
            records.append(
                {"target": target, "n_rows": n_rows, "rf_r2": float("nan"), "svr_r2": float("nan"),
                 "ridge_r2": float("nan"), "lasso_r2": float("nan")}
            )
            continue
        records.append(
            {
                "target": target,
                "n_rows": n_rows,
                "rf_r2": fit_random_forest_importance(df, cols, target).r2,
                "svr_r2": fit_svr_baseline(df, cols, target).r2,
                "ridge_r2": fit_regularized_linear(df, cols, target, penalty="ridge").r2,
                "lasso_r2": fit_regularized_linear(df, cols, target, penalty="lasso", alpha=0.1).r2,
            }
        )
    return pd.DataFrame.from_records(records)


def _regularization_table(scores: pd.DataFrame) -> list[str]:
    lines = [
        "| target | rows | random forest R² | SVR R² | Ridge R² | Lasso R² |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in scores.itertuples():
        lines.append(
            f"| {row.target} | {row.n_rows} | {_fmt(row.rf_r2)} | {_fmt(row.svr_r2)} | "
            f"{_fmt(row.ridge_r2)} | {_fmt(row.lasso_r2)} |"
        )
    return lines


def compute_cv_stability_scores(
    data_dir: Path,
    sonde_readings: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
    n_splits: int = 5,
) -> pd.DataFrame:
    """Is a single 50/50 split's R^2 on this window a stable property of
    the model, or noise? Ridge (already the most robust option above) run
    through TimeSeriesSplit instead of one split -- `n_splits` independent
    estimates from the same rows."""
    start, end = _sonde_window(sonde_readings)
    records = []
    for target in ("TOC_mg_L", "Alk_mg_L"):
        lag = lag_days[target]
        cols = national_features[target]
        df = build_dataset(data_dir, lag_days=lag).loc[start:end]
        n_rows = len(df.dropna(subset=[*cols, target]))
        if n_rows < n_splits * 4:
            continue
        scores = time_series_cv_scores(df, cols, target, Ridge(alpha=1.0), n_splits=n_splits)
        for fold, score in enumerate(scores):
            records.append({"target": target, "fold": fold, "r2": score})
    return pd.DataFrame.from_records(records, columns=["target", "fold", "r2"])


def _cv_stability_table(scores: pd.DataFrame) -> list[str]:
    lines = ["| target | folds | mean R² | std R² | min R² | max R² |", "|---|---:|---:|---:|---:|---:|"]
    if scores.empty:
        lines.append("| (too few rows for this many folds) | | | | | |")
        return lines
    for target, group in scores.groupby("target"):
        lines.append(
            f"| {target} | {len(group)} | {_fmt(group['r2'].mean())} | {_fmt(group['r2'].std())} | "
            f"{_fmt(group['r2'].min())} | {_fmt(group['r2'].max())} |"
        )
    return lines


def compute_delta_target_scores(
    data_dir: Path,
    sonde_readings: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
) -> pd.DataFrame:
    """The window can't learn a year-scale signal it never saw (2026's
    drought). Differencing the target removes that signal entirely and
    asks only what the window's own storm-to-storm dynamics can support --
    does that framing score better than predicting the absolute level?"""
    start, end = _sonde_window(sonde_readings)
    records = []
    for target in ("TOC_mg_L", "Alk_mg_L"):
        lag = lag_days[target]
        cols = national_features[target]
        df = build_dataset(data_dir, lag_days=lag).loc[start:end]
        delta_col = f"{target}_delta"
        df = df.assign(**{delta_col: df[target].diff()})

        for framing, col in [("level", target), ("day-over-day delta", delta_col)]:
            n_rows = len(df.dropna(subset=[*cols, col]))
            rf_r2 = float("nan") if n_rows < 10 else fit_random_forest_importance(df, cols, col).r2
            ridge_r2 = float("nan") if n_rows < 10 else fit_regularized_linear(df, cols, col, penalty="ridge").r2
            records.append({"target": target, "framing": framing, "n_rows": n_rows, "rf_r2": rf_r2, "ridge_r2": ridge_r2})
    return pd.DataFrame.from_records(records)


def _delta_target_table(scores: pd.DataFrame) -> list[str]:
    lines = ["| target | framing | rows | random forest R² | Ridge R² |", "|---|---|---:|---:|---:|"]
    for row in scores.itertuples():
        lines.append(
            f"| {row.target} | {row.framing} | {row.n_rows} | {_fmt(row.rf_r2)} | {_fmt(row.ridge_r2)} |"
        )
    return lines


def _linear_r2_from_series(predictor: pd.Series, target: pd.Series) -> tuple[float, int]:
    frame = pd.DataFrame({"x": predictor, "y": target}).dropna()
    if len(frame) < 10:
        return float("nan"), len(frame)
    train, test = time_ordered_split(frame)
    model = LinearRegression()
    model.fit(train[["x"]], train["y"])
    predictions = model.predict(test[["x"]])
    return r2_score(test["y"], predictions), len(frame)


def compute_two_stage_chain_scores(data_dir: Path, sonde_readings: pd.DataFrame) -> pd.DataFrame:
    """Every other limited-window model is capped at ~100-130 daily rows by
    the lab-result cadence. The sonde itself has 390 casts (roughly every 6
    hours) -- does routing through cast-level features to estimate the
    upstream gage's own turbidity reading (stage 1, ~3x the rows), then
    feeding that estimate into a simple TOC~turbidity model (stage 2), do
    any better than just using the real gage reading directly?"""
    casts = cast_summary(sonde_readings)
    gage = load_usgs_gage(data_dir)
    casts = casts.assign(gage_turbidity=casts["date"].map(gage["Turbidity_Median"]))
    clean = casts.dropna(subset=[*CAST_FEATURES, "gage_turbidity"])

    if len(clean) < 10:
        return pd.DataFrame.from_records(
            [{"stage": name, "n_rows": len(clean), "r2": float("nan")} for name in
             ("stage 1: casts -> gage turbidity", "stage 2: real gage turbidity -> TOC (direct)",
              "stage 2: sonde-derived turbidity -> TOC (chain)")]
        )

    train, test = time_ordered_split(clean)
    stage1_model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    stage1_model.fit(train[CAST_FEATURES], train["gage_turbidity"])
    stage1_r2 = r2_score(test["gage_turbidity"], stage1_model.predict(test[CAST_FEATURES]))

    predicted = clean.assign(predicted_gage_turbidity=stage1_model.predict(clean[CAST_FEATURES]))
    daily_predicted = predicted.groupby("date")["predicted_gage_turbidity"].mean()
    daily_predicted.index.name = "DATE"

    start, end = _sonde_window(sonde_readings)
    target = load_target(data_dir)["TOC_mg_L"]
    real_series = gage["Turbidity_Median"].shift(CHAIN_LAG_DAYS, freq="D").loc[start:end]
    chain_series = daily_predicted.shift(CHAIN_LAG_DAYS, freq="D").loc[start:end]

    real_r2, real_n = _linear_r2_from_series(real_series, target.loc[start:end])
    chain_r2, chain_n = _linear_r2_from_series(chain_series, target.loc[start:end])

    return pd.DataFrame.from_records(
        [
            {"stage": "stage 1: casts -> gage turbidity", "n_rows": len(clean), "r2": stage1_r2},
            {"stage": "stage 2: real gage turbidity -> TOC (direct)", "n_rows": real_n, "r2": real_r2},
            {"stage": "stage 2: sonde-derived turbidity -> TOC (chain)", "n_rows": chain_n, "r2": chain_r2},
        ]
    )


def _two_stage_chain_table(scores: pd.DataFrame) -> list[str]:
    lines = ["| stage | rows | held-out R² |", "|---|---:|---:|"]
    for row in scores.itertuples():
        lines.append(f"| {row.stage} | {row.n_rows} | {_fmt(row.r2)} |")
    return lines


def build_robustness_section(
    data_dir: Path,
    sonde_readings: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
) -> list[str]:
    lines = [
        "\n## Other ideas for the limited 4-month window (shrink parameters, measure "
        "stability, change the target, use higher-frequency data)\n",
        "\n### Shrink parameters instead of adding features (Ridge/Lasso)\n",
    ]
    lines += _regularization_table(
        compute_regularization_comparison(data_dir, sonde_readings, national_features, lag_days)
    )
    lines.append(
        "\n### Is a single 50/50 split's R² on this window stable, or noise? "
        "(TimeSeriesSplit cross-validation, Ridge)\n"
    )
    lines += _cv_stability_table(
        compute_cv_stability_scores(data_dir, sonde_readings, national_features, lag_days)
    )
    lines.append("\n### Predict day-over-day deltas instead of absolute levels\n")
    lines += _delta_target_table(
        compute_delta_target_scores(data_dir, sonde_readings, national_features, lag_days)
    )
    lines.append(
        "\n### Two-stage chain: route through the sonde's 390 casts instead of its "
        "~100-130 daily rows\n"
    )
    lines += _two_stage_chain_table(compute_two_stage_chain_scores(data_dir, sonde_readings))
    return lines

"""Report sections for the "still genuinely missing" model families:
hyperparameter tuning, flow forecasting, quantile/peak-focused regression,
joint multi-output modeling, Gaussian Process uncertainty, and SARIMAX.

Split out of analyze_parameters.py (which was already at this repo's
file-length gate) -- `analyze_parameters.main()` calls
`build_report_sections` and appends the result to its own report lines, so
there is still only one `results/parameter_summary.md` and one command to
run.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import build_flow_forecast_dataset
from models import fit_gradient_boosting_importance, fit_linear_baseline, fit_random_forest_importance
from models_advanced import (
    fit_gaussian_process,
    fit_multioutput_random_forest,
    fit_quantile_regressor,
    fit_sarimax_baseline,
    fit_tuned_gradient_boosting,
)

FLOW_FEATURES = ["Flow_CFS", "roll_flow_7", "SWE", "roll_swe_7", "TMAX", "TMIN"]


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _tuning_table(df_toc: pd.DataFrame, toc_features: list[str], df_alk: pd.DataFrame, alk_features: list[str]) -> list[str]:
    """Hyperparameter tuning: GridSearchCV over GradientBoostingRegressor,
    scored with TimeSeriesSplit within the training half. Direct follow-up
    to the earlier finding that an untuned GradientBoostingRegressor was
    this catalog's worst TOC model -- does tuning fix that?"""
    untuned_toc_r2 = fit_gradient_boosting_importance(df_toc, toc_features, "TOC_mg_L").r2
    untuned_alk_r2 = fit_gradient_boosting_importance(df_alk, alk_features, "Alk_mg_L").r2
    toc_result = fit_tuned_gradient_boosting(df_toc, toc_features, "TOC_mg_L")
    alk_result = fit_tuned_gradient_boosting(df_alk, alk_features, "Alk_mg_L")
    return [
        "| target | untuned R^2 | tuned R^2 | best params |",
        "|---|---:|---:|---|",
        f"| TOC_mg_L | {_fmt(untuned_toc_r2)} | {_fmt(toc_result.r2)} | {toc_result.best_params} |",
        f"| Alk_mg_L | {_fmt(untuned_alk_r2)} | {_fmt(alk_result.r2)} | {alk_result.best_params} |",
    ]


def _flow_forecast_table(data_dir: Path, lead_days: int = 3) -> list[str]:
    """parameters.md's first, longest-standing unimplemented idea: forecast
    streamflow a few days out from today's snowpack/weather/current flow,
    independent of the TOC/alkalinity question entirely."""
    flow_df = build_flow_forecast_dataset(data_dir, lead_days=lead_days)
    target = "Flow_CFS_future"

    rf_result = fit_random_forest_importance(flow_df, FLOW_FEATURES, target)
    _, linear_r2 = fit_linear_baseline(flow_df, "Flow_CFS", target)

    lines = [
        f"Predicting Flow_CFS {lead_days} days ahead from today's conditions ({flow_df[target].notna().sum()} rows).\n",
        "| model | held-out R^2 |",
        "|---|---:|",
        f"| Linear (today's Flow_CFS only) | {_fmt(linear_r2)} |",
        f"| Random forest ({', '.join(FLOW_FEATURES)}) | {_fmt(rf_result.r2)} |",
        "",
        "| feature | importance |",
        "|---|---:|",
    ]
    for feature, importance in rf_result.importances.items():
        lines.append(f"| {feature} | {_fmt(importance)} |")
    return lines


def _quantile_table(df_toc: pd.DataFrame, toc_features: list[str], df_alk: pd.DataFrame, alk_features: list[str]) -> list[str]:
    """guide.md section 10: Jake cares more about catching peaks than mean
    R^2, but every model in this catalog (and his) minimizes squared error.
    A quantile regressor targets the 90th percentile directly; `coverage`
    close to 0.90 means the predicted band is well-calibrated."""
    toc_result = fit_quantile_regressor(df_toc, toc_features, "TOC_mg_L", quantile=0.9)
    alk_result = fit_quantile_regressor(df_alk, alk_features, "Alk_mg_L", quantile=0.9)
    return [
        "| target | quantile | coverage (actual <= predicted) |",
        "|---|---:|---:|",
        f"| TOC_mg_L | {toc_result.quantile} | {_fmt(toc_result.coverage)} |",
        f"| Alk_mg_L | {alk_result.quantile} | {_fmt(alk_result.coverage)} |",
    ]


def _multioutput_table(df_toc: pd.DataFrame, toc_features: list[str], alk_features: list[str], all_features: list[str]) -> list[str]:
    """A single random forest fit jointly on TOC_mg_L and Alk_mg_L (sklearn
    forests accept 2D y natively) vs. fitting each independently -- both
    computed fresh here, on the same lag_days=2 frame, for a same-run,
    apples-to-apples comparison."""
    independent_toc_r2 = fit_random_forest_importance(df_toc, toc_features, "TOC_mg_L").r2
    independent_alk_r2 = fit_random_forest_importance(df_toc, alk_features, "Alk_mg_L").r2
    joint_result = fit_multioutput_random_forest(df_toc, all_features, ["TOC_mg_L", "Alk_mg_L"])
    return [
        "Both scored on the same lag_days=2 frame, so alkalinity's independent number here differs "
        "from the model family comparison above (which uses alkalinity's own lag_days=4 frame).\n",
        "| target | independent random-forest R^2 | joint random-forest R^2 |",
        "|---|---:|---:|",
        f"| TOC_mg_L | {_fmt(independent_toc_r2)} | {_fmt(joint_result.r2_by_target['TOC_mg_L'])} |",
        f"| Alk_mg_L | {_fmt(independent_alk_r2)} | {_fmt(joint_result.r2_by_target['Alk_mg_L'])} |",
    ]


def _gaussian_process_table(df_toc: pd.DataFrame, toc_features: list[str], df_alk: pd.DataFrame, alk_features: list[str]) -> list[str]:
    """GaussianProcessRegressor: the one model family here that returns
    calibrated uncertainty alongside a point estimate."""
    toc_result = fit_gaussian_process(df_toc, toc_features, "TOC_mg_L")
    alk_result = fit_gaussian_process(df_alk, alk_features, "Alk_mg_L")
    return [
        "| target | held-out R^2 | mean predicted std dev |",
        "|---|---:|---:|",
        f"| TOC_mg_L | {_fmt(toc_result.r2)} | {_fmt(toc_result.std.mean())} |",
        f"| Alk_mg_L | {_fmt(alk_result.r2)} | {_fmt(alk_result.std.mean())} |",
    ]


def _sarimax_table(df_toc: pd.DataFrame, df_alk: pd.DataFrame) -> list[str]:
    """SARIMAX: models each target's own autocorrelation directly (AR/MA
    terms) instead of via a manually lagged predictor column, with the
    single strongest engineered predictor as an exogenous regressor."""
    toc_result = fit_sarimax_baseline(df_toc, "TOC_mg_L", exog_cols=["turb_flow"], order=(1, 0, 1))
    alk_result = fit_sarimax_baseline(df_alk, "Alk_mg_L", exog_cols=["Specific_Cond_Mean"], order=(1, 0, 1))
    return [
        "| target | exog | order | held-out R^2 |",
        "|---|---|---|---:|",
        f"| TOC_mg_L | turb_flow | {toc_result.order} | {_fmt(toc_result.r2)} |",
        f"| Alk_mg_L | Specific_Cond_Mean | {alk_result.order} | {_fmt(alk_result.r2)} |",
    ]


def build_report_sections(
    data_dir: Path,
    df_toc: pd.DataFrame,
    toc_features: list[str],
    df_alk: pd.DataFrame,
    alk_features: list[str],
    all_features: list[str],
) -> list[str]:
    """Every "still genuinely missing" section, in the order parameters.md
    ranks them."""
    lines: list[str] = []

    lines.append("\n## Hyperparameter tuning (GridSearchCV over gradient boosting)\n")
    lines += _tuning_table(df_toc, toc_features, df_alk, alk_features)

    lines.append("\n## Flow forecasting (previously an unimplemented idea in parameters.md)\n")
    lines += _flow_forecast_table(data_dir)

    lines.append("\n## Quantile / peak-focused regression (guide.md: \"catching peaks matters more\")\n")
    lines += _quantile_table(df_toc, toc_features, df_alk, alk_features)

    lines.append("\n## Joint multi-output modeling of TOC_mg_L and Alk_mg_L\n")
    lines += _multioutput_table(df_toc, toc_features, alk_features, all_features)

    lines.append("\n## Gaussian Process regression (uncertainty bands)\n")
    lines += _gaussian_process_table(df_toc, toc_features, df_alk, alk_features)

    lines.append("\n## SARIMAX (time-series-native, vs. lag-as-feature)\n")
    lines += _sarimax_table(df_toc, df_alk)

    return lines

"""Figures 19-24: the six "still genuinely missing" model families --
hyperparameter tuning, flow forecasting, quantile/peak-focused regression,
joint multi-output modeling, Gaussian Process uncertainty bands, and
SARIMAX. Split out of visualize.py to keep files under the repo's
file-length gate; see visualize.py for the entry point (`main()`), and
analyze_advanced.py for the matching report tables these numbers also
appear in.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import build_flow_forecast_dataset
from models import (
    fit_gradient_boosting_importance,
    fit_linear_baseline,
    fit_random_forest_importance,
    time_ordered_split,
)
from models_advanced import (
    fit_gaussian_process,
    fit_multioutput_random_forest,
    fit_quantile_regressor,
    fit_sarimax_baseline,
    fit_tuned_gradient_boosting,
)

FLOW_FEATURES = ["Flow_CFS", "roll_flow_7", "SWE", "roll_swe_7", "TMAX", "TMIN"]


def plot_tuning_comparison(
    df_toc: pd.DataFrame, toc_features: list[str], df_alk: pd.DataFrame, alk_features: list[str], out_path: Path
) -> None:
    """Fig 19: untuned vs. GridSearchCV-tuned gradient boosting, held-out
    R^2. Direct follow-up to the earlier finding that an untuned
    GradientBoostingRegressor was this catalog's worst TOC model."""
    untuned_toc = fit_gradient_boosting_importance(df_toc, toc_features, "TOC_mg_L").r2
    untuned_alk = fit_gradient_boosting_importance(df_alk, alk_features, "Alk_mg_L").r2
    tuned_toc = fit_tuned_gradient_boosting(df_toc, toc_features, "TOC_mg_L").r2
    tuned_alk = fit_tuned_gradient_boosting(df_alk, alk_features, "Alk_mg_L").r2

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    axes[0].bar(["Untuned", "Tuned\n(GridSearchCV)"], [untuned_toc, tuned_toc], color=["tab:gray", "tab:orange"])
    axes[0].set_title("TOC_mg_L")
    axes[0].set_ylabel("held-out R^2")
    axes[0].axhline(0, color="black", linewidth=0.8)

    axes[1].bar(["Untuned", "Tuned\n(GridSearchCV)"], [untuned_alk, tuned_alk], color=["tab:gray", "tab:orange"])
    axes[1].set_title("Alk_mg_L")
    axes[1].axhline(0, color="black", linewidth=0.8)

    fig.suptitle("Gradient boosting: untuned vs. GridSearchCV-tuned (held-out R^2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_flow_forecast(data_dir: Path, out_path: Path, lead_days: int = 3) -> None:
    """Fig 20: 3-day-ahead streamflow forecast, actual vs. linear-model
    prediction over the held-out half. parameters.md's longest-standing
    unimplemented idea, independent of the TOC/alkalinity question."""
    flow_df = build_flow_forecast_dataset(data_dir, lead_days=lead_days)
    target = "Flow_CFS_future"
    clean = flow_df.dropna(subset=["Flow_CFS", target])
    _, test = time_ordered_split(clean)

    model, r2 = fit_linear_baseline(flow_df, "Flow_CFS", target)
    predictions = model.predict(test[["Flow_CFS"]])

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(test.index, test[target], color="tab:blue", linewidth=1.0, label="actual (future flow)")
    ax.plot(test.index, predictions, color="tab:orange", linewidth=1.0, alpha=0.8, label="linear prediction")
    ax.set_ylabel("Flow_CFS")
    ax.set_title(f"{lead_days}-day-ahead streamflow forecast, held-out half (R^2={r2:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_quantile_bands(
    df_toc: pd.DataFrame, toc_features: list[str], df_alk: pd.DataFrame, alk_features: list[str], out_path: Path
) -> None:
    """Fig 21: predicted 90th-percentile band vs. actual for both targets --
    guide.md section 10 says catching peaks matters more than mean R^2, but
    every other model here minimizes squared error instead."""
    toc_result = fit_quantile_regressor(df_toc, toc_features, "TOC_mg_L", quantile=0.9)
    alk_result = fit_quantile_regressor(df_alk, alk_features, "Alk_mg_L", quantile=0.9)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for ax, result, name in [(axes[0], toc_result, "TOC_mg_L"), (axes[1], alk_result, "Alk_mg_L")]:
        ax.plot(result.actual.index, result.actual, color="tab:blue", linewidth=1.0, label="actual")
        ax.plot(
            result.predictions.index, result.predictions, color="tab:red", linewidth=1.0,
            label=f"predicted {int(result.quantile * 100)}th percentile",
        )
        exceeded = result.actual[result.actual > result.predictions]
        ax.scatter(exceeded.index, exceeded, color="tab:red", s=14, zorder=5, label="actual exceeds prediction")
        ax.set_title(f"{name} (coverage={result.coverage:.3f}, target={result.quantile})")
        ax.legend(fontsize=8)
    fig.suptitle("Quantile / peak-focused regression: predicted 90th percentile vs. actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_multioutput_comparison(
    df_toc: pd.DataFrame, toc_features: list[str], alk_features: list[str], all_features: list[str], out_path: Path
) -> None:
    """Fig 22: independent random forests vs. a single joint random forest
    fit on both targets at once (sklearn forests accept 2D y natively).
    Both scored on the same lag_days=2 frame for a fair comparison."""
    independent_toc = fit_random_forest_importance(df_toc, toc_features, "TOC_mg_L").r2
    independent_alk = fit_random_forest_importance(df_toc, alk_features, "Alk_mg_L").r2
    joint = fit_multioutput_random_forest(df_toc, all_features, ["TOC_mg_L", "Alk_mg_L"])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = ["TOC_mg_L", "Alk_mg_L"]
    width = 0.35
    positions = range(len(x))
    ax.bar(
        [p - width / 2 for p in positions], [independent_toc, independent_alk], width,
        color="tab:purple", label="independent random forest",
    )
    ax.bar(
        [p + width / 2 for p in positions],
        [joint.r2_by_target["TOC_mg_L"], joint.r2_by_target["Alk_mg_L"]], width,
        color="tab:orange", label="joint random forest",
    )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(x)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("held-out R^2")
    ax.set_title("Joint multi-output modeling vs. independent models (same lag_days=2 frame)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_gaussian_process(
    df_toc: pd.DataFrame, toc_features: list[str], df_alk: pd.DataFrame, alk_features: list[str], out_path: Path
) -> None:
    """Fig 23: Gaussian Process predictions with a 95% uncertainty band
    (+-1.96 std) -- the one model family here that returns calibrated
    uncertainty alongside a point estimate, which fits "actionable time to
    prepare" better than a bare point forecast."""
    toc_result = fit_gaussian_process(df_toc, toc_features, "TOC_mg_L")
    alk_result = fit_gaussian_process(df_alk, alk_features, "Alk_mg_L")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, result, name in [(axes[0], toc_result, "TOC_mg_L"), (axes[1], alk_result, "Alk_mg_L")]:
        ax.plot(result.actual.index, result.actual, color="tab:blue", linewidth=1.0, label="actual")
        ax.plot(result.predictions.index, result.predictions, color="tab:green", linewidth=1.0, label="GP mean")
        ax.fill_between(
            result.predictions.index,
            result.predictions - 1.96 * result.std,
            result.predictions + 1.96 * result.std,
            color="tab:green", alpha=0.2, label="95% band",
        )
        ax.set_title(f"{name} (R^2={result.r2:.3f}, mean std={result.std.mean():.3f})")
        ax.legend(fontsize=8)
    fig.suptitle("Gaussian Process regression: predictions with uncertainty bands")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_sarimax(df_toc: pd.DataFrame, df_alk: pd.DataFrame, out_path: Path) -> None:
    """Fig 24: SARIMAX forecast vs. actual for both targets -- a
    time-series-native alternative to this catalog's "lag as an engineered
    feature" approach used everywhere else."""
    toc_result = fit_sarimax_baseline(df_toc, "TOC_mg_L", exog_cols=["turb_flow"], order=(1, 0, 1))
    alk_result = fit_sarimax_baseline(df_alk, "Alk_mg_L", exog_cols=["Specific_Cond_Mean"], order=(1, 0, 1))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, result, name in [(axes[0], toc_result, "TOC_mg_L"), (axes[1], alk_result, "Alk_mg_L")]:
        ax.plot(result.actual.index, result.actual, color="tab:blue", linewidth=1.0, label="actual")
        ax.plot(
            result.predictions.index, result.predictions, color="tab:red", linewidth=1.0,
            label=f"SARIMAX{result.order} forecast",
        )
        ax.set_title(f"{name} (R^2={result.r2:.3f})")
        ax.legend(fontsize=8)
    fig.suptitle("SARIMAX: time-series-native forecast vs. actual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

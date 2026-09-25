"""Figures 27-30: this catalog's lag-tuning and NOAA-ENSO follow-up work --
per-source/hybrid lag tuning, the precip-extra-days grid behind it, whether
the best model family also wins on the hybrid-lag frame, and the NOAA ENSO
(ONI) experiment. Split out of visualize.py to keep files under the repo's
file-length gate; see visualize.py for the entry point (`main()`), and
analyze_parameters.py / analyze_enso.py for the matching report tables
these numbers also appear in.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analyze_enso import ALK_FEATURES as ENSO_ALK_FEATURES
from analyze_enso import ONI_PATH, TOC_FEATURES as ENSO_TOC_FEATURES
from data_loader import build_dataset, build_dataset_per_source_lag
from enso_loader import load_oni, oni_feature_for_dates
from models import (
    fit_gradient_boosting_importance,
    fit_random_forest_importance,
    fit_svr_baseline,
)
from models_advanced import fit_tuned_gradient_boosting

# Found by analyze_parameters._precip_extra_lag_grid_table's 0-7 day scan.
BEST_TOC_PRECIP_EXTRA_DAYS = 4
BEST_ALK_PRECIP_EXTRA_DAYS = 6


def plot_lag_approach_comparison(
    toc_features: list[str], alk_features: list[str], data_dir: Path, out_path: Path
) -> None:
    """Fig 27: held-out R^2 across the three lag approaches tried this
    round -- this catalog's uniform lag, Jake's own fixed per-source lag
    (+2 days for precip), and the tuned hybrid (+4 TOC / +6 Alk) -- for
    both targets side by side."""
    uniform_toc = fit_random_forest_importance(build_dataset(data_dir, lag_days=2), toc_features, "TOC_mg_L").r2
    fixed_toc = fit_random_forest_importance(
        build_dataset_per_source_lag(data_dir, lag_days=2, precip_extra_days=2), toc_features, "TOC_mg_L"
    ).r2
    hybrid_toc = fit_random_forest_importance(
        build_dataset_per_source_lag(data_dir, lag_days=2, precip_extra_days=BEST_TOC_PRECIP_EXTRA_DAYS),
        toc_features,
        "TOC_mg_L",
    ).r2

    uniform_alk = fit_random_forest_importance(build_dataset(data_dir, lag_days=4), alk_features, "Alk_mg_L").r2
    fixed_alk = fit_random_forest_importance(
        build_dataset_per_source_lag(data_dir, lag_days=4, precip_extra_days=2), alk_features, "Alk_mg_L"
    ).r2
    hybrid_alk = fit_random_forest_importance(
        build_dataset_per_source_lag(data_dir, lag_days=4, precip_extra_days=BEST_ALK_PRECIP_EXTRA_DAYS),
        alk_features,
        "Alk_mg_L",
    ).r2

    labels = ["Uniform\n(this catalog)", "Jake's fixed\n(+2 days)", "Hybrid\n(tuned)"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    axes[0].bar(labels, [uniform_toc, fixed_toc, hybrid_toc], color=["tab:gray", "tab:orange", "tab:green"])
    axes[0].set_title("TOC_mg_L")
    axes[0].set_ylabel("held-out R^2")
    axes[1].bar(labels, [uniform_alk, fixed_alk, hybrid_alk], color=["tab:gray", "tab:orange", "tab:green"])
    axes[1].set_title("Alk_mg_L")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
    fig.suptitle("Lag approach comparison (random forest, held-out R^2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_precip_extra_lag_grid(
    toc_features: list[str], alk_features: list[str], data_dir: Path, out_path: Path
) -> None:
    """Fig 28: held-out R^2 vs. precip_extra_days (0-7), the scan behind the
    hybrid-lag result -- each target's own base lag_days held fixed (2 for
    TOC, 4 for Alk) while precip's extra shift varies."""
    extra_days_grid = [0, 1, 2, 3, 4, 5, 6, 7]
    toc_scores = []
    alk_scores = []
    for extra in extra_days_grid:
        df_toc = build_dataset_per_source_lag(data_dir, lag_days=2, precip_extra_days=extra)
        df_alk = build_dataset_per_source_lag(data_dir, lag_days=4, precip_extra_days=extra)
        toc_scores.append(fit_random_forest_importance(df_toc, toc_features, "TOC_mg_L").r2)
        alk_scores.append(fit_random_forest_importance(df_alk, alk_features, "Alk_mg_L").r2)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(extra_days_grid, toc_scores, marker="o", color="tab:green", label="TOC_mg_L")
    ax.plot(extra_days_grid, alk_scores, marker="o", color="tab:blue", label="Alk_mg_L")
    ax.axvline(BEST_TOC_PRECIP_EXTRA_DAYS, color="tab:green", linestyle=":", alpha=0.5)
    ax.axvline(BEST_ALK_PRECIP_EXTRA_DAYS, color="tab:blue", linestyle=":", alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("precip_extra_days (beyond each target's own base lag)")
    ax.set_ylabel("held-out R^2 (random forest)")
    ax.set_title("Precip-extra-lag grid search (dotted lines: best per target)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_hybrid_lag_model_family(
    toc_features: list[str], alk_features: list[str], data_dir: Path, out_path: Path
) -> None:
    """Fig 29: does the best model family also win on the hybrid-lag frame?
    Random forest, SVR, and gradient boosting (untuned + GridSearchCV-tuned)
    refit on each target's own hybrid-lag frame."""
    df_toc = build_dataset_per_source_lag(data_dir, lag_days=2, precip_extra_days=BEST_TOC_PRECIP_EXTRA_DAYS)
    df_alk = build_dataset_per_source_lag(data_dir, lag_days=4, precip_extra_days=BEST_ALK_PRECIP_EXTRA_DAYS)

    toc_scores = [
        fit_random_forest_importance(df_toc, toc_features, "TOC_mg_L").r2,
        fit_svr_baseline(df_toc, toc_features, "TOC_mg_L").r2,
        fit_gradient_boosting_importance(df_toc, toc_features, "TOC_mg_L").r2,
        fit_tuned_gradient_boosting(df_toc, toc_features, "TOC_mg_L").r2,
    ]
    alk_scores = [
        fit_random_forest_importance(df_alk, alk_features, "Alk_mg_L").r2,
        fit_svr_baseline(df_alk, alk_features, "Alk_mg_L").r2,
        fit_gradient_boosting_importance(df_alk, alk_features, "Alk_mg_L").r2,
        fit_tuned_gradient_boosting(df_alk, alk_features, "Alk_mg_L").r2,
    ]
    labels = ["Random\nforest", "SVR\n(RBF)", "Gradient\nboosting", "Tuned\nboosting"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    axes[0].bar(labels, toc_scores, color=["tab:purple", "tab:red", "tab:orange", "tab:brown"])
    axes[0].set_title("TOC_mg_L")
    axes[0].set_ylabel("held-out R^2")
    axes[1].bar(labels, alk_scores, color=["tab:purple", "tab:red", "tab:orange", "tab:brown"])
    axes[1].set_title("Alk_mg_L")
    for ax in axes:
        ax.axhline(0, color="black", linewidth=0.8)
    fig.suptitle("Model family comparison on the hybrid-lag frame")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_enso_experiment(data_dir: Path, out_path: Path) -> None:
    """Fig 30: does NOAA's Oceanic Nino Index (ONI) improve predictions?
    Held-out R^2 with vs. without ONI_prev_month, on both the uniform-lag
    baseline and the hybrid-lag frame, for both targets."""
    oni = load_oni(ONI_PATH)

    def scored(df: pd.DataFrame, features: list[str], target: str) -> tuple[float, float]:
        without_oni = fit_random_forest_importance(df, features, target).r2
        with_oni = df.copy()
        with_oni["ONI_prev_month"] = oni_feature_for_dates(oni, with_oni.index)
        return without_oni, fit_random_forest_importance(with_oni, [*features, "ONI_prev_month"], target).r2

    toc_uniform = scored(build_dataset(data_dir, lag_days=2), ENSO_TOC_FEATURES, "TOC_mg_L")
    alk_uniform = scored(build_dataset(data_dir, lag_days=4), ENSO_ALK_FEATURES, "Alk_mg_L")
    toc_hybrid = scored(
        build_dataset_per_source_lag(data_dir, lag_days=2, precip_extra_days=BEST_TOC_PRECIP_EXTRA_DAYS),
        ENSO_TOC_FEATURES,
        "TOC_mg_L",
    )
    alk_hybrid = scored(
        build_dataset_per_source_lag(data_dir, lag_days=4, precip_extra_days=BEST_ALK_PRECIP_EXTRA_DAYS),
        ENSO_ALK_FEATURES,
        "Alk_mg_L",
    )

    groups = ["TOC\nuniform", "TOC\nhybrid", "Alk\nuniform", "Alk\nhybrid"]
    without_scores = [toc_uniform[0], toc_hybrid[0], alk_uniform[0], alk_hybrid[0]]
    with_scores = [toc_uniform[1], toc_hybrid[1], alk_uniform[1], alk_hybrid[1]]

    x = range(len(groups))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width / 2 for i in x], without_scores, width, label="without ONI", color="tab:gray")
    ax.bar([i + width / 2 for i in x], with_scores, width, label="with ONI", color="tab:cyan")
    ax.set_xticks(list(x), labels=groups)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("held-out R^2 (random forest)")
    ax.set_title("NOAA Oceanic Nino Index (ONI): does it help?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

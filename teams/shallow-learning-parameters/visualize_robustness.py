"""Figures 32-35: the four "other ideas for the limited window" experiments
in analyze_robustness.py (regularization, CV stability, delta-target,
two-stage chain). Split out to keep files under the repo's file-length
gate; see visualize.py for the entry point (`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_robustness import (
    compute_cv_stability_scores,
    compute_delta_target_scores,
    compute_regularization_comparison,
    compute_two_stage_chain_scores,
)


def plot_regularization_comparison(
    data_dir: Path,
    sonde_readings: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
    out_path: Path,
) -> None:
    """Fig 32: random forest/SVR (unregularized) vs Ridge/Lasso (shrunk)
    on the sonde window's least-bad feature set -- does shrinking
    parameters fix the overfitting fig 31 diagnosed?"""
    scores = compute_regularization_comparison(data_dir, sonde_readings, national_features, lag_days)
    models = ["rf_r2", "svr_r2", "ridge_r2", "lasso_r2"]
    labels = ["random forest", "SVR", "Ridge", "Lasso"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, target in zip(axes, ("TOC_mg_L", "Alk_mg_L")):
        row = scores[scores["target"] == target].iloc[0]
        ax.bar(labels, [row[m] for m in models], color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(target)
    axes[0].set_ylabel("held-out R²")

    fig.suptitle("Limited window: does shrinking parameters (Ridge/Lasso) beat random forest/SVR?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_cv_stability(
    data_dir: Path,
    sonde_readings: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
    out_path: Path,
) -> None:
    """Fig 33: per-fold TimeSeriesSplit R^2 for each target -- is the
    single 50/50-split number elsewhere in this catalog a stable property
    of the model, or one lucky/unlucky draw from a much wider spread?"""
    scores = compute_cv_stability_scores(data_dir, sonde_readings, national_features, lag_days)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if scores.empty:
        ax.text(0.5, 0.5, "Too few rows for cross-validation", ha="center", va="center")
    else:
        targets = sorted(scores["target"].unique())
        data = [scores.loc[scores["target"] == t, "r2"].to_numpy() for t in targets]
        ax.boxplot(data, tick_labels=targets)
        for i, values in enumerate(data, start=1):
            ax.scatter(np.full(len(values), i), values, color="black", zorder=3, s=15)
        ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("held-out R² per fold")
    ax.set_title("Limited window: TimeSeriesSplit fold-to-fold spread (Ridge)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_delta_target_comparison(
    data_dir: Path,
    sonde_readings: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
    out_path: Path,
) -> None:
    """Fig 34: predicting the absolute level vs. the day-over-day delta --
    does removing the year-scale signal the window never saw help?"""
    scores = compute_delta_target_scores(data_dir, sonde_readings, national_features, lag_days)
    framings = ["level", "day-over-day delta"]
    x = np.arange(len(framings))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, target in zip(axes, ("TOC_mg_L", "Alk_mg_L")):
        rows = scores[scores["target"] == target].set_index("framing").loc[framings]
        ax.bar(x - width / 2, rows["rf_r2"], width, label="random forest", color="tab:blue")
        ax.bar(x + width / 2, rows["ridge_r2"], width, label="Ridge", color="tab:green")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(framings)
        ax.set_title(target)
    axes[0].set_ylabel("held-out R²")
    axes[0].legend(fontsize=8)

    fig.suptitle("Limited window: predicting levels vs. day-over-day deltas")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_two_stage_chain(data_dir: Path, sonde_readings: pd.DataFrame, out_path: Path) -> None:
    """Fig 35: does routing TOC prediction through the sonde's 390
    higher-frequency casts (stage 1: casts -> gage turbidity) beat using
    the real gage reading directly (stage 2)?"""
    scores = compute_two_stage_chain_scores(data_dir, sonde_readings)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [s.replace(": ", ":\n") for s in scores["stage"]]
    colors = ["tab:purple", "tab:blue", "tab:orange"]
    ax.bar(labels, scores["r2"], color=colors[: len(scores)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("held-out R²")
    ax.tick_params(axis="x", labelsize=8)
    ax.set_title("Two-stage chain: sonde casts -> gage turbidity -> TOC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

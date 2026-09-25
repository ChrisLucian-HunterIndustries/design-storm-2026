"""Report sections for this catalog's lag-tuning experiments: the plain
lag-day grid search, whether Jake's per-source lag (precip shifted further
than other sources) beats a flat 2/4-day lag, the tuned hybrid that beats
both, and whether the best model family also wins on that hybrid-lag frame.

Split out from analyze_parameters.py (already near this repo's file-length
gate) the same way analyze_advanced.py/analyze_dosing.py/analyze_sonde.py
were -- its section is appended by analyze_parameters.main() the same way.
"""
from __future__ import annotations

from pathlib import Path

from data_loader import build_dataset, build_dataset_per_source_lag
from models import fit_gradient_boosting_importance, fit_random_forest_importance, fit_svr_baseline
from models_advanced import fit_tuned_gradient_boosting

LAG_DAYS_GRID = [0, 1, 2, 3, 4, 5, 7, 10]
PRECIP_EXTRA_DAYS_GRID = [0, 1, 2, 3, 4, 5, 6, 7]


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _lag_day_grid_search_table(data_dir: Path, toc_features: list[str], alk_features: list[str]) -> list[str]:
    """Deck slide 9: "try different... lag-times... to see if they can
    improve performance." Rebuilds the whole dataset at each candidate lag
    (this also re-times the `turb_flow` engineered feature) and refits the
    same random forest used in the main results table above."""
    lines = ["| lag (days) | TOC_mg_L held-out R^2 | Alk_mg_L held-out R^2 |", "|---:|---:|---:|"]
    for lag in LAG_DAYS_GRID:
        df = build_dataset(data_dir, lag_days=lag)
        toc_r2 = fit_random_forest_importance(df, toc_features, "TOC_mg_L").r2
        alk_r2 = fit_random_forest_importance(df, alk_features, "Alk_mg_L").r2
        lines.append(f"| {lag} | {_fmt(toc_r2)} | {_fmt(alk_r2)} |")
    return lines


def _per_source_lag_table(data_dir: Path, toc_features: list[str], alk_features: list[str]) -> list[str]:
    """Jake's own notebooks (scripts/TOC_SoftSensor.ipynb,
    scripts/Alkalinity_Soft_Sensor.ipynb) lag NOAA precip 2 days longer than
    the other sources (data_loader.build_dataset_per_source_lag). Compares
    that per-source approach against this catalog's uniform-lag build_dataset,
    same random forest and feature list, to answer whether Jake's approach
    actually scores better than a flat 2/4-day lag."""
    lines = ["| target | lag approach | held-out R^2 |", "|---|---|---:|"]
    uniform_toc = fit_random_forest_importance(
        build_dataset(data_dir, lag_days=2), toc_features, "TOC_mg_L"
    ).r2
    per_source_toc = fit_random_forest_importance(
        build_dataset_per_source_lag(data_dir, lag_days=2), toc_features, "TOC_mg_L"
    ).r2
    uniform_alk = fit_random_forest_importance(
        build_dataset(data_dir, lag_days=4), alk_features, "Alk_mg_L"
    ).r2
    per_source_alk = fit_random_forest_importance(
        build_dataset_per_source_lag(data_dir, lag_days=4), alk_features, "Alk_mg_L"
    ).r2
    lines.append(f"| TOC_mg_L | uniform (lag_days=2 for every source) | {_fmt(uniform_toc)} |")
    lines.append(f"| TOC_mg_L | per-source (Jake's: precip lagged 2 days further) | {_fmt(per_source_toc)} |")
    lines.append(f"| Alk_mg_L | uniform (lag_days=4 for every source) | {_fmt(uniform_alk)} |")
    lines.append(f"| Alk_mg_L | per-source (Jake's: precip lagged 2 days further) | {_fmt(per_source_alk)} |")
    return lines


def _precip_extra_lag_grid_table(
    data_dir: Path, toc_features: list[str], alk_features: list[str]
) -> tuple[list[str], int, int]:
    """A hybrid tuned per target: instead of assuming precip needs 0 extra
    days (uniform build_dataset) or exactly Jake's fixed +2, scan every
    candidate `precip_extra_days` for each target's own base lag_days (2 for
    TOC, 4 for Alk) and pick whichever scores best, independently per
    target. Same random forest/feature list as the tables above. Returns the
    table lines plus the winning precip_extra_days for TOC and Alk, so the
    caller can build a dataset with that hybrid choice."""
    lines = [
        "| precip extra days | TOC_mg_L held-out R^2 | Alk_mg_L held-out R^2 |",
        "|---:|---:|---:|",
    ]
    toc_scores: dict[int, float] = {}
    alk_scores: dict[int, float] = {}
    for extra in PRECIP_EXTRA_DAYS_GRID:
        df_toc = build_dataset_per_source_lag(data_dir, lag_days=2, precip_extra_days=extra)
        df_alk = build_dataset_per_source_lag(data_dir, lag_days=4, precip_extra_days=extra)
        toc_scores[extra] = fit_random_forest_importance(df_toc, toc_features, "TOC_mg_L").r2
        alk_scores[extra] = fit_random_forest_importance(df_alk, alk_features, "Alk_mg_L").r2
        lines.append(f"| {extra} | {_fmt(toc_scores[extra])} | {_fmt(alk_scores[extra])} |")

    best_toc_extra = max(toc_scores, key=lambda k: toc_scores[k])
    best_alk_extra = max(alk_scores, key=lambda k: alk_scores[k])
    lines.append("")
    lines.append(
        f"Best precip_extra_days for TOC_mg_L: {best_toc_extra} "
        f"(R^2={_fmt(toc_scores[best_toc_extra])})"
    )
    lines.append(
        f"Best precip_extra_days for Alk_mg_L: {best_alk_extra} "
        f"(R^2={_fmt(alk_scores[best_alk_extra])})"
    )
    return lines, best_toc_extra, best_alk_extra


def _hybrid_lag_model_family_table(
    data_dir: Path, toc_features: list[str], alk_features: list[str], best_toc_extra: int, best_alk_extra: int
) -> list[str]:
    """Does the best lag also combine with the best model family, or does
    picking a better lag only help the random forest specifically? Refits
    every model family already compared on the uniform-lag frame
    (`_model_family_table`) on each target's own hybrid-lag frame instead."""
    df_toc_hybrid = build_dataset_per_source_lag(data_dir, lag_days=2, precip_extra_days=best_toc_extra)
    df_alk_hybrid = build_dataset_per_source_lag(data_dir, lag_days=4, precip_extra_days=best_alk_extra)

    toc_rf_r2 = fit_random_forest_importance(df_toc_hybrid, toc_features, "TOC_mg_L").r2
    toc_svr_r2 = fit_svr_baseline(df_toc_hybrid, toc_features, "TOC_mg_L").r2
    toc_gbr_r2 = fit_gradient_boosting_importance(df_toc_hybrid, toc_features, "TOC_mg_L").r2
    toc_tuned_gbr_r2 = fit_tuned_gradient_boosting(df_toc_hybrid, toc_features, "TOC_mg_L").r2
    alk_rf_r2 = fit_random_forest_importance(df_alk_hybrid, alk_features, "Alk_mg_L").r2
    alk_svr_r2 = fit_svr_baseline(df_alk_hybrid, alk_features, "Alk_mg_L").r2
    alk_gbr_r2 = fit_gradient_boosting_importance(df_alk_hybrid, alk_features, "Alk_mg_L").r2
    alk_tuned_gbr_r2 = fit_tuned_gradient_boosting(df_alk_hybrid, alk_features, "Alk_mg_L").r2

    return [
        "| target | model (on the hybrid-lag frame) | held-out R^2 |",
        "|---|---|---:|",
        f"| TOC_mg_L | Random forest | {_fmt(toc_rf_r2)} |",
        f"| TOC_mg_L | SVR (RBF kernel, scaled features) | {_fmt(toc_svr_r2)} |",
        f"| TOC_mg_L | Gradient boosting (untuned) | {_fmt(toc_gbr_r2)} |",
        f"| TOC_mg_L | Gradient boosting (GridSearchCV-tuned) | {_fmt(toc_tuned_gbr_r2)} |",
        f"| Alk_mg_L | Random forest | {_fmt(alk_rf_r2)} |",
        f"| Alk_mg_L | SVR (RBF kernel, scaled features) | {_fmt(alk_svr_r2)} |",
        f"| Alk_mg_L | Gradient boosting (untuned) | {_fmt(alk_gbr_r2)} |",
        f"| Alk_mg_L | Gradient boosting (GridSearchCV-tuned) | {_fmt(alk_tuned_gbr_r2)} |",
    ]


def build_lag_experiments_section(data_dir: Path, toc_features: list[str], alk_features: list[str]) -> list[str]:
    """All four lag-tuning sections, in the order parameters.md discusses
    them: plain grid search, per-source lag, tuned hybrid, then whether the
    best model family also wins on the hybrid frame."""
    lines: list[str] = []

    lines.append("\n## Lag-day grid search (Scenario 1: \"try different... lag-times\")\n")
    lines.append(
        "Same random forest and feature list as above, refit at each candidate lag. "
        "Elsewhere in this document TOC uses lag_days=2 and Alk_mg_L uses lag_days=4 "
        "(guide.md section 14); compare those rows below against the rest of the grid.\n"
    )
    lines += _lag_day_grid_search_table(data_dir, toc_features, alk_features)

    lines.append("\n## Per-source lag: does Jake's approach beat a flat 2/4-day lag?\n")
    lines.append(
        "Jake's original notebooks lag NOAA precip 2 days further than the gage/DWR/SNOTEL "
        "sources (4 vs. 2 for TOC, 6 vs. 4 for alkalinity) to cover NOAA's own reporting delay, "
        "rather than shifting every predictor by one uniform number the way this catalog's own "
        "build_dataset does. Same random forest and feature list as the model-family table above.\n"
    )
    lines += _per_source_lag_table(data_dir, toc_features, alk_features)

    lines.append("\n## Hybrid lag: tuning precip's extra lag per target beats both fixed choices\n")
    lines.append(
        "Neither fixed choice above is forced: uniform lag is precip_extra_days=0, Jake's own "
        "notebooks use precip_extra_days=2 for both targets. Scanning every candidate value "
        "per target picks whichever wins independently, rather than assuming one number fits "
        "both. Same random forest and feature list as the tables above; the usual caveat from "
        "the lag-day grid search applies here too -- a single 50/50-split R^2 is not a fully "
        "stable property of the model, so treat the winning value as a direction, not a promise.\n"
    )
    precip_grid_lines, best_toc_extra, best_alk_extra = _precip_extra_lag_grid_table(
        data_dir, toc_features, alk_features
    )
    lines += precip_grid_lines
    hybrid_toc_r2 = fit_random_forest_importance(
        build_dataset_per_source_lag(data_dir, lag_days=2, precip_extra_days=best_toc_extra),
        toc_features,
        "TOC_mg_L",
    ).r2
    hybrid_alk_r2 = fit_random_forest_importance(
        build_dataset_per_source_lag(data_dir, lag_days=4, precip_extra_days=best_alk_extra),
        alk_features,
        "Alk_mg_L",
    ).r2
    lines.append("")
    lines.append(
        f"Hybrid (TOC precip_extra_days={best_toc_extra}, Alk precip_extra_days={best_alk_extra}) "
        f"vs. the two fixed choices:"
    )
    lines.append("| target | uniform (extra=0) | Jake's fixed (extra=2) | hybrid (tuned) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| TOC_mg_L | 0.334 | 0.599 | {_fmt(hybrid_toc_r2)} |")
    lines.append(f"| Alk_mg_L | 0.234 | 0.184 | {_fmt(hybrid_alk_r2)} |")

    lines.append("\n## Does the best model family also win on the best-lag frame?\n")
    lines.append(
        "The model-family comparison above (SVR beating the random forest) and the hybrid-lag "
        "result above it were each found on a *different* frame -- SVR was only ever tried on "
        "the uniform lag_days=2/4 frame, not the hybrid-lag frame that beat it. Refits every "
        "model family on each target's own hybrid-lag frame to check whether the two "
        "improvements stack.\n"
    )
    lines += _hybrid_lag_model_family_table(data_dir, toc_features, alk_features, best_toc_extra, best_alk_extra)

    return lines

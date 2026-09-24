"""Compute grounded numbers on the real Design Storm datasets: correlations,
held-out R^2, feature importances, cluster sizes, and classifier scores.

Run from this directory: `python analyze_parameters.py`. Writes
results/parameter_summary.md, which is what parameters.md's numbers are
drawn from -- nothing in that document is invented (see AGENTS.md: "Never
invent numbers").
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import (
    build_dataset,
    feature_columns,
    load_dwr_telemetry,
    load_snowpack,
    load_target,
    with_calendar_year_and_doy,
)
from models import (
    cluster_hydrologic_regimes,
    fit_linear_baseline,
    fit_random_forest_importance,
    fit_threshold_classifier,
)

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

    output_path = RESULTS_DIR / "parameter_summary.md"
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

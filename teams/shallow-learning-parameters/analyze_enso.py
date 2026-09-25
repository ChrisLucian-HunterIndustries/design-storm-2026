"""Experiment: does NOAA's Oceanic Nino Index (ONI, a public NOAA/PSL
climate dataset not otherwise used anywhere in this catalog) improve
TOC/alkalinity predictions? See enso_loader.py for how it's parsed and
lag-aligned, fetch_oni.py for how oni.txt was obtained.

Tested on both the plain uniform-lag frame (this catalog's long-standing
baseline, R^2=0.334/0.234) and the hybrid-lag frame found to be the current
best (R^2=0.650/0.241) -- so this answers not just "does ONI help at all"
but "does it still help once the lag itself is already tuned."
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import build_dataset, build_dataset_per_source_lag
from enso_loader import load_oni, oni_feature_for_dates
from models import fit_random_forest_importance

ONI_PATH = Path(__file__).resolve().parent / "oni.txt"

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


def _add_oni_feature(df: pd.DataFrame, oni: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ONI_prev_month"] = oni_feature_for_dates(oni, out.index)
    return out


def _oni_correlation_table(df_toc: pd.DataFrame, df_alk: pd.DataFrame, oni: pd.DataFrame) -> list[str]:
    """A direct grounding check before touching any model: is ONI even
    correlated with either target at all, on its own?"""
    toc_oni = oni_feature_for_dates(oni, df_toc.index)
    alk_oni = oni_feature_for_dates(oni, df_alk.index)
    toc_corr = pd.concat([df_toc["TOC_mg_L"], toc_oni], axis=1).corr().iloc[0, 1]
    alk_corr = pd.concat([df_alk["Alk_mg_L"], alk_oni], axis=1).corr().iloc[0, 1]
    return [
        "| target | Pearson r with ONI (previous month) |",
        "|---|---:|",
        f"| TOC_mg_L | {_fmt(toc_corr)} |",
        f"| Alk_mg_L | {_fmt(alk_corr)} |",
    ]


def _oni_model_table(data_dir: Path, oni: pd.DataFrame) -> list[str]:
    """Held-out R^2 with vs. without ONI_prev_month, same random forest,
    on both the uniform-lag baseline frame and the hybrid-lag best frame
    (TOC precip_extra_days=4, Alk precip_extra_days=6 -- see
    analyze_parameters._precip_extra_lag_grid_table)."""
    lines = ["| target | frame | features | held-out R^2 |", "|---|---|---|---:|"]

    frames = [
        ("uniform lag", build_dataset(data_dir, lag_days=2), build_dataset(data_dir, lag_days=4)),
        (
            "hybrid lag (tuned)",
            build_dataset_per_source_lag(data_dir, lag_days=2, precip_extra_days=4),
            build_dataset_per_source_lag(data_dir, lag_days=4, precip_extra_days=6),
        ),
    ]
    for frame_name, df_toc, df_alk in frames:
        toc_r2 = fit_random_forest_importance(df_toc, TOC_FEATURES, "TOC_mg_L").r2
        alk_r2 = fit_random_forest_importance(df_alk, ALK_FEATURES, "Alk_mg_L").r2
        df_toc_oni = _add_oni_feature(df_toc, oni)
        df_alk_oni = _add_oni_feature(df_alk, oni)
        toc_oni_r2 = fit_random_forest_importance(
            df_toc_oni, [*TOC_FEATURES, "ONI_prev_month"], "TOC_mg_L"
        ).r2
        alk_oni_r2 = fit_random_forest_importance(
            df_alk_oni, [*ALK_FEATURES, "ONI_prev_month"], "Alk_mg_L"
        ).r2
        lines.append(f"| TOC_mg_L | {frame_name} | without ONI | {_fmt(toc_r2)} |")
        lines.append(f"| TOC_mg_L | {frame_name} | with ONI | {_fmt(toc_oni_r2)} |")
        lines.append(f"| Alk_mg_L | {frame_name} | without ONI | {_fmt(alk_r2)} |")
        lines.append(f"| Alk_mg_L | {frame_name} | with ONI | {_fmt(alk_oni_r2)} |")
    return lines


def build_enso_experiment_section(data_dir: Path) -> list[str]:
    oni = load_oni(ONI_PATH)
    df_toc = build_dataset(data_dir, lag_days=2)
    df_alk = build_dataset(data_dir, lag_days=4)

    lines = [
        "\n## NOAA ENSO experiment: does the Oceanic Nino Index help? (new dataset, not in data/)\n",
        "NOAA/PSL's Oceanic Nino Index (ONI, https://psl.noaa.gov/data/correlation/oni.data) is a "
        "public NOAA dataset not otherwise used anywhere in this catalog -- a monthly, basin-scale "
        "climate index (El Nino/La Nina strength), rather than a local daily watershed reading like "
        "everything else here. Snapshot saved to oni.txt (fetch_oni.py); ONI_prev_month uses each "
        "day's most recently *fully observed* month, one month behind, avoiding lookahead "
        "(enso_loader.py).\n",
    ]
    lines += _oni_correlation_table(df_toc, df_alk, oni)
    lines.append("")
    lines += _oni_model_table(data_dir, oni)
    return lines

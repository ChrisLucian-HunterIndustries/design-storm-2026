"""Experiment: does combining Denver Water's own measured outflow (the
COND20CO/COND26CO conduits carrying water from Strontia Springs Reservoir
to the Foothills Treatment Plant -- the same plant data/FoothillsInfluent.csv
is drawn from) with the MTBS burn-scar recency feature (analyze_burn_scar.py)
improve TOC/alkalinity predictions, beyond what the existing inflow feature
(Flow_CFS, the river gage above the reservoir) already captures?

See denver_outflow_loader.py for the outflow data (fetch_denver_outflow.py)
and burn_scar_loader.py for the fire data (fetch_mtbs.py).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analyze_burn_scar import BASIN_PATH, MTBS_PATH
from burn_scar_loader import days_since_fire_feature, filter_fires_in_basin, load_mtbs_fires
from data_loader import build_dataset
from denver_outflow_loader import load_conduit_discharge
from models import fit_random_forest_importance

OUTFLOW_PATH = Path(__file__).resolve().parent / "denver_conduit_outflow.json"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _add_outflow_feature(df: pd.DataFrame, outflow: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """Join Denver's own outflow onto the dataset with the same lookahead-safe
    forward shift `build_dataset` already applied to every other predictor."""
    shifted = outflow[["Outflow_CFS"]].shift(lag_days, freq="D")
    return df.join(shifted, how="left")


def _add_fire_feature(df: pd.DataFrame, fires_in_basin: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["days_since_fire"] = days_since_fire_feature(fires_in_basin, out.index)
    return out


def _denver_outflow_model_table(
    data_dir: Path,
    outflow_path: Path,
    fires_in_basin: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
) -> list[str]:
    """Held-out R^2 for baseline vs. +Outflow_CFS vs. +days_since_fire vs.
    both combined, same random forest, on the full multi-year record."""
    outflow = load_conduit_discharge(outflow_path)
    lines = ["| target | features | held-out R² |", "|---|---|---:|"]
    for target in ("TOC_mg_L", "Alk_mg_L"):
        lag = lag_days[target]
        cols = national_features[target]
        df = build_dataset(data_dir, lag_days=lag)
        base_r2 = fit_random_forest_importance(df, cols, target).r2

        df_outflow = _add_outflow_feature(df, outflow, lag)
        outflow_r2 = fit_random_forest_importance(df_outflow, [*cols, "Outflow_CFS"], target).r2

        df_fire = _add_fire_feature(df, fires_in_basin)
        fire_r2 = fit_random_forest_importance(df_fire, [*cols, "days_since_fire"], target).r2

        df_combined = _add_fire_feature(df_outflow, fires_in_basin)
        combined_r2 = fit_random_forest_importance(
            df_combined, [*cols, "Outflow_CFS", "days_since_fire"], target
        ).r2

        lines.append(f"| {target} | baseline | {_fmt(base_r2)} |")
        lines.append(f"| {target} | + Outflow_CFS | {_fmt(outflow_r2)} |")
        lines.append(f"| {target} | + days_since_fire | {_fmt(fire_r2)} |")
        lines.append(f"| {target} | + Outflow_CFS + days_since_fire | {_fmt(combined_r2)} |")
    return lines


def build_denver_outflow_section(
    data_dir: Path, national_features: dict[str, list[str]], lag_days: dict[str, int]
) -> list[str]:
    fires_in_basin = filter_fires_in_basin(load_mtbs_fires(MTBS_PATH), BASIN_PATH)
    lines = [
        "\n## Denver Water's own outflow, combined with fire history (Scenario 1: \"combine fire data with "
        "inflow and outflow data from Denver's web API\")\n",
        "COND20CO (\"DENVER WATER CONDUIT NO 20\") and COND26CO (\"DW CONDUIT 26\", at the dam itself) are "
        "Denver Water's own measured discharge out of Strontia Springs Reservoir, telemetered through the "
        "same public Colorado DWR CDSS REST API this repo already uses for reservoir storage (see "
        "water-system-3d/fetch_storage_history.py) -- a different parameter (DISCHRG) and different "
        "stations. Their sum, `Outflow_CFS`, is combined here with the existing inflow feature (`Flow_CFS`, "
        "the river gage above the reservoir) already in this catalog's baseline, and with the MTBS "
        "burn-scar recency feature (`days_since_fire`, see the section above) from the same drainage "
        "basin's real fire history:\n",
    ]
    lines += _denver_outflow_model_table(data_dir, OUTFLOW_PATH, fires_in_basin, national_features, lag_days)
    lines.append(
        "\n`+ Outflow_CFS` alone is a genuine, modest improvement over baseline for both targets -- a real "
        "measured release, distinct information from the upstream inflow gage already in the baseline "
        "features. The much larger jump from adding `days_since_fire` (and the negligible extra gain from "
        "combining it with outflow) carries the same caveat raised in the burn-scar section above: it is "
        "flagged as an unresolved, not-yet-trustworthy finding, not a confirmed result -- see that section's "
        "time-index-artifact controls before treating the combined-feature R^2 here as real.\n"
    )
    return lines

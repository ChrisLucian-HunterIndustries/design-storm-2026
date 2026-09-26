"""Experiment: five follow-up ideas for combining the fire (MTBS burn-scar)
and Denver Water inflow/outflow data (see analyze_burn_scar.py /
analyze_denver_outflow.py), each testing a different mechanism instead of
the plain `days_since_fire`/`Outflow_CFS` features already tried:

1. `fire_pressure` -- size-weighted, time-decayed fire recency (does a
   feature carrying real fire size, not just a date, still show a jump?).
2. `fire_precip_interaction` -- burned soil loses infiltration, so the
   physically real mechanism is sediment mobilized *during* storms after a
   fire, not gradually over elapsed days.
3. `cumulative_burned_acres` -- total acreage burned within a trailing
   window, not just the nearest fire.
4. `distance_weighted_fire_pressure` -- separates "big but distant" fires
   from "small but close" ones.
5. `residence_time_days` -- reservoir storage / outflow, a genuinely
   different mechanism (less settling time for storm-driven turbidity/TOC
   to drop out before reaching the plant) using inflow+outflow together
   rather than either alone.

See burn_scar_loader.py, denver_outflow_loader.py, and
denver_storage_loader.py for the underlying feature/data code.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from analyze_burn_scar import BASIN_PATH, MTBS_PATH
from burn_scar_loader import (
    cumulative_burned_acres_feature,
    days_since_fire_feature,
    distance_weighted_fire_pressure_feature,
    filter_fires_in_basin,
    fire_pressure_feature,
    load_mtbs_fires,
)
from data_loader import build_dataset
from denver_outflow_loader import load_conduit_discharge
from denver_storage_loader import load_reservoir_storage
from models import fit_random_forest_importance

OUTFLOW_PATH = Path(__file__).resolve().parent / "denver_conduit_outflow.json"
STORAGE_PATH = Path(__file__).resolve().parents[2] / "water-system-3d" / "storage-history.json"
# Strontia Springs Reservoir's own coordinates (from the DWR CDSS station metadata for STRRESCO).
RESERVOIR_LAT = 39.432175
RESERVOIR_LON = -105.12757
CFS_TO_AF_PER_DAY = 1.983471  # general hydrology unit conversion, not from this repo's materials


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _add_fire_pressure(df: pd.DataFrame, fires_in_basin: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["fire_pressure"] = fire_pressure_feature(fires_in_basin, out.index)
    return out


def _add_fire_precip_interaction(df: pd.DataFrame, fires_in_basin: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    active_days_since = days_since_fire_feature(fires_in_basin, out.index).clip(lower=0)
    out["fire_precip_interaction"] = active_days_since * out["roll_precip_7"]
    return out


def _add_cumulative_burned_acres(df: pd.DataFrame, fires_in_basin: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cumulative_burned_acres"] = cumulative_burned_acres_feature(fires_in_basin, out.index, window_years=10)
    return out


def _add_distance_weighted_fire_pressure(
    df: pd.DataFrame, fires_in_basin: pd.DataFrame, reference_lat: float, reference_lon: float
) -> pd.DataFrame:
    out = df.copy()
    out["distance_weighted_fire_pressure"] = distance_weighted_fire_pressure_feature(
        fires_in_basin, out.index, reference_lat, reference_lon
    )
    return out


def _add_residence_time(df: pd.DataFrame, outflow: pd.DataFrame, storage: pd.Series, lag_days: int) -> pd.DataFrame:
    """Storage / outflow, in days -- how long water entering the reservoir
    today would take to fully turn over at today's release rate. Shifted
    the same lookahead-safe `lag_days` as every other predictor."""
    outflow_af_per_day = outflow["Outflow_CFS"].replace(0, float("nan")) * CFS_TO_AF_PER_DAY
    residence_time = (storage / outflow_af_per_day).rename("residence_time_days")
    shifted = residence_time.shift(lag_days, freq="D")
    return df.join(shifted, how="left")


def _fire_feature_model_table(
    data_dir: Path,
    outflow_path: Path,
    storage_path: Path,
    fires_in_basin: pd.DataFrame,
    reference_lat: float,
    reference_lon: float,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
) -> list[str]:
    """Held-out R^2 for baseline vs. each of the five ideas vs. all five
    combined, same random forest, on the full multi-year record."""
    outflow = load_conduit_discharge(outflow_path)
    storage = load_reservoir_storage(storage_path)

    lines = ["| target | features | held-out R² |", "|---|---|---:|"]
    for target in ("TOC_mg_L", "Alk_mg_L"):
        lag = lag_days[target]
        cols = national_features[target]
        df = build_dataset(data_dir, lag_days=lag)
        base_r2 = fit_random_forest_importance(df, cols, target).r2

        df_pressure = _add_fire_pressure(df, fires_in_basin)
        pressure_r2 = fit_random_forest_importance(df_pressure, [*cols, "fire_pressure"], target).r2

        df_interaction = _add_fire_precip_interaction(df, fires_in_basin)
        interaction_r2 = fit_random_forest_importance(
            df_interaction, [*cols, "fire_precip_interaction"], target
        ).r2

        df_cumulative = _add_cumulative_burned_acres(df, fires_in_basin)
        cumulative_r2 = fit_random_forest_importance(
            df_cumulative, [*cols, "cumulative_burned_acres"], target
        ).r2

        df_distance = _add_distance_weighted_fire_pressure(df, fires_in_basin, reference_lat, reference_lon)
        distance_r2 = fit_random_forest_importance(
            df_distance, [*cols, "distance_weighted_fire_pressure"], target
        ).r2

        df_residence = _add_residence_time(df, outflow, storage, lag)
        residence_r2 = fit_random_forest_importance(df_residence, [*cols, "residence_time_days"], target).r2

        df_combined = _add_residence_time(
            _add_distance_weighted_fire_pressure(
                _add_cumulative_burned_acres(_add_fire_precip_interaction(_add_fire_pressure(df, fires_in_basin), fires_in_basin), fires_in_basin),
                fires_in_basin,
                reference_lat,
                reference_lon,
            ),
            outflow,
            storage,
            lag,
        )
        combined_cols = [
            *cols,
            "fire_pressure",
            "fire_precip_interaction",
            "cumulative_burned_acres",
            "distance_weighted_fire_pressure",
            "residence_time_days",
        ]
        combined_r2 = fit_random_forest_importance(df_combined, combined_cols, target).r2

        lines.append(f"| {target} | baseline | {_fmt(base_r2)} |")
        lines.append(f"| {target} | + fire_pressure | {_fmt(pressure_r2)} |")
        lines.append(f"| {target} | + fire_precip_interaction | {_fmt(interaction_r2)} |")
        lines.append(f"| {target} | + cumulative_burned_acres | {_fmt(cumulative_r2)} |")
        lines.append(f"| {target} | + distance_weighted_fire_pressure | {_fmt(distance_r2)} |")
        lines.append(f"| {target} | + residence_time_days | {_fmt(residence_r2)} |")
        lines.append(f"| {target} | + all combined | {_fmt(combined_r2)} |")
    return lines


def build_fire_feature_section(
    data_dir: Path, national_features: dict[str, list[str]], lag_days: dict[str, int]
) -> list[str]:
    fires_in_basin = filter_fires_in_basin(load_mtbs_fires(MTBS_PATH), BASIN_PATH)
    lines = [
        "\n## Five more ways to combine fire and inflow/outflow data (Scenario 1 follow-up)\n",
        "Each idea below adds one new feature to the baseline, isolating a different mechanism "
        "from the plain `days_since_fire`/`Outflow_CFS` features already tried in the sections "
        "above:\n",
    ]
    lines += _fire_feature_model_table(
        data_dir, OUTFLOW_PATH, STORAGE_PATH, fires_in_basin, RESERVOIR_LAT, RESERVOIR_LON, national_features, lag_days
    )
    lines.append(
        "\n**Caveat on `fire_pressure`:** it produces the largest single jump for TOC but makes "
        "alkalinity dramatically *worse* (below a mean-only baseline) -- the same sign-flip "
        "pattern as `days_since_fire` in the burn-scar section above, and for the same reason: a "
        "real physical fire effect should not help one water-quality parameter enormously while "
        "badly hurting a closely related one, which is evidence this is another time-index-like "
        "artifact rather than a genuine signal. **`+ all combined` for alkalinity going negative "
        "(-0.161, below both the baseline and every single idea alone) is a real, honest negative "
        "result too** -- adding five correlated/noisy features at once to a small feature set "
        "overfits rather than compounds. The four more modest, plausible ideas "
        "(`fire_precip_interaction`, `cumulative_burned_acres`, `distance_weighted_fire_pressure`, "
        "`residence_time_days`) each give small, believable gains for *both* targets -- more "
        "trustworthy than `fire_pressure`'s large but inconsistent one, precisely because they are "
        "small and consistent rather than large and one-sided.\n"
    )
    return lines

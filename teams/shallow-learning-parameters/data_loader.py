"""Load and join the Design Storm daily datasets into one modeling frame.

Mirrors the pipeline Jake Slawson used (see ../../guide.md, sections 6-7):
read each daily CSV, engineer rolling/derived features on its own timeline,
shift the predictor tables forward by a lag so a row dated "today" only ever
holds information that was actually available N days ago, then left-join
everything onto the target table (FoothillsInfluent.csv).

Every function here is a pure transform (path/DataFrame in, DataFrame out) so
it can be unit tested against small synthetic fixtures instead of the full
multi-year CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["TOC_mg_L", "Alk_mg_L"]

GAGE_COLUMNS = [
    "Turbidity_Median",
    "Specific_Cond_Mean",
    "pH_Median",
    "Temp_C_Mean",
    "Dissolved_Oxygen_Mean",
]
WEATHER_COLUMNS = ["PRCP", "SNOW", "TMAX", "TMIN"]


def _read_dated_csv(path: Path, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()


def load_target(data_dir: Path) -> pd.DataFrame:
    """Foothills influent lab results: TOC_mg_L and Alk_mg_L. This is what a
    Scenario 1 model predicts."""
    return _read_dated_csv(Path(data_dir) / "FoothillsInfluent.csv", "DATE")


def load_usgs_gage(data_dir: Path) -> pd.DataFrame:
    """USGS river sensor above Strontia Springs: turbidity, conductance, pH,
    temperature, dissolved oxygen."""
    return _read_dated_csv(Path(data_dir) / "USGS_South_Platte.csv", "Date")


def load_dwr_telemetry(data_dir: Path) -> pd.DataFrame:
    """Colorado DWR gage: flow and gage height. Its Precip column is dropped
    downstream (see guide.md section 4: it's a dirty running total)."""
    return _read_dated_csv(Path(data_dir) / "SouthPlatteTelemetry.csv", "Date")


def load_snowpack(data_dir: Path) -> pd.DataFrame:
    """SNOTEL snow water equivalent, Hoosier Pass station (Jake's post-Sep-4
    replacement for the faulty Michigan Creek feed)."""
    return _read_dated_csv(Path(data_dir) / "HoosierPass.csv", "DATE")


def load_weather(data_dir: Path) -> pd.DataFrame:
    """NOAA GHCN daily precipitation, snow, and temperature."""
    return _read_dated_csv(Path(data_dir) / "USC00058022.csv", "DATE")


def engineer_predictor_features(
    gage: pd.DataFrame,
    telemetry: pd.DataFrame,
    snow: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Build the derived features from guide.md section 7 on each source's
    own (unshifted) timeline: rolling means, the turb_flow loading term,
    flow delta, and a circular month-of-year encoding."""
    features = pd.DataFrame(index=telemetry.index)
    features["Flow_CFS"] = telemetry["Flow_CFS"]
    features["GageHeight_ft"] = telemetry["GageHeight_ft"]
    features["roll_flow_7"] = telemetry["Flow_CFS"].rolling(7).mean()
    features["flow_delta"] = telemetry["Flow_CFS"].diff()

    features = features.join(gage[GAGE_COLUMNS])
    features["roll_turb_3"] = gage["Turbidity_Median"].rolling(3).mean()
    features["turb_flow"] = gage["Turbidity_Median"] * telemetry["Flow_CFS"]

    features = features.join(snow[["SWE"]])
    features["roll_swe_7"] = snow["SWE"].rolling(7).mean()

    features = features.join(weather[WEATHER_COLUMNS])
    features["roll_precip_7"] = weather["PRCP"].rolling(7).sum()

    month = features.index.month
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    return features


def build_dataset(data_dir: Path, lag_days: int = 2) -> pd.DataFrame:
    """Return the target table left-joined with predictor features shifted
    `lag_days` forward, so every predictor column on a given date reflects
    what was known `lag_days` earlier. Rows with no predictor coverage yet
    (or lab result) are left as NaN for the caller to drop as needed."""
    target = load_target(data_dir)
    gage = load_usgs_gage(data_dir)
    telemetry = load_dwr_telemetry(data_dir)
    snow = load_snowpack(data_dir)
    weather = load_weather(data_dir)

    features = engineer_predictor_features(gage, telemetry, snow, weather)
    shifted = features.shift(lag_days, freq="D")

    return target.join(shifted, how="left")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All engineered/predictor columns in a built dataset, i.e. every
    column that isn't a target."""
    return [c for c in df.columns if c not in TARGET_COLUMNS]


def with_calendar_year_and_doy(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'year' and 'day_of_year' columns from the DatetimeIndex, for
    year-over-year comparison plots (Scenario 3: drought vs. wet years)."""
    out = df.copy()
    out["year"] = out.index.year
    out["day_of_year"] = out.index.dayofyear
    return out

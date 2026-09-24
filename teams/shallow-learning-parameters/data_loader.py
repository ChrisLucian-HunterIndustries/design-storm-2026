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


def load_michigan_creek(data_dir: Path) -> pd.DataFrame:
    """SNOTEL snow water equivalent, Michigan Creek station -- the feed
    Hoosier Pass replaced. Kept for its longer record, but has a known bad
    patch (SWE 9.0 on 2026-05-12 to 05-15, bracketed by near-zero readings;
    see AGENTS.md) that makes it a real, labeled test case for anomaly
    detection rather than a recommended predictor."""
    return _read_dated_csv(Path(data_dir) / "MichiganCreek.csv", "DATE")


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


def build_dataset_per_source_lag(
    data_dir: Path, lag_days: int = 2, precip_extra_days: int = 2
) -> pd.DataFrame:
    """Like `build_dataset`, but lags each source separately as Jake's own
    notebooks do (scripts/TOC_SoftSensor.ipynb, scripts/Alkalinity_Soft_Sensor.ipynb):
    gage/telemetry/snow shift by `lag_days`, NOAA weather shifts by
    `lag_days + precip_extra_days` -- extra days to cover NOAA's own ~2-day
    reporting delay, not a longer physical transit time (see the notebooks'
    own comment on this). Shifting each raw source before engineering
    features (rather than shifting the combined frame once) reproduces the
    notebooks' join order exactly."""
    target = load_target(data_dir)
    gage = load_usgs_gage(data_dir).shift(lag_days, freq="D")
    telemetry = load_dwr_telemetry(data_dir).shift(lag_days, freq="D")
    snow = load_snowpack(data_dir).shift(lag_days, freq="D")
    weather = load_weather(data_dir).shift(lag_days + precip_extra_days, freq="D")

    features = engineer_predictor_features(gage, telemetry, snow, weather)
    return target.join(features, how="left")


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


def peak_loading_date(data_dir: Path, start, end) -> pd.Timestamp:
    """The date with the highest turbidity x flow ("loading") within
    [start, end] -- a simple, real way to pick "the storm" in a given
    window. Shared by the whole-record transit-time trace and any
    sonde-window analysis, so both pick "the storm" the same way."""
    gage = load_usgs_gage(data_dir).loc[start:end]
    telemetry = load_dwr_telemetry(data_dir).loc[start:end]
    loading = (gage["Turbidity_Median"] * telemetry["Flow_CFS"]).dropna()
    return loading.idxmax()


def build_flow_forecast_dataset(data_dir: Path, lead_days: int = 3) -> pd.DataFrame:
    """Predict streamflow `lead_days` ahead from today's snowpack, weather,
    and current flow -- parameters.md's first "what this folder adds" idea,
    documented since this catalog's first version but never built until now.

    Unlike `build_dataset` (which shifts predictors *forward* so a row
    holds what was known before a later lab result), this shifts the FLOW
    target *backward*: a row dated "today" holds today's conditions as
    features and the actual flow `lead_days` later as the target, which is
    what a forecast needs."""
    telemetry = load_dwr_telemetry(data_dir)
    snow = load_snowpack(data_dir)
    weather = load_weather(data_dir)

    features = pd.DataFrame(index=telemetry.index)
    features["Flow_CFS"] = telemetry["Flow_CFS"]
    features["roll_flow_7"] = telemetry["Flow_CFS"].rolling(7).mean()
    features = features.join(snow[["SWE"]])
    features["roll_swe_7"] = snow["SWE"].rolling(7).mean()
    features = features.join(weather[["TMAX", "TMIN", "PRCP"]])

    target = telemetry["Flow_CFS"].shift(-lead_days, freq="D").rename("Flow_CFS_future")
    return features.join(target, how="left")

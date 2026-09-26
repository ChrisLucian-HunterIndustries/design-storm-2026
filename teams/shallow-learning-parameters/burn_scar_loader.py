"""Load MTBS (Monitoring Trends in Burn Severity) burned-area boundaries and
turn the ones inside a given watershed into lag-safe daily features. See
fetch_mtbs.py for how mtbs_fires.json was obtained.

General water-treatment knowledge, not something stated in this repo's own
materials: a burned watershed is a well-documented driver of elevated
post-storm turbidity and TOC (less soil infiltration, ash/sediment
mobilization) -- worth testing here as a genuinely different KIND of
predictor (a one-time event with a before/after transition) from anything
else in this catalog, which is all continuous daily/weekly/monthly readings.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

EARTH_RADIUS_KM = 6371.0  # general geography knowledge, not from this repo's materials


def load_mtbs_fires(path: Path) -> pd.DataFrame:
    """Parse the raw ArcGIS REST query JSON (fetch_mtbs.py) into one row
    per fire with name/year/ignition date/acres/lat/long."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for feature in data["features"]:
        attrs = feature["attributes"]
        rows.append(
            {
                "fire_name": attrs["fire_name"],
                "year": attrs["year"],
                "ignition_date": pd.to_datetime(str(attrs["ig_date"]), format="%Y%m%d"),
                "acres": attrs["acres"],
                "latitude": attrs["latitude"],
                "longitude": attrs["longitude"],
            }
        )
    return pd.DataFrame(rows)


def _point_in_polygon(lon: float, lat: float, polygon: list[list[float]]) -> bool:
    """Standard ray-casting point-in-polygon test -- no shapely dependency
    needed for a single centroid-in-basin check."""
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def filter_fires_in_basin(fires: pd.DataFrame, basin_geojson_path: Path) -> pd.DataFrame:
    """Keep only fires whose centroid falls inside the given basin polygon
    (e.g. strontia-brief/basins/south-platte-above-strontia-06707525.json)
    -- fetch_mtbs.py's bounding-box query is generous and includes fires
    well outside the actual drainage."""
    basin = json.loads(Path(basin_geojson_path).read_text(encoding="utf-8"))
    polygon = basin["features"][0]["geometry"]["coordinates"][0]
    if fires.empty:
        return fires
    mask = fires.apply(lambda row: _point_in_polygon(row["longitude"], row["latitude"], polygon), axis=1)
    return fires[mask]


def days_since_fire_feature(fires_in_basin: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Signed days since the most recent fire (in the basin) that had
    already ignited as of each date -- negative before any fire, zero on
    ignition day, growing positive after. Before the first fire in the
    record, uses days until that first fire instead (still continuous,
    still never reflects a fire that hasn't happened yet as of that date).
    Continuous and monotonic rather than a same-day binary flag, so a tree
    model can split on "how recently burned" at any threshold."""
    dates = pd.DatetimeIndex(dates)
    if fires_in_basin.empty:
        return pd.Series(float("nan"), index=dates, name="days_since_fire")

    ignitions = fires_in_basin["ignition_date"].sort_values().reset_index(drop=True)
    values = []
    for date in dates:
        prior = ignitions[ignitions <= date]
        reference = prior.iloc[-1] if len(prior) else ignitions.iloc[0]
        values.append((date - reference).days)
    return pd.Series(values, index=dates, name="days_since_fire")


def fire_pressure_feature(fires_in_basin: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """Size-weighted, time-decayed "fire pressure": sum over every fire
    already ignited as of each date of `acres / (1 + days_since_ignition)`.
    Unlike `days_since_fire`, this carries real per-fire information (size)
    rather than a single date, so a large jump here that a plain date-only
    control can't reproduce is stronger (though still not conclusive)
    evidence of a genuine fire effect rather than a time-index artifact."""
    dates = pd.DatetimeIndex(dates)
    if fires_in_basin.empty:
        return pd.Series(0.0, index=dates, name="fire_pressure")

    values = []
    for date in dates:
        ignited = fires_in_basin[fires_in_basin["ignition_date"] <= date]
        if ignited.empty:
            values.append(0.0)
            continue
        days_since = (date - ignited["ignition_date"]).dt.days
        values.append((ignited["acres"] / (1 + days_since)).sum())
    return pd.Series(values, index=dates, name="fire_pressure")


def cumulative_burned_acres_feature(
    fires_in_basin: pd.DataFrame, dates: pd.DatetimeIndex, window_years: int = 10
) -> pd.Series:
    """Rolling trailing-window sum of burned acres: total acreage from
    every fire ignited within the last `window_years` of each date. Unlike
    a running total since the record's start, this can decrease (an old
    fire ages out of the window with no new fire replacing it), so it is
    not purely monotonic in time the way `days_since_fire` is."""
    dates = pd.DatetimeIndex(dates)
    if fires_in_basin.empty:
        return pd.Series(0.0, index=dates, name="cumulative_burned_acres")

    values = []
    for date in dates:
        window_start = date - pd.DateOffset(years=window_years)
        in_window = fires_in_basin[
            (fires_in_basin["ignition_date"] > window_start) & (fires_in_basin["ignition_date"] <= date)
        ]
        values.append(in_window["acres"].sum())
    return pd.Series(values, index=dates, name="cumulative_burned_acres")


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance -- general geography knowledge, not from this
    repo's own materials."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def distance_weighted_fire_pressure_feature(
    fires_in_basin: pd.DataFrame, dates: pd.DatetimeIndex, reference_lat: float, reference_lon: float
) -> pd.Series:
    """Sum over every fire already ignited as of each date of
    `acres / (1 + distance_km)` from `reference_lat`/`reference_lon` (e.g.
    the reservoir) -- isolates distance/size as its own dimension, deliberately
    without a recency decay term (already covered separately by
    `fire_pressure_feature`/`days_since_fire_feature`), so a fire twice as
    far contributes noticeably less regardless of how long ago it ignited."""
    dates = pd.DatetimeIndex(dates)
    if fires_in_basin.empty:
        return pd.Series(0.0, index=dates, name="distance_weighted_fire_pressure")

    distances_km = fires_in_basin.apply(
        lambda row: _haversine_km(reference_lat, reference_lon, row["latitude"], row["longitude"]), axis=1
    )
    values = []
    for date in dates:
        ignited_mask = fires_in_basin["ignition_date"] <= date
        if not ignited_mask.any():
            values.append(0.0)
            continue
        weighted = fires_in_basin.loc[ignited_mask, "acres"] / (1 + distances_km[ignited_mask])
        values.append(weighted.sum())
    return pd.Series(values, index=dates, name="distance_weighted_fire_pressure")

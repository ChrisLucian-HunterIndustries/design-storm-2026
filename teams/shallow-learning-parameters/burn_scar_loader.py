"""Load MTBS (Monitoring Trends in Burn Severity) burned-area boundaries and
turn the ones inside a given watershed into a lag-safe daily feature. See
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
from pathlib import Path

import pandas as pd


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

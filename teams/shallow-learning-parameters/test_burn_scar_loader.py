"""Characterization tests for burn_scar_loader.py, against a tiny synthetic
MTBS query-result JSON (same shape as the real ArcGIS REST response, see
fetch_mtbs.py) and a tiny synthetic basin polygon, rather than the real
network-fetched snapshot or strontia-brief's real (much larger) polygon.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from burn_scar_loader import (
    cumulative_burned_acres_feature,
    days_since_fire_feature,
    distance_weighted_fire_pressure_feature,
    filter_fires_in_basin,
    fire_pressure_feature,
    load_mtbs_fires,
)

# A simple 1x1 degree square basin, centered at (-105.2, 38.9).
SQUARE_BASIN = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-105.7, 38.4], [-105.7, 39.4], [-104.7, 39.4], [-104.7, 38.4], [-105.7, 38.4]]],
            },
        }
    ],
}


def _mtbs_json(fires: list[dict]) -> dict:
    return {"features": [{"attributes": fire} for fire in fires]}


@pytest.fixture
def mtbs_path(tmp_path: Path) -> Path:
    path = tmp_path / "mtbs_fires.json"
    path.write_text(
        json.dumps(
            _mtbs_json(
                [
                    {  # inside the square basin
                        "fire_name": "403", "year": 2023, "ig_date": 20230330, "acres": 1769.0,
                        "latitude": 38.8921122, "longitude": -105.3516004,
                    },
                    {  # also inside, bigger and earlier
                        "fire_name": "BIG OLD FIRE", "year": 2000, "ig_date": 20000601, "acres": 50000.0,
                        "latitude": 39.1, "longitude": -105.3,
                    },
                    {  # well outside the square basin
                        "fire_name": "FAR AWAY", "year": 2022, "ig_date": 20220701, "acres": 500.0,
                        "latitude": 45.0, "longitude": -110.0,
                    },
                ]
            )
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def basin_path(tmp_path: Path) -> Path:
    path = tmp_path / "basin.json"
    path.write_text(json.dumps(SQUARE_BASIN), encoding="utf-8")
    return path


@pytest.fixture
def fires_in_basin(mtbs_path: Path, basin_path: Path) -> pd.DataFrame:
    return filter_fires_in_basin(load_mtbs_fires(mtbs_path), basin_path)


def test_load_mtbs_fires_parses_ignition_date(mtbs_path: Path) -> None:
    fires = load_mtbs_fires(mtbs_path)
    assert len(fires) == 3
    assert fires.loc[fires["fire_name"] == "403", "ignition_date"].iloc[0] == pd.Timestamp("2023-03-30")


def test_filter_fires_in_basin_excludes_fires_outside_polygon(mtbs_path: Path, basin_path: Path) -> None:
    fires = load_mtbs_fires(mtbs_path)
    in_basin = filter_fires_in_basin(fires, basin_path)
    assert set(in_basin["fire_name"]) == {"403", "BIG OLD FIRE"}


def test_days_since_fire_feature_is_negative_before_and_positive_after(fires_in_basin: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex(["2023-03-30", "2023-04-29"])
    feature = days_since_fire_feature(fires_in_basin, dates)
    assert feature.loc["2023-03-30"] == 0
    assert feature.loc["2023-04-29"] == 30


def test_days_since_fire_feature_is_nan_with_no_fires_in_basin() -> None:
    empty = pd.DataFrame(columns=["ignition_date"])
    dates = pd.DatetimeIndex(["2023-01-01"])
    feature = days_since_fire_feature(empty, dates)
    assert pd.isna(feature.loc["2023-01-01"])


def test_fire_pressure_feature_is_zero_before_any_fire(fires_in_basin: pd.DataFrame) -> None:
    dates = pd.date_range("1999-01-01", periods=1, freq="D")
    result = fire_pressure_feature(fires_in_basin, dates)
    assert result.iloc[0] == 0.0


def test_fire_pressure_feature_grows_with_acres_and_decays_with_time(fires_in_basin: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2023-03-30"), pd.Timestamp("2023-04-30")])
    result = fire_pressure_feature(fires_in_basin, dates)
    # On ignition day, pressure = acres / (1+0); a month later it's decayed.
    assert result.iloc[0] == pytest.approx(1769.0 + 50000.0 / (1 + (pd.Timestamp("2023-03-30") - pd.Timestamp("2000-06-01")).days))
    assert result.iloc[1] < result.iloc[0]


def test_cumulative_burned_acres_feature_sums_within_window(fires_in_basin: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2023-04-01")])
    result = cumulative_burned_acres_feature(fires_in_basin, dates, window_years=10)
    # only "403" (2023) is within a 10-year trailing window of 2023-04-01; the 2000 fire has aged out.
    assert result.iloc[0] == pytest.approx(1769.0)


def test_cumulative_burned_acres_feature_includes_older_fire_with_wide_window(fires_in_basin: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2023-04-01")])
    result = cumulative_burned_acres_feature(fires_in_basin, dates, window_years=30)
    assert result.iloc[0] == pytest.approx(1769.0 + 50000.0)


def test_distance_weighted_fire_pressure_feature_favors_closer_fires(fires_in_basin: pd.DataFrame) -> None:
    dates = pd.DatetimeIndex([pd.Timestamp("2023-04-01")])
    # reference point right at "403"'s own centroid -- it should dominate despite being much smaller.
    near_403 = distance_weighted_fire_pressure_feature(fires_in_basin, dates, 38.8921122, -105.3516004)
    # reference point right at "BIG OLD FIRE"'s own centroid instead.
    near_big = distance_weighted_fire_pressure_feature(fires_in_basin, dates, 39.1, -105.3)
    assert near_big.iloc[0] > near_403.iloc[0]

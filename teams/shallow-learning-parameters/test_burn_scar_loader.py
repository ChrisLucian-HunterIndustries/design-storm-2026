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

from burn_scar_loader import days_since_fire_feature, filter_fires_in_basin, load_mtbs_fires

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


def test_load_mtbs_fires_parses_ignition_date(mtbs_path: Path) -> None:
    fires = load_mtbs_fires(mtbs_path)
    assert len(fires) == 2
    assert fires.loc[fires["fire_name"] == "403", "ignition_date"].iloc[0] == pd.Timestamp("2023-03-30")


def test_filter_fires_in_basin_excludes_fires_outside_polygon(mtbs_path: Path, basin_path: Path) -> None:
    fires = load_mtbs_fires(mtbs_path)
    in_basin = filter_fires_in_basin(fires, basin_path)
    assert list(in_basin["fire_name"]) == ["403"]


def test_days_since_fire_feature_is_negative_before_and_positive_after(mtbs_path: Path, basin_path: Path) -> None:
    fires = load_mtbs_fires(mtbs_path)
    in_basin = filter_fires_in_basin(fires, basin_path)
    dates = pd.DatetimeIndex(["2023-01-01", "2023-03-30", "2023-04-29"])
    feature = days_since_fire_feature(in_basin, dates)
    assert feature.loc["2023-01-01"] < 0
    assert feature.loc["2023-03-30"] == 0
    assert feature.loc["2023-04-29"] == 30


def test_days_since_fire_feature_is_nan_with_no_fires_in_basin() -> None:
    empty = pd.DataFrame(columns=["ignition_date"])
    dates = pd.DatetimeIndex(["2023-01-01"])
    feature = days_since_fire_feature(empty, dates)
    assert pd.isna(feature.loc["2023-01-01"])

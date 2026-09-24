"""Tests for sonde_loader.py using small synthetic frames (already in the
post-rename column shape) plus one tiny real .xlsx fixture for the load
function itself.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sonde_loader import (
    assign_cast_ids,
    cast_summary,
    daily_surface_features,
    full_depth_casts,
    load_sonde_readings,
)


def _reading(timestamp: str, depth: float, temp: float, **overrides: float) -> dict:
    row = {
        "timestamp": pd.Timestamp(timestamp),
        "Depth_m": depth,
        "Temp_C": temp,
        "Conductivity": 200.0,
        "pH": 8.0,
        "ORP_mV": 100.0,
        "Turbidity_NTU": 1.0,
        "Chl_ugL": 0.5,
        "Phycocyanin": 0.5,
        "ODO_sat": 90.0,
        "ODO_mgL": 9.0,
    }
    row.update(overrides)
    return row


@pytest.fixture
def two_cast_frame() -> pd.DataFrame:
    """Two casts on the same day, six hours apart, each a simple 3-point
    depth profile with a clear surface-warmer-than-bottom gradient."""
    rows = [
        _reading("2026-06-01 00:00:00", 1.0, 18.0, Turbidity_NTU=2.0),
        _reading("2026-06-01 00:00:05", 5.0, 14.0, Turbidity_NTU=2.0),
        _reading("2026-06-01 00:00:10", 10.0, 10.0, Turbidity_NTU=2.0),
        _reading("2026-06-01 06:00:00", 1.0, 19.0, Turbidity_NTU=3.0),
        _reading("2026-06-01 06:00:05", 5.0, 15.0, Turbidity_NTU=3.0),
        _reading("2026-06-01 06:00:10", 10.0, 11.0, Turbidity_NTU=3.0),
    ]
    return pd.DataFrame(rows)


def test_assign_cast_ids_splits_on_large_gaps(two_cast_frame: pd.DataFrame) -> None:
    cast_ids = assign_cast_ids(two_cast_frame, gap_minutes=60)
    assert list(cast_ids) == [0, 0, 0, 1, 1, 1]


def test_assign_cast_ids_keeps_close_readings_together() -> None:
    df = pd.DataFrame(
        [
            _reading("2026-06-01 00:00:00", 1.0, 18.0),
            _reading("2026-06-01 00:00:30", 2.0, 17.0),
            _reading("2026-06-01 00:01:00", 3.0, 16.0),
        ]
    )
    cast_ids = assign_cast_ids(df, gap_minutes=60)
    assert list(cast_ids) == [0, 0, 0]


def test_cast_summary_computes_stratification(two_cast_frame: pd.DataFrame) -> None:
    summary = cast_summary(two_cast_frame)
    assert len(summary) == 2
    first = summary.iloc[0]
    assert first["surface_temp_c"] == pytest.approx(18.0)
    assert first["bottom_temp_c"] == pytest.approx(10.0)
    assert first["temp_diff_c"] == pytest.approx(8.0)
    assert first["n_readings"] == 3
    assert first["depth_min_m"] == pytest.approx(1.0)
    assert first["depth_max_m"] == pytest.approx(10.0)


def test_daily_surface_features_averages_same_day_casts(two_cast_frame: pd.DataFrame) -> None:
    daily = daily_surface_features(two_cast_frame)
    assert len(daily) == 1
    # both casts' 1m readings (Turbidity 2.0 and 3.0) average to 2.5
    assert daily.iloc[0]["Turbidity_NTU"] == pytest.approx(2.5)
    assert daily.index[0] == pd.Timestamp("2026-06-01")


def test_daily_surface_features_excludes_deep_readings(two_cast_frame: pd.DataFrame) -> None:
    daily = daily_surface_features(two_cast_frame)
    # only the 1m rows (surface) should feed the average, not the 5m/10m rows
    assert daily.iloc[0]["Temp_C"] == pytest.approx((18.0 + 19.0) / 2)


def test_full_depth_casts_excludes_short_casts(two_cast_frame: pd.DataFrame) -> None:
    # both casts in the fixture have 3 readings; a threshold of 5 should drop both
    summary = cast_summary(two_cast_frame)
    assert len(full_depth_casts(summary, min_readings=5)) == 0
    assert len(full_depth_casts(summary, min_readings=3)) == 2


def test_load_sonde_readings_renames_and_sorts(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        {
            "Time stamp": [pd.Timestamp("2026-06-01 06:00:00"), pd.Timestamp("2026-06-01 00:00:00")],
            "Temp C": [15.0, 18.0],
            "Conductivity ": [200.0, 201.0],
            "Vertical Position ": [5.0, 1.0],
            "pH": [8.0, 8.1],
            "ORP mV": [100.0, 101.0],
            "Turbidity NTU": [2.0, 2.5],
            "Chl ug/L": [0.4, 0.5],
            "Phycocyanin ": [0.3, 0.35],
            "ODO & sat": [90.0, 91.0],
            "ODO mg/L": [9.0, 9.1],
        }
    )
    path = tmp_path / "Strontia 0407_0819.xlsx"
    raw.to_excel(path, index=False)

    loaded = load_sonde_readings(tmp_path)
    assert list(loaded.columns)[:3] == ["timestamp", "Temp_C", "Conductivity"]
    assert loaded["timestamp"].is_monotonic_increasing
    assert loaded.iloc[0]["Depth_m"] == pytest.approx(1.0)

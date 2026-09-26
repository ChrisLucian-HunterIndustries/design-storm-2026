"""Characterization tests for denver_storage_loader.py, against a tiny
synthetic storage-history.json (same water-year-indexed shape as
water-system-3d/fetch_storage_history.py's real output) rather than the
real file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from denver_storage_loader import load_reservoir_storage


@pytest.fixture
def storage_path(tmp_path: Path) -> Path:
    path = tmp_path / "storage-history.json"
    # Water year 2023 = Oct 1 2022 .. Sep 30 2023. Day-of-water-year index 0 = Oct 1.
    years = {"2023": [None] * 366}
    years["2023"][0] = 5000  # 2022-10-01
    years["2023"][1] = 5100  # 2022-10-02
    path.write_text(
        json.dumps({"reservoirs": {"STRRESCO": {"years": years, "median": [], "worst": {}}}}),
        encoding="utf-8",
    )
    return path


def test_load_reservoir_storage_maps_water_year_index_to_calendar_date(storage_path: Path) -> None:
    series = load_reservoir_storage(storage_path)
    assert series.loc[pd.Timestamp("2022-10-01")] == 5000
    assert series.loc[pd.Timestamp("2022-10-02")] == 5100


def test_load_reservoir_storage_drops_null_days(storage_path: Path) -> None:
    series = load_reservoir_storage(storage_path)
    assert pd.Timestamp("2022-10-03") not in series.index

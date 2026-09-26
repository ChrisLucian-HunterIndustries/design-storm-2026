"""Load Strontia Springs Reservoir's daily storage from the already-fetched
water-system-3d/storage-history.json (see fetch_storage_history.py) --
reused as-is rather than fetched again, since that file already covers this
catalog's full 2022-2026 modeling window.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

DAYS_IN_WATER_YEAR = 366


def load_reservoir_storage(path: Path, abbrev: str = "STRRESCO") -> pd.Series:
    """Parse the water-year-indexed day array (index 0 = Oct 1 of the
    prior calendar year) back into a calendar-date-indexed Series, dropping
    null (not-yet-recorded) days."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    years = data["reservoirs"][abbrev]["years"]
    values: dict[pd.Timestamp, float] = {}
    for water_year_str, days in years.items():
        water_year_start = date(int(water_year_str) - 1, 10, 1)
        for index, value in enumerate(days):
            if value is not None:
                values[pd.Timestamp(water_year_start + timedelta(days=index))] = value
    return pd.Series(values, name="Storage_AF").sort_index()

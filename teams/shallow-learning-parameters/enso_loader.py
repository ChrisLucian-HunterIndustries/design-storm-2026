"""Load NOAA PSL's Oceanic Nino Index (ONI) and turn it into a lag-safe
daily feature -- see fetch_oni.py for how oni.txt is obtained.

ONI is published monthly as a 3-month running-mean sea-surface-temperature
anomaly (El Nino/La Nina strength). This is a genuinely different signal
from anything else in this catalog: every existing predictor is a local,
daily watershed reading, while ONI is a basin-scale, monthly climate index
-- a plausible candidate for explaining the year-to-year variation this
catalog has already flagged (e.g. the sonde-vs-gage section: "most of that
full-record correlation comes from year-to-year variation... not
day-to-day variation").
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MISSING_SENTINEL = -99.0


def parse_oni_text(text: str) -> pd.DataFrame:
    """Parse NOAA/PSL's fixed-width ONI format: a header line, then one row
    per year (year followed by 12 monthly values), then a footer. Sentinel
    values (-99.90, for months not yet observed) are dropped. Returns a
    DataFrame indexed by month-start Timestamp with one 'ONI' column."""
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        for month, value in enumerate(values, start=1):
            if value <= MISSING_SENTINEL:
                continue
            rows.append({"DATE": pd.Timestamp(year=year, month=month, day=1), "ONI": value})
    return pd.DataFrame(rows).set_index("DATE").sort_index()


def load_oni(path: Path) -> pd.DataFrame:
    """Read and parse oni.txt from disk."""
    return parse_oni_text(Path(path).read_text(encoding="utf-8"))


def oni_feature_for_dates(oni: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """The previous calendar month's ONI value for each date -- e.g. every
    day in March gets February's ONI -- so a row is never trained on an
    ONI reading published after that row's own date, the same lookahead
    discipline data_loader.build_dataset applies to the daily sources."""
    prev_month_start = (dates.to_period("M") - 1).to_timestamp()
    aligned = oni["ONI"].reindex(prev_month_start)
    return pd.Series(aligned.to_numpy(), index=dates, name="ONI_prev_month")

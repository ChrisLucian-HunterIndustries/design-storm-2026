"""Load the U.S. Drought Monitor's weekly county drought-severity CSV and
turn it into a lag-safe daily feature -- see fetch_usdm.py for how
usdm_jefferson.csv is obtained.

USDM publishes five *cumulative* area-percent columns (D0..D4, each "at
least this severe"); the standard single-number summary is the Drought
Severity Coefficient Index (DSCI = D0+D1+D2+D3+D4, range 0-500). Unlike
NOAA's ONI (monthly, basin-scale), this is weekly and county-specific --
a genuinely higher-frequency, more local public signal, closer in spirit
to the daily watershed readings already in this catalog than ONI is.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DROUGHT_COLUMNS = ["D0", "D1", "D2", "D3", "D4"]


def parse_usdm_csv(text: str) -> pd.DataFrame:
    """Parse the source's CSV (header + one row per week). Returns a
    DataFrame indexed by ValidStart (the week's first day) with the D0-D4
    columns plus a computed DSCI column, sorted ascending."""
    from io import StringIO

    df = pd.read_csv(StringIO(text))
    df["ValidStart"] = pd.to_datetime(df["ValidStart"])
    df["ValidEnd"] = pd.to_datetime(df["ValidEnd"])
    df = df.set_index("ValidStart").sort_index()
    df["DSCI"] = df[DROUGHT_COLUMNS].sum(axis=1)
    return df[[*DROUGHT_COLUMNS, "DSCI", "ValidEnd"]]


def load_usdm(path: Path) -> pd.DataFrame:
    """Read and parse usdm_jefferson.csv from disk."""
    return parse_usdm_csv(Path(path).read_text(encoding="utf-8"))


def dsci_feature_for_dates(usdm: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.Series:
    """The most recently *fully ended* week's DSCI for each date -- a date
    only ever sees a week whose ValidEnd is strictly before it, so a row
    is never trained on a drought reading published after that row's own
    date (same lookahead discipline as enso_loader.oni_feature_for_dates).
    Implemented as an as-of backward join on ValidEnd (one row per week,
    weeks don't overlap, so "most recent ValidEnd < date" is unambiguous)."""
    by_end = usdm.sort_values("ValidEnd")
    lookup = pd.DataFrame({"date": pd.DatetimeIndex(dates).sort_values()})
    # subtract 1 day so a week is only visible strictly after its ValidEnd, not on it
    joined = pd.merge_asof(lookup, by_end[["ValidEnd", "DSCI"]], left_on="date", right_on="ValidEnd", allow_exact_matches=False)
    aligned = joined.set_index("date")["DSCI"].reindex(dates)
    return aligned.rename("DSCI_prev_week")

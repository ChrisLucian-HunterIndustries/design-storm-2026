"""Characterization tests for drought_loader.py, against a tiny synthetic
USDM CSV sample (same format as usdm_jefferson.csv, see fetch_usdm.py)
rather than the real, network-fetched file.
"""
from __future__ import annotations

import pandas as pd
import pytest

from drought_loader import dsci_feature_for_dates, parse_usdm_csv

SAMPLE_USDM_CSV = """\
MapDate,FIPS,County,State,None,D0,D1,D2,D3,D4,ValidStart,ValidEnd,StatisticFormatID
20260101,08059,Jefferson County,CO,0.00,100.00,80.00,40.00,0.00,0.00,2026-01-01,2026-01-07,1
20260108,08059,Jefferson County,CO,0.00,100.00,90.00,50.00,10.00,0.00,2026-01-08,2026-01-14,1
20260115,08059,Jefferson County,CO,0.00,100.00,100.00,60.00,20.00,0.00,2026-01-15,2026-01-21,1
"""


def test_parse_usdm_csv_computes_dsci() -> None:
    usdm = parse_usdm_csv(SAMPLE_USDM_CSV)
    assert len(usdm) == 3
    first = usdm.loc[pd.Timestamp("2026-01-01")]
    assert first["DSCI"] == pytest.approx(100 + 80 + 40 + 0 + 0)


def test_parse_usdm_csv_sorted_ascending() -> None:
    usdm = parse_usdm_csv(SAMPLE_USDM_CSV)
    assert list(usdm.index) == sorted(usdm.index)


def test_dsci_feature_for_dates_uses_most_recently_ended_week() -> None:
    """A date inside the second week (2026-01-10) should see the FIRST
    week's DSCI (ValidEnd 2026-01-07), not its own in-progress week's."""
    usdm = parse_usdm_csv(SAMPLE_USDM_CSV)
    dates = pd.DatetimeIndex(["2026-01-10"])
    feature = dsci_feature_for_dates(usdm, dates)
    assert feature.loc["2026-01-10"] == pytest.approx(100 + 80 + 40)


def test_dsci_feature_for_dates_excludes_the_ending_day_itself() -> None:
    """A date exactly on a week's ValidEnd should not see that same week
    (not yet published as of that day) -- only the one before it."""
    usdm = parse_usdm_csv(SAMPLE_USDM_CSV)
    dates = pd.DatetimeIndex(["2026-01-14"])
    feature = dsci_feature_for_dates(usdm, dates)
    assert feature.loc["2026-01-14"] == pytest.approx(100 + 80 + 40)


def test_dsci_feature_for_dates_is_nan_before_any_data() -> None:
    usdm = parse_usdm_csv(SAMPLE_USDM_CSV)
    dates = pd.DatetimeIndex(["2026-01-02"])
    feature = dsci_feature_for_dates(usdm, dates)
    assert pd.isna(feature.loc["2026-01-02"])

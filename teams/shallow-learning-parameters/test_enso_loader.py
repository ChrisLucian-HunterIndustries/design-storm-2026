"""Characterization tests for enso_loader.py, against a tiny synthetic ONI
text sample (same format NOAA/PSL publishes, see fetch_oni.py) rather than
the real, network-fetched oni.txt.
"""
from __future__ import annotations

import pandas as pd
import pytest

from enso_loader import oni_feature_for_dates, parse_oni_text

SAMPLE_ONI_TEXT = """\
 1950         2026
 2024   1.81   1.66   1.27   0.93   0.74   0.64   0.57   0.43   0.39   0.44   0.50   0.61
 2025  -0.45  -0.24  -0.06   0.02  -0.02  -0.04  -0.11  -0.26  -0.43  -0.57  -0.61  -0.60
 2026  -0.39  -0.21   0.11   0.46   0.95   1.39   1.80 -99.90 -99.90 -99.90 -99.90 -99.90
  -99.9
 ONI from CPC
  Provided by NOAA/PSL
"""


def test_parse_oni_text_reads_one_row_per_month() -> None:
    oni = parse_oni_text(SAMPLE_ONI_TEXT)
    assert oni.loc["2025-01-01", "ONI"] == pytest.approx(-0.45)
    assert oni.loc["2026-07-01", "ONI"] == pytest.approx(1.80)


def test_parse_oni_text_drops_missing_sentinel_months() -> None:
    oni = parse_oni_text(SAMPLE_ONI_TEXT)
    assert "2026-08-01" not in oni.index
    assert "2026-12-01" not in oni.index


def test_parse_oni_text_ignores_header_and_footer_lines() -> None:
    oni = parse_oni_text(SAMPLE_ONI_TEXT)
    # 2 full years (2024, 2025) + 7 observed months of 2026
    assert len(oni) == 24 + 7


def test_oni_feature_for_dates_uses_previous_calendar_month() -> None:
    """A day in March 2026 should carry February 2026's ONI, not March's own
    (not-yet-fully-observed-by-then) value -- no lookahead."""
    oni = parse_oni_text(SAMPLE_ONI_TEXT)
    dates = pd.DatetimeIndex(["2026-03-05", "2026-03-20"])
    feature = oni_feature_for_dates(oni, dates)
    assert feature.loc["2026-03-05"] == pytest.approx(-0.21)
    assert feature.loc["2026-03-20"] == pytest.approx(-0.21)


def test_oni_feature_for_dates_handles_year_boundary() -> None:
    """A day in January should carry the PRIOR year's December value."""
    oni = parse_oni_text(SAMPLE_ONI_TEXT)
    dates = pd.DatetimeIndex(["2026-01-15"])
    feature = oni_feature_for_dates(oni, dates)
    assert feature.loc["2026-01-15"] == pytest.approx(-0.60)


def test_oni_feature_for_dates_is_nan_when_month_unavailable() -> None:
    """A day in August 2026 wants July 2026's ONI -- present here -- but a
    day in September wants August's, which is the dropped sentinel month."""
    oni = parse_oni_text(SAMPLE_ONI_TEXT)
    dates = pd.DatetimeIndex(["2026-09-10"])
    feature = oni_feature_for_dates(oni, dates)
    assert pd.isna(feature.loc["2026-09-10"])

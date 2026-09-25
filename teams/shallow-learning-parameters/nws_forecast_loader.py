"""Load the NWS (National Weather Service) gridpoint forecast for
Strontia Springs Reservoir. See fetch_nws_forecast.py for how
nws_forecast_snapshot.json was obtained.

Important limitation, stated up front rather than glossed over: the NWS
API only ever serves the CURRENT forecast -- it has no historical archive
of what a forecast said on a past date. Every other public dataset in this
catalog (ONI, DSCI, MTBS) can be fetched once and then tested against this
record's full 2022-2026 history; this one cannot. It can only ever be
demonstrated on "today" (see analyze_nws_forecast.py), the same
live-conditions framing dosing_alerts.score_current_conditions already
uses for the same reason (this repo's data ends on a fixed date with no
"sensor exists, lab result doesn't yet" row to genuinely backtest against).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_forecast_periods(path: Path) -> pd.DataFrame:
    """Parse the raw gridpoint forecast JSON into one row per forecast
    period (roughly one per day/night, ~14 periods total)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for period in data["properties"]["periods"]:
        precip = period.get("probabilityOfPrecipitation") or {}
        rows.append(
            {
                "name": period["name"],
                "start_time": pd.Timestamp(period["startTime"]),
                "temperature": period["temperature"],
                "precip_probability_pct": precip.get("value"),
                "short_forecast": period["shortForecast"],
            }
        )
    return pd.DataFrame(rows)

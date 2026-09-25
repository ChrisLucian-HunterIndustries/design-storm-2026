"""Characterization tests for nws_forecast_loader.py, against a tiny
synthetic NWS forecast JSON (same shape as the real gridpoint forecast
response, see fetch_nws_forecast.py) rather than the real, network-fetched
snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nws_forecast_loader import load_forecast_periods

SAMPLE_FORECAST = {
    "properties": {
        "periods": [
            {
                "name": "This Afternoon",
                "startTime": "2026-09-25T13:00:00-06:00",
                "temperature": 74,
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 64},
                "shortForecast": "Chance Showers And Thunderstorms",
            },
            {
                "name": "Tonight",
                "startTime": "2026-09-25T18:00:00-06:00",
                "temperature": 45,
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 47},
                "shortForecast": "Chance Showers And Thunderstorms then Partly Cloudy",
            },
            {
                "name": "Saturday",
                "startTime": "2026-09-26T06:00:00-06:00",
                "temperature": 79,
                "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 0},
                "shortForecast": "Sunny",
            },
        ]
    }
}


def test_load_forecast_periods_parses_name_temp_and_precip(tmp_path: Path) -> None:
    path = tmp_path / "forecast.json"
    path.write_text(json.dumps(SAMPLE_FORECAST), encoding="utf-8")

    periods = load_forecast_periods(path)

    assert len(periods) == 3
    assert list(periods.columns) == ["name", "start_time", "temperature", "precip_probability_pct", "short_forecast"]
    assert periods.loc[0, "name"] == "This Afternoon"
    assert periods.loc[0, "precip_probability_pct"] == 64
    assert periods.loc[0, "start_time"] == pd.Timestamp("2026-09-25T13:00:00-06:00")


def test_load_forecast_periods_handles_missing_precip_probability(tmp_path: Path) -> None:
    """Some periods have a null probabilityOfPrecipitation value in the
    real feed -- should become NaN, not crash."""
    data = {
        "properties": {
            "periods": [
                {
                    "name": "Later",
                    "startTime": "2026-09-27T06:00:00-06:00",
                    "temperature": 70,
                    "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": None},
                    "shortForecast": "Sunny",
                }
            ]
        }
    }
    path = tmp_path / "forecast.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    periods = load_forecast_periods(path)
    assert pd.isna(periods.loc[0, "precip_probability_pct"])

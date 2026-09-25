"""Characterization tests for analyze_nws_forecast.py, against a tiny
synthetic forecast (see test_nws_forecast_loader.py).
"""
from __future__ import annotations

from analyze_nws_forecast import summarize_forecast
from nws_forecast_loader import load_forecast_periods


def test_summarize_forecast_finds_peak_precip_period(tmp_path) -> None:
    import json

    path = tmp_path / "forecast.json"
    path.write_text(
        json.dumps(
            {
                "properties": {
                    "periods": [
                        {
                            "name": "This Afternoon",
                            "startTime": "2026-09-25T13:00:00-06:00",
                            "temperature": 74,
                            "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 64},
                            "shortForecast": "Chance Showers",
                        },
                        {
                            "name": "Tonight",
                            "startTime": "2026-09-25T18:00:00-06:00",
                            "temperature": 45,
                            "probabilityOfPrecipitation": {"unitCode": "wmoUnit:percent", "value": 20},
                            "shortForecast": "Partly Cloudy",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    periods = load_forecast_periods(path)

    summary = summarize_forecast(periods)

    assert summary["max_precip_probability_pct"] == 64
    assert summary["period_name"] == "This Afternoon"
    assert summary["n_periods"] == 2


def test_summarize_forecast_handles_all_missing_precip_probability(tmp_path) -> None:
    import json

    path = tmp_path / "forecast.json"
    path.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    periods = load_forecast_periods(path)

    summary = summarize_forecast(periods)

    import pandas as pd

    assert pd.isna(summary["max_precip_probability_pct"])
    assert summary["period_name"] is None

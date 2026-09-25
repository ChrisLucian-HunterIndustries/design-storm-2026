"""Refetch the NWS (National Weather Service) gridpoint forecast for
Strontia Springs Reservoir -- a public dataset not otherwise used in this
catalog. Network required; rerun whenever a fresh "current conditions"
demo is wanted (see nws_forecast_loader.py / analyze_nws_forecast.py).
Saves the raw JSON to nws_forecast_snapshot.json in this directory.

Unlike every other public-data fetch script in this catalog, this one
CANNOT be re-run to reproduce a past result -- the NWS API only ever
serves the current forecast, with no historical archive. The snapshot
committed here is a point-in-time demonstration, not a dataset this
catalog's held-out R^2 numbers can be computed against.

Source: https://api.weather.gov (NWS API), /points/{lat},{lon} to resolve
the grid cell, then /gridpoints/{office}/{x},{y}/forecast.
"""
from __future__ import annotations

from pathlib import Path

import requests

LATITUDE, LONGITUDE = 39.42453, -105.13845  # Strontia Springs Reservoir centroid
USER_AGENT = "design-storm-2026 (github.com/denverwater/design-storm-2026)"
OUT_PATH = Path(__file__).resolve().parent / "nws_forecast_snapshot.json"


def main() -> None:
    points_response = requests.get(
        f"https://api.weather.gov/points/{LATITUDE},{LONGITUDE}",
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    points_response.raise_for_status()
    forecast_url = points_response.json()["properties"]["forecast"]

    forecast_response = requests.get(forecast_url, headers={"User-Agent": USER_AGENT}, timeout=30)
    forecast_response.raise_for_status()
    OUT_PATH.write_text(forecast_response.text, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(forecast_response.text)} bytes)")


if __name__ == "__main__":
    main()

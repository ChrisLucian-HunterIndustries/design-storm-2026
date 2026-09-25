"""Refetch the U.S. Drought Monitor's weekly county drought-severity
history for Jefferson County, CO (FIPS 08059, the county the South Platte
gage above Strontia Springs sits in per its own WQP/STORET record) -- a
public dataset not otherwise used in this catalog, tested here as a
possible new predictor (see drought_loader.py / analyze_drought.py).
Network required; rerun rarely, since USDM only updates once a week.
Saves the raw CSV to usdm_jefferson.csv in this directory, in the exact
format the source publishes it in (same "keep originals as-received"
convention as data/*.csv).

Source: https://usdmdataservices.unl.edu (National Drought Mitigation
Center), CountyStatistics/GetDroughtSeverityStatisticsByAreaPercent
endpoint, area-percent-in-category format, weekly since 2000-01-04.
"""
from __future__ import annotations

from pathlib import Path

import requests

USDM_URL = (
    "https://usdmdataservices.unl.edu/api/CountyStatistics/"
    "GetDroughtSeverityStatisticsByAreaPercent"
)
COUNTY_FIPS = "08059"
OUT_PATH = Path(__file__).resolve().parent / "usdm_jefferson.csv"


def main() -> None:
    response = requests.get(
        USDM_URL,
        params={"aoi": COUNTY_FIPS, "startdate": "1/1/2000", "enddate": "12/31/2026", "statisticsType": "1"},
        timeout=30,
    )
    response.raise_for_status()
    OUT_PATH.write_text(response.text, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(response.text)} bytes)")


if __name__ == "__main__":
    main()

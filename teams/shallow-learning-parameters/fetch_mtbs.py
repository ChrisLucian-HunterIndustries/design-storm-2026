"""Refetch MTBS (Monitoring Trends in Burn Severity) burned-area boundaries
for a generous bounding box around the South Platte watershed above
Strontia Springs -- a public USFS dataset not otherwise used in this
catalog, tested here as a possible new predictor (see burn_scar_loader.py /
analyze_burn_scar.py). Network required; rerun rarely (MTBS updates once a
year, after each fire season's final assessment). Saves the raw JSON to
mtbs_fires.json in this directory, in the exact format the source publishes
it in (same "keep originals as-received" convention as data/*.csv).

Source: https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer
(USFS Enterprise Data Warehouse), layer 63 ("Burned Area Boundaries (All
Years)"). The bounding box is intentionally generous -- it includes fires
outside the actual drainage basin, which burn_scar_loader.filter_fires_in_basin
filters out using the real basin polygon.
"""
from __future__ import annotations

from pathlib import Path

import requests

MTBS_QUERY_URL = "https://apps.fs.usda.gov/arcx/rest/services/EDW/EDW_MTBS_01/MapServer/63/query"
BASIN_BBOX = "-106.3,38.9,-105.0,39.6"  # generous box around the upper South Platte watershed
OUT_PATH = Path(__file__).resolve().parent / "mtbs_fires.json"


def main() -> None:
    response = requests.get(
        MTBS_QUERY_URL,
        params={
            "geometry": BASIN_BBOX,
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "where": "1=1",
            "f": "json",
        },
        timeout=30,
    )
    response.raise_for_status()
    OUT_PATH.write_text(response.text, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(response.text)} bytes)")


if __name__ == "__main__":
    main()

"""Refetch NOAA PSL's Oceanic Nino Index (ONI) snapshot -- a public NOAA
dataset not otherwise used in this catalog, tested here as a possible new
predictor (see enso_loader.py / analyze_enso.py). Network required; rerun
rarely, since ONI only updates once a month. Saves the raw text to oni.txt
in this directory, in the exact format NOAA/PSL publishes it in (same
"keep originals as-received" convention as data/*.csv).

Source: https://psl.noaa.gov/data/correlation/oni.data (NOAA Physical
Sciences Laboratory, itself mirroring CPC's oni.ascii.txt -- see the
footer of the fetched file for the exact provenance NOAA states).
"""
from __future__ import annotations

from pathlib import Path

import requests

ONI_URL = "https://psl.noaa.gov/data/correlation/oni.data"
OUT_PATH = Path(__file__).resolve().parent / "oni.txt"


def main() -> None:
    response = requests.get(ONI_URL, timeout=30)
    response.raise_for_status()
    OUT_PATH.write_text(response.text, encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({len(response.text)} bytes)")


if __name__ == "__main__":
    main()

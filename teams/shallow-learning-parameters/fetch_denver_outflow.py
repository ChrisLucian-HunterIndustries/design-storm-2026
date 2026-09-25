"""Refetch daily discharge history for Denver Water's two Strontia Springs
outlet conduits (COND20CO, COND26CO) -- a public Colorado DWR dataset not
otherwise used in this catalog, tested here as a possible new predictor
(see denver_outflow_loader.py / analyze_denver_outflow.py). Network
required; rerun rarely. Saves the raw JSON to denver_conduit_outflow.json
in this directory, in the exact format the source publishes it in (same
"keep originals as-received" convention as data/*.csv).

Source: https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/
telemetrytimeseriesday/ (Colorado Division of Water Resources CDSS REST
API) -- the same API this repo already uses for reservoir storage, see
water-system-3d/fetch_storage_history.py.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

API = "https://dwr.state.co.us/Rest/GET/api/v2/telemetrystations/telemetrytimeseriesday/"
CONDUIT_ABBREVS = ["COND20CO", "COND26CO"]
START_DATE = "2022-01-01"  # margin before FoothillsInfluent.csv's 2022-04-01 start
OUT_PATH = Path(__file__).resolve().parent / "denver_conduit_outflow.json"


def fetch_conduit(abbrev: str) -> dict:
    query = urllib.parse.urlencode(
        {
            "format": "json",
            "abbrev": abbrev,
            "parameter": "DISCHRG",
            "startDate": START_DATE,
            "endDate": date.today().isoformat(),
            "pageSize": 50000,
        }
    )
    with urllib.request.urlopen(f"{API}?{query}") as response:
        return json.load(response)


def main() -> None:
    snapshot = {abbrev: fetch_conduit(abbrev) for abbrev in CONDUIT_ABBREVS}
    OUT_PATH.write_text(json.dumps(snapshot), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({sum(len(v.get('ResultList', [])) for v in snapshot.values())} total rows)")


if __name__ == "__main__":
    main()

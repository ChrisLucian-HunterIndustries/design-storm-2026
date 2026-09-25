"""Load daily discharge history for Denver Water's two Strontia Springs
outlet conduits and sum them into a single outflow feature. See
fetch_denver_outflow.py for how denver_conduit_outflow.json was obtained.

Both stations are DWR (Colorado Division of Water Resources) telemetry
gages, hosted at https://dwr.state.co.us -- the same public REST API this
repo already uses for reservoir storage (see
water-system-3d/fetch_storage_history.py), just a different parameter
(DISCHRG instead of STORAGE) and different stations:
COND26CO ("DW CONDUIT 26", at the dam itself) is the reservoir's main
outlet to the Foothills Treatment Plant; COND20CO ("DENVER WATER CONDUIT
NO 20", a few miles downstream) is a smaller secondary conduit. Together
they are Denver Water's own measured outflow from the reservoir this
catalog's target (data/FoothillsInfluent.csv, the plant's own intake) is
drawn from -- as distinct from Flow_CFS (data/SouthPlatteTelemetry.csv),
the river gage already used elsewhere in this catalog as the reservoir's
*inflow*.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CONDUIT_ABBREVS = ["COND20CO", "COND26CO"]


def load_conduit_discharge(path: Path) -> pd.DataFrame:
    """Parse the raw CDSS telemetrytimeseriesday JSON snapshot into one
    column per conduit plus their summed Outflow_CFS (missing conduit
    readings on a date contribute 0, not NaN, unless every conduit is
    missing that date)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = {}
    for abbrev in CONDUIT_ABBREVS:
        rows = data.get(abbrev, {}).get("ResultList", [])
        columns[abbrev] = pd.Series(
            {pd.Timestamp(row["measDate"]).tz_localize(None).normalize(): row["measValue"] for row in rows}
        )
    df = pd.DataFrame(columns).sort_index()
    df["Outflow_CFS"] = df[CONDUIT_ABBREVS].sum(axis=1, min_count=1)
    return df

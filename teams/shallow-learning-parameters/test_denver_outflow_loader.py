"""Characterization tests for denver_outflow_loader.py, against a tiny
synthetic CDSS telemetrytimeseriesday JSON snapshot (same shape as
fetch_denver_outflow.py's raw response -- see fetch_storage_history.py in
water-system-3d/ for the same public API used elsewhere in this repo)
rather than the real network-fetched snapshot.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from denver_outflow_loader import load_conduit_discharge


def _cdss_series(rows: list[tuple[str, float]]) -> dict:
    return {"ResultList": [{"measDate": date, "measValue": value} for date, value in rows]}


@pytest.fixture
def snapshot_path(tmp_path: Path) -> Path:
    path = tmp_path / "denver_conduit_outflow.json"
    path.write_text(
        json.dumps(
            {
                "COND20CO": _cdss_series(
                    [("2022-04-01T00:00:00-06:00", 10.0), ("2022-04-02T00:00:00-06:00", 12.0)]
                ),
                "COND26CO": _cdss_series(
                    [("2022-04-01T00:00:00-06:00", 200.0), ("2022-04-02T00:00:00-06:00", 210.0)]
                ),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_conduit_discharge_parses_each_conduit_column(snapshot_path):
    df = load_conduit_discharge(snapshot_path)
    assert list(df.index) == [pd.Timestamp("2022-04-01"), pd.Timestamp("2022-04-02")]
    assert df.loc[pd.Timestamp("2022-04-01"), "COND20CO"] == 10.0
    assert df.loc[pd.Timestamp("2022-04-01"), "COND26CO"] == 200.0


def test_load_conduit_discharge_sums_into_outflow_cfs(snapshot_path):
    df = load_conduit_discharge(snapshot_path)
    assert df.loc[pd.Timestamp("2022-04-01"), "Outflow_CFS"] == 210.0
    assert df.loc[pd.Timestamp("2022-04-02"), "Outflow_CFS"] == 222.0


def test_load_conduit_discharge_treats_one_missing_conduit_as_zero_contribution(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps({"COND20CO": _cdss_series([("2022-04-01T00:00:00-06:00", 10.0)]), "COND26CO": _cdss_series([])}),
        encoding="utf-8",
    )
    df = load_conduit_discharge(path)
    assert df.loc[pd.Timestamp("2022-04-01"), "Outflow_CFS"] == 10.0

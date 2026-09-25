"""Characterization tests for analyze_denver_outflow.py, against small
synthetic national-dataset CSVs (see test_analyze_burn_scar.py) plus a
synthetic Denver Water conduit-outflow snapshot and MTBS fire/basin fixture
(see test_denver_outflow_loader.py / test_burn_scar_loader.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analyze_denver_outflow import _denver_outflow_model_table
from burn_scar_loader import filter_fires_in_basin, load_mtbs_fires

TOC_FEATURES = ["Flow_CFS", "Turbidity_Median"]
ALK_FEATURES = ["Specific_Cond_Mean"]

SQUARE_BASIN = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-105.7, 38.4], [-105.7, 39.4], [-104.7, 39.4], [-104.7, 38.4], [-105.7, 38.4]]],
            },
        }
    ],
}


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Thirty consecutive days of every national source file (matches
    test_analyze_burn_scar.py's fixture shape)."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    date_strs = [d.strftime("%m/%d/%Y") for d in dates]

    _write_csv(
        tmp_path / "FoothillsInfluent.csv",
        "DATE,TOC_mg_L,Alk_mg_L\n"
        + "\n".join(f"{s},{2.0 + i * 0.05},{70 + i * 0.3}" for i, s in enumerate(date_strs)),
    )
    _write_csv(
        tmp_path / "USGS_South_Platte.csv",
        "Date,site_no,Dissolved_Oxygen_Mean,Specific_Cond_Mean,Temp_C_Mean,Turbidity_Median,pH_Median\n"
        + "\n".join(
            f"{s},6707525,{9.0 + i * 0.01},{300 + i},{5.0 + i * 0.1},{1.0 + i * 0.2},{8.0 + i * 0.01}"
            for i, s in enumerate(date_strs)
        ),
    )
    _write_csv(
        tmp_path / "SouthPlatteTelemetry.csv",
        "Date,Flow_CFS,GageHeight_ft,Precip\n"
        + "\n".join(f"{s},{100 + i * 5},{2.0 + i * 0.05},10" for i, s in enumerate(date_strs)),
    )
    _write_csv(
        tmp_path / "HoosierPass.csv",
        "DATE,SWE\n" + "\n".join(f"{d.date()},{12.0 + i * 0.1}" for i, d in enumerate(dates)),
    )
    _write_csv(
        tmp_path / "USC00058022.csv",
        "STATION,DATE,PRCP,SNOW,TMAX,TMIN\n"
        + "\n".join(f"USC00058022,{s},{0.1 * i},0,{50 + i},{20 + i}" for i, s in enumerate(date_strs)),
    )
    return tmp_path


@pytest.fixture
def outflow_path(tmp_path: Path) -> Path:
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    path = tmp_path / "denver_conduit_outflow.json"
    path.write_text(
        json.dumps(
            {
                "COND20CO": {
                    "ResultList": [
                        {"measDate": d.strftime("%Y-%m-%dT00:00:00-06:00"), "measValue": 10.0 + i}
                        for i, d in enumerate(dates)
                    ]
                },
                "COND26CO": {
                    "ResultList": [
                        {"measDate": d.strftime("%Y-%m-%dT00:00:00-06:00"), "measValue": 200.0 + i}
                        for i, d in enumerate(dates)
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def fires_in_basin(tmp_path: Path) -> pd.DataFrame:
    mtbs_path = tmp_path / "mtbs.json"
    mtbs_path.write_text(
        json.dumps(
            {
                "features": [
                    {
                        "attributes": {
                            "fire_name": "TEST FIRE", "year": 2024, "ig_date": 20240110, "acres": 100.0,
                            "latitude": 38.9, "longitude": -105.2,
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    basin_path = tmp_path / "basin.json"
    basin_path.write_text(json.dumps(SQUARE_BASIN), encoding="utf-8")
    return filter_fires_in_basin(load_mtbs_fires(mtbs_path), basin_path)


def test_denver_outflow_model_table_reports_baseline_outflow_fire_and_combined(
    dataset_dir: Path, outflow_path: Path, fires_in_basin: pd.DataFrame
) -> None:
    lines = _denver_outflow_model_table(
        dataset_dir,
        outflow_path,
        fires_in_basin,
        {"TOC_mg_L": TOC_FEATURES, "Alk_mg_L": ALK_FEATURES},
        {"TOC_mg_L": 2, "Alk_mg_L": 4},
    )
    text = "\n".join(lines)
    assert "baseline" in text
    assert "+ Outflow_CFS" in text
    assert "+ days_since_fire" in text
    assert "+ Outflow_CFS + days_since_fire" in text
    assert "TOC_mg_L" in text
    assert "Alk_mg_L" in text

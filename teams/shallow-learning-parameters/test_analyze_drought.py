"""Characterization tests for analyze_drought.py's report-table builders,
against small synthetic national-dataset CSVs (same headers as
data_loader.py's real files, see test_data_loader.py) plus tiny synthetic
ONI/USDM samples (see test_enso_loader.py / test_drought_loader.py),
rather than the real multi-year data or network-fetched snapshots.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analyze_drought import _combined_public_data_table, _drought_model_table
from drought_loader import parse_usdm_csv
from enso_loader import parse_oni_text

TOC_FEATURES = ["Flow_CFS", "Turbidity_Median"]
ALK_FEATURES = ["Specific_Cond_Mean"]

SAMPLE_ONI_TEXT = """\
 1950         2026
 2023  -0.30  -0.28  -0.25  -0.20  -0.10   0.05   0.20   0.35   0.50   0.60   0.65   0.68
 2024   0.70   0.68 -99.90 -99.90 -99.90 -99.90 -99.90 -99.90 -99.90 -99.90 -99.90 -99.90
  -99.9
 ONI from CPC
  Provided by NOAA/PSL
"""

SAMPLE_USDM_CSV = """\
MapDate,FIPS,County,State,None,D0,D1,D2,D3,D4,ValidStart,ValidEnd,StatisticFormatID
20231224,08059,Jefferson County,CO,0.00,100.00,80.00,40.00,0.00,0.00,2023-12-24,2023-12-30,1
20231231,08059,Jefferson County,CO,0.00,100.00,90.00,50.00,10.00,0.00,2023-12-31,2024-01-06,1
20240107,08059,Jefferson County,CO,0.00,100.00,100.00,60.00,20.00,0.00,2024-01-07,2024-01-13,1
"""


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Ten consecutive January-2024 days of every national source file --
    same fixture shape as test_data_loader.py's own dataset_dir."""
    _write_csv(
        tmp_path / "FoothillsInfluent.csv",
        """
DATE,TOC_mg_L,Alk_mg_L
1/1/2024,2.0,70
1/2/2024,2.1,71
1/3/2024,2.2,72
1/4/2024,2.3,73
1/5/2024,2.4,74
1/6/2024,2.5,75
1/7/2024,2.6,76
1/8/2024,2.7,77
1/9/2024,2.8,78
1/10/2024,2.9,79
""",
    )
    _write_csv(
        tmp_path / "USGS_South_Platte.csv",
        """
Date,site_no,Dissolved_Oxygen_Mean,Specific_Cond_Mean,Temp_C_Mean,Turbidity_Median,pH_Median
1/1/2024,6707525,9.0,300,5.0,1.0,8.0
1/2/2024,6707525,9.1,301,5.1,2.0,8.1
1/3/2024,6707525,9.2,302,5.2,3.0,8.2
1/4/2024,6707525,9.3,303,5.3,4.0,8.3
1/5/2024,6707525,9.4,304,5.4,5.0,8.4
1/6/2024,6707525,9.5,305,5.5,6.0,8.5
1/7/2024,6707525,9.6,306,5.6,7.0,8.6
1/8/2024,6707525,9.7,307,5.7,8.0,8.7
1/9/2024,6707525,9.8,308,5.8,9.0,8.8
1/10/2024,6707525,9.9,309,5.9,10.0,8.9
""",
    )
    _write_csv(
        tmp_path / "SouthPlatteTelemetry.csv",
        """
Date,Flow_CFS,GageHeight_ft,Precip
1/1/2024,100,2.0,10
1/2/2024,110,2.1,10
1/3/2024,120,2.2,10
1/4/2024,130,2.3,10
1/5/2024,140,2.4,10
1/6/2024,150,2.5,10
1/7/2024,160,2.6,10
1/8/2024,170,2.7,10
1/9/2024,180,2.8,10
1/10/2024,190,2.9,10
""",
    )
    _write_csv(
        tmp_path / "HoosierPass.csv",
        """
DATE,SWE
2024-01-01,12.0
2024-01-02,12.1
2024-01-03,12.2
2024-01-04,12.3
2024-01-05,12.4
2024-01-06,12.5
2024-01-07,12.6
2024-01-08,12.7
2024-01-09,12.8
2024-01-10,12.9
""",
    )
    _write_csv(
        tmp_path / "USC00058022.csv",
        """
STATION,DATE,PRCP,SNOW,TMAX,TMIN
USC00058022,1/1/2024,0,0,40,20
USC00058022,1/2/2024,0,0,41,21
USC00058022,1/3/2024,0,0,42,22
USC00058022,1/4/2024,0,0,43,23
USC00058022,1/5/2024,0,0,44,24
USC00058022,1/6/2024,0,0,45,25
USC00058022,1/7/2024,0,0,46,26
USC00058022,1/8/2024,0,0,47,27
USC00058022,1/9/2024,0,0,48,28
USC00058022,1/10/2024,1,0,49,29
""",
    )
    return tmp_path


@pytest.fixture
def usdm() -> pd.DataFrame:
    return parse_usdm_csv(SAMPLE_USDM_CSV)


@pytest.fixture
def oni() -> pd.DataFrame:
    return parse_oni_text(SAMPLE_ONI_TEXT)


def test_drought_model_table_covers_both_uniform_and_hybrid_lag_frames(
    dataset_dir: Path, usdm: pd.DataFrame
) -> None:
    """Mirrors analyze_enso.py's _oni_model_table, which checks both the
    uniform-lag baseline AND the hybrid-lag (tuned) frame -- DSCI should
    get the same "does it still help once the lag itself is tuned" test,
    not just the uniform-lag one."""
    lines = _drought_model_table(dataset_dir, usdm, TOC_FEATURES, ALK_FEATURES)
    text = "\n".join(lines)
    assert "uniform lag" in text
    assert "hybrid lag (tuned)" in text


def test_combined_public_data_table_reports_both_targets_and_frames(
    dataset_dir: Path, oni: pd.DataFrame, usdm: pd.DataFrame
) -> None:
    """The two public-data ideas found in this catalog (ONI, DSCI) tried
    together, on both frames, for both targets."""
    lines = _combined_public_data_table(dataset_dir, oni, usdm, TOC_FEATURES, ALK_FEATURES)
    text = "\n".join(lines)
    assert "national + ONI + DSCI" in text
    assert "uniform lag" in text
    assert "hybrid lag (tuned)" in text
    assert "TOC_mg_L" in text
    assert "Alk_mg_L" in text

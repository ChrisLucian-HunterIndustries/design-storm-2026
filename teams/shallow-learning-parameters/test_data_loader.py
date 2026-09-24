"""Characterization tests for data_loader.py.

Uses small synthetic CSVs (written to tmp_path) with the same headers as the
real files, so the join/shift/rolling logic can be verified precisely
without depending on the multi-year datasets in data/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_loader import (
    build_dataset,
    engineer_predictor_features,
    feature_columns,
    load_target,
)


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Ten consecutive days of every source file, small enough to hand-check."""
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


def test_load_target_indexes_by_date(dataset_dir: Path) -> None:
    target = load_target(dataset_dir)
    assert list(target.columns) == ["TOC_mg_L", "Alk_mg_L"]
    assert target.index[0] == pd.Timestamp("2024-01-01")
    assert target.loc["2024-01-03", "TOC_mg_L"] == 2.2


def test_engineered_features_flow_delta_and_turb_flow(dataset_dir: Path) -> None:
    from data_loader import load_dwr_telemetry, load_snowpack, load_usgs_gage, load_weather

    gage = load_usgs_gage(dataset_dir)
    telemetry = load_dwr_telemetry(dataset_dir)
    snow = load_snowpack(dataset_dir)
    weather = load_weather(dataset_dir)

    features = engineer_predictor_features(gage, telemetry, snow, weather)

    # day 2 flow (110) - day 1 flow (100)
    assert features.loc["2024-01-02", "flow_delta"] == pytest.approx(10.0)
    # turbidity(2.0) * flow(110) on day 2
    assert features.loc["2024-01-02", "turb_flow"] == pytest.approx(220.0)
    # 7-day rolling mean only available once 7 rows have accumulated
    assert np.isnan(features.loc["2024-01-06", "roll_flow_7"])
    assert features.loc["2024-01-07", "roll_flow_7"] == pytest.approx(
        sum(range(100, 170, 10)) / 7
    )


def test_month_sin_cos_encoding(dataset_dir: Path) -> None:
    from data_loader import load_dwr_telemetry, load_snowpack, load_usgs_gage, load_weather

    features = engineer_predictor_features(
        load_usgs_gage(dataset_dir),
        load_dwr_telemetry(dataset_dir),
        load_snowpack(dataset_dir),
        load_weather(dataset_dir),
    )
    row = features.loc["2024-01-01"]
    assert row["month_sin"] == pytest.approx(np.sin(2 * np.pi * 1 / 12))
    assert row["month_cos"] == pytest.approx(np.cos(2 * np.pi * 1 / 12))


def test_build_dataset_shifts_predictors_forward(dataset_dir: Path) -> None:
    """The row dated Jan 4 should carry Jan 2's raw flow reading when
    lag_days=2 (see guide.md section 6: shift(2, freq='D'))."""
    joined = build_dataset(dataset_dir, lag_days=2)

    assert joined.loc["2024-01-04", "Flow_CFS"] == pytest.approx(110.0)
    assert joined.loc["2024-01-04", "TOC_mg_L"] == pytest.approx(2.3)


def test_build_dataset_has_no_predictor_data_before_lag_elapses(dataset_dir: Path) -> None:
    joined = build_dataset(dataset_dir, lag_days=2)
    assert np.isnan(joined.loc["2024-01-01", "Flow_CFS"])
    assert np.isnan(joined.loc["2024-01-02", "Flow_CFS"])


def test_build_dataset_keeps_every_target_row(dataset_dir: Path) -> None:
    """A left join from the target must not drop or duplicate lab-result days,
    even when predictor coverage is incomplete."""
    target = load_target(dataset_dir)
    joined = build_dataset(dataset_dir, lag_days=4)
    assert len(joined) == len(target)


def test_feature_columns_excludes_targets(dataset_dir: Path) -> None:
    joined = build_dataset(dataset_dir, lag_days=2)
    columns = feature_columns(joined)
    assert "TOC_mg_L" not in columns
    assert "Alk_mg_L" not in columns
    assert "turb_flow" in columns


def test_with_calendar_year_and_doy(dataset_dir: Path) -> None:
    from data_loader import with_calendar_year_and_doy

    target = load_target(dataset_dir)
    tagged = with_calendar_year_and_doy(target)

    assert tagged.loc["2024-01-01", "year"] == 2024
    assert tagged.loc["2024-01-01", "day_of_year"] == 1
    assert tagged.loc["2024-01-10", "day_of_year"] == 10
    # original columns untouched
    assert tagged.loc["2024-01-03", "TOC_mg_L"] == 2.2

"""Characterization tests for analyze_snowmelt.py's melt-out date regression
(parameters.md's long-documented-but-unbuilt "predict SWE decline/melt-out
date from temperature" idea), against small synthetic snowpack + weather
CSVs (see test_data_loader.py's fixture shape).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analyze_snowmelt import compute_melt_out_summary


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Two synthetic 'water years', each with a clean peak-then-decline SWE
    curve and a spring TMAX series -- enough to compute one melt-out row
    per year."""
    dates_2023 = pd.date_range("2023-03-01", "2023-06-30", freq="D")
    dates_2024 = pd.date_range("2024-03-01", "2024-06-30", freq="D")

    def _swe_curve(dates: pd.DatetimeIndex, peak: float, melt_start_day: int) -> list[float]:
        values = []
        for i, _ in enumerate(dates):
            if i < melt_start_day:
                values.append(peak)
            else:
                values.append(max(peak - (i - melt_start_day) * 1.0, 0.0))
        return values

    swe_2023 = _swe_curve(dates_2023, peak=20.0, melt_start_day=30)
    swe_2024 = _swe_curve(dates_2024, peak=10.0, melt_start_day=10)
    all_dates = list(dates_2023) + list(dates_2024)
    all_swe = swe_2023 + swe_2024

    snow_lines = "DATE,SWE\n" + "\n".join(f"{d.date()},{v}" for d, v in zip(all_dates, all_swe))
    _write_csv(tmp_path / "HoosierPass.csv", snow_lines)

    weather_lines = "STATION,DATE,PRCP,SNOW,TMAX,TMIN\n" + "\n".join(
        f"USC00058022,{d.month}/{d.day}/{d.year},0,0,60,30" for d in all_dates
    )
    _write_csv(tmp_path / "USC00058022.csv", weather_lines)
    return tmp_path


def test_compute_melt_out_summary_one_row_per_year(dataset_dir: Path) -> None:
    summary = compute_melt_out_summary(dataset_dir)
    assert set(summary["year"]) == {2023, 2024}


def test_compute_melt_out_summary_more_snow_melts_out_later(dataset_dir: Path) -> None:
    """2023 has both a higher peak AND a later melt start, so it should
    have a later melt-out day-of-year than 2024."""
    summary = compute_melt_out_summary(dataset_dir).set_index("year")
    assert summary.loc[2023, "melt_out_day_of_year"] > summary.loc[2024, "melt_out_day_of_year"]
    assert summary.loc[2023, "peak_swe"] == pytest.approx(20.0)
    assert summary.loc[2024, "peak_swe"] == pytest.approx(10.0)

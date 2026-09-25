"""Characterization tests for analyze_robustness.py's four follow-up
experiments on the sonde's limited 4-month window, against small synthetic
national-dataset CSVs + a synthetic sonde .xlsx (same fixture shapes as
test_analyze_drought.py / test_sonde_loader.py), not the real multi-year
data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analyze_robustness import (
    compute_cv_stability_scores,
    compute_delta_target_scores,
    compute_regularization_comparison,
    compute_two_stage_chain_scores,
)

TOC_FEATURES = ["Flow_CFS", "Turbidity_Median"]
ALK_FEATURES = ["Specific_Cond_Mean"]


def _write_csv(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _sonde_rows(dates: list[str]) -> list[dict]:
    rows = []
    for i, day in enumerate(dates):
        for depth, temp in [(1.0, 18.0 - i * 0.1), (20.0, 10.0 - i * 0.1)]:
            rows.append(
                {
                    "Time stamp": pd.Timestamp(f"{day} 00:00:00"),
                    "Temp C": temp,
                    "Conductivity ": 200.0 + i,
                    "Vertical Position ": depth,
                    "pH": 8.0,
                    "ORP mV": 100.0,
                    "Turbidity NTU": 1.0 + i * 0.2,
                    "Chl ug/L": 0.5,
                    "Phycocyanin ": 0.5,
                    "ODO & sat": 90.0,
                    "ODO mg/L": 9.0,
                }
            )
    return rows


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """Thirty consecutive days of every national source file (rolling(7)
    warmup needs more rows than the 10-row fixture other test files use),
    plus a matching one-cast-per-day synthetic sonde file covering the same
    range -- so `sonde_readings["timestamp"].min()/.max()` restricts every
    limited-window function to these same 30 days."""
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
    pd.DataFrame(_sonde_rows([str(d.date()) for d in dates])).to_excel(
        tmp_path / "Strontia 0407_0819.xlsx", index=False
    )
    return tmp_path


@pytest.fixture
def sonde_readings(dataset_dir: Path) -> pd.DataFrame:
    from sonde_loader import load_sonde_readings

    return load_sonde_readings(dataset_dir)


def test_regularization_comparison_reports_all_four_models(dataset_dir: Path, sonde_readings: pd.DataFrame) -> None:
    scores = compute_regularization_comparison(
        dataset_dir, sonde_readings, {"TOC_mg_L": TOC_FEATURES, "Alk_mg_L": ALK_FEATURES}, {"TOC_mg_L": 2, "Alk_mg_L": 4}
    )
    assert set(scores["target"]) == {"TOC_mg_L", "Alk_mg_L"}
    for col in ("rf_r2", "svr_r2", "ridge_r2", "lasso_r2"):
        assert col in scores.columns


def test_cv_stability_scores_returns_one_row_per_fold(dataset_dir: Path, sonde_readings: pd.DataFrame) -> None:
    scores = compute_cv_stability_scores(
        dataset_dir,
        sonde_readings,
        {"TOC_mg_L": TOC_FEATURES, "Alk_mg_L": ALK_FEATURES},
        {"TOC_mg_L": 2, "Alk_mg_L": 4},
        n_splits=3,
    )
    assert "fold" in scores.columns
    assert "r2" in scores.columns
    assert scores["target"].isin({"TOC_mg_L", "Alk_mg_L"}).all()


def test_delta_target_scores_reports_level_and_delta(dataset_dir: Path, sonde_readings: pd.DataFrame) -> None:
    scores = compute_delta_target_scores(
        dataset_dir, sonde_readings, {"TOC_mg_L": TOC_FEATURES, "Alk_mg_L": ALK_FEATURES}, {"TOC_mg_L": 2, "Alk_mg_L": 4}
    )
    assert set(scores["framing"]) == {"level", "day-over-day delta"}
    assert set(scores["target"]) == {"TOC_mg_L", "Alk_mg_L"}


def test_two_stage_chain_reports_three_stages(dataset_dir: Path, sonde_readings: pd.DataFrame) -> None:
    scores = compute_two_stage_chain_scores(dataset_dir, sonde_readings)
    assert len(scores) == 3
    assert "stage" in scores.columns
    assert "r2" in scores.columns

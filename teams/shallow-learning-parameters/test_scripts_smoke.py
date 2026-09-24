"""Smoke tests: analyze_parameters.main() and visualize.main() run
end-to-end against a synthetic dataset and produce their output files,
without depending on the real (large, external) data/ directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analyze_parameters
import visualize


def _write_synthetic_dataset(data_dir: Path, n_days: int = 90) -> None:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2023-04-01", periods=n_days, freq="D")

    flow = rng.uniform(100, 900, size=n_days)
    turbidity = rng.uniform(0.5, 10, size=n_days)
    swe = np.clip(20 - np.arange(n_days) * 0.1 + rng.normal(0, 0.5, n_days), 0, None)

    pd.DataFrame(
        {
            "DATE": dates.strftime("%m/%d/%Y"),
            "TOC_mg_L": 2 + 0.0002 * flow + 0.05 * turbidity + rng.normal(0, 0.2, n_days),
            "Alk_mg_L": 60 + rng.normal(0, 5, n_days),
        }
    ).to_csv(data_dir / "FoothillsInfluent.csv", index=False)

    pd.DataFrame(
        {
            "Date": dates.strftime("%m/%d/%Y"),
            "site_no": 6707525,
            "Dissolved_Oxygen_Mean": rng.uniform(8, 11, n_days),
            "Specific_Cond_Mean": rng.uniform(280, 330, n_days),
            "Temp_C_Mean": rng.uniform(2, 15, n_days),
            "Turbidity_Median": turbidity,
            "pH_Median": rng.uniform(8.0, 8.6, n_days),
        }
    ).to_csv(data_dir / "USGS_South_Platte.csv", index=False)

    pd.DataFrame(
        {
            "Date": dates.strftime("%m/%d/%Y"),
            "Flow_CFS": flow,
            "GageHeight_ft": rng.uniform(2, 5, n_days),
            "Precip": rng.uniform(-100, 20, n_days),
        }
    ).to_csv(data_dir / "SouthPlatteTelemetry.csv", index=False)

    pd.DataFrame({"DATE": dates.strftime("%Y-%m-%d"), "SWE": swe}).to_csv(
        data_dir / "HoosierPass.csv", index=False
    )

    pd.DataFrame(
        {
            "STATION": "USC00058022",
            "DATE": dates.strftime("%m/%d/%Y"),
            "PRCP": rng.uniform(0, 1, n_days),
            "SNOW": rng.uniform(0, 2, n_days),
            "TMAX": rng.uniform(40, 80, n_days),
            "TMIN": rng.uniform(20, 50, n_days),
        }
    ).to_csv(data_dir / "USC00058022.csv", index=False)


@pytest.fixture
def synthetic_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_synthetic_dataset(data_dir)
    return data_dir


def test_analyze_parameters_writes_summary(
    synthetic_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setattr(analyze_parameters, "DATA_DIR", synthetic_data_dir)
    monkeypatch.setattr(analyze_parameters, "RESULTS_DIR", results_dir)

    analyze_parameters.main()

    output = results_dir / "parameter_summary.md"
    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "Random forest, held-out R^2" in text
    assert "Unsupervised hydrologic-regime clusters" in text
    assert "Empirical transit-time lag scan" in text


def test_visualize_writes_all_figures(
    synthetic_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figures_dir = tmp_path / "figures"
    monkeypatch.setattr(visualize, "DATA_DIR", synthetic_data_dir)
    monkeypatch.setattr(visualize, "FIGURES_DIR", figures_dir)

    visualize.main()

    expected = [
        "01_targets_timeseries.png",
        "02_correlation_heatmap.png",
        "03_turb_flow_vs_toc.png",
        "04_feature_importance.png",
        "05_hydrologic_regimes.png",
        "06_alkalinity_classifier_pr_curve.png",
        "07_snowpack_streamflow_by_year.png",
        "08_lag_correlation_scan.png",
        "09_storm_event_trace.png",
        "10_transit_animation.gif",
    ]
    for name in expected:
        assert (figures_dir / name).exists(), name

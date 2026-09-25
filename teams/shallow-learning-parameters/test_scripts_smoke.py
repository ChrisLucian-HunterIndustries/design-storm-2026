"""Smoke tests: analyze_parameters.main() and visualize.main() run
end-to-end against a synthetic dataset and produce their output files,
without depending on the real (large, external) data/ directory.
"""
from __future__ import annotations

import json
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

    michigan_swe = np.clip(15 - np.arange(n_days) * 0.12 + rng.normal(0, 0.4, n_days), 0, None)
    michigan_swe[40:44] = 9.0  # a synthetic stuck-sensor patch, like the real MichiganCreek.csv one
    pd.DataFrame({"DATE": dates.strftime("%m/%d/%Y"), "SWE": michigan_swe}).to_csv(
        data_dir / "MichiganCreek.csv", index=False
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

    _write_synthetic_sonde(data_dir, dates, rng)


def _write_synthetic_sonde(data_dir: Path, dates: pd.DatetimeIndex, rng: np.random.Generator) -> None:
    """One cast per day, five depths each, surface warmer than bottom --
    enough for cast_summary/daily_surface_features and a storm before/after
    comparison anywhere in the date range."""
    depths = [round(d, 2) for d in np.linspace(1.0, 40.0, 24)]
    rows = []
    for day in dates:
        surface_temp = rng.uniform(10, 20)
        for depth in depths:
            rows.append(
                {
                    "Time stamp": day + pd.Timedelta(hours=12, minutes=depth),
                    "Temp C": surface_temp - depth * 0.15,
                    "Conductivity ": rng.uniform(200, 300),
                    "Vertical Position ": depth,
                    "pH": rng.uniform(7.8, 8.5),
                    "ORP mV": rng.uniform(80, 120),
                    "Turbidity NTU": rng.uniform(0.5, 5),
                    "Chl ug/L": rng.uniform(0.2, 2),
                    "Phycocyanin ": rng.uniform(0.2, 2),
                    "ODO & sat": rng.uniform(70, 95),
                    "ODO mg/L": rng.uniform(7, 10),
                }
            )
    pd.DataFrame(rows).to_excel(data_dir / "Strontia 0407_0819.xlsx", index=False)


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
    assert "Model family comparison" in text
    assert "Lag-day grid search" in text
    assert "Per-source lag: does Jake's approach beat a flat 2/4-day lag?" in text
    assert "Hybrid lag: tuning precip's extra lag per target beats both fixed choices" in text
    assert "Does the best model family also win on the best-lag frame?" in text
    assert "lake turnover" in text
    assert "closer predictor" in text
    assert "depth profile" in text
    assert "classifier: model family comparison" in text
    assert "Sensor-fault anomaly detection" in text
    assert "Hyperparameter tuning" in text
    assert "Flow forecasting" in text
    assert "Quantile / peak-focused regression" in text
    assert "Joint multi-output modeling" in text
    assert "Gaussian Process regression" in text
    assert "SARIMAX" in text
    assert "Chemical-dosing alert calendar" in text
    assert "Is a dose required right now?" in text
    assert "NOAA ENSO experiment: does the Oceanic Nino Index help?" in text
    assert "Other ideas for the limited 4-month window" in text
    assert "MTBS burn-scar experiment" in text

    json_path = results_dir / "predictions.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"generated_at", "toc", "alk", "context"}
    for target_key in ("toc", "alk"):
        assert set(payload[target_key].keys()) == {
            "lag_days",
            "test_r2",
            "actual",
            "predicted_train",
            "predicted_test",
        }
        assert len(payload[target_key]["actual"]) > 0
    assert set(payload["context"].keys()) == {"Flow_CFS", "Turbidity_Median", "PRCP", "SWE"}


def test_visualize_writes_all_figures(
    synthetic_data_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figures_dir = tmp_path / "figures"
    monkeypatch.setattr(visualize, "DATA_DIR", synthetic_data_dir)
    monkeypatch.setattr(visualize, "FIGURES_DIR", figures_dir)

    visualize.main()

    expected = [
        "scenario1_toc_alkalinity/01_targets_timeseries.png",
        "scenario1_toc_alkalinity/02_correlation_heatmap.png",
        "scenario1_toc_alkalinity/03_turb_flow_vs_toc.png",
        "scenario1_toc_alkalinity/04_feature_importance.png",
        "scenario2_storm_runoff/05_hydrologic_regimes.png",
        "scenario1_toc_alkalinity/06_alkalinity_classifier_pr_curve.png",
        "scenario3_snowpack_system/07_snowpack_streamflow_by_year.png",
        "scenario3_snowpack_system/08_lag_correlation_scan.png",
        "scenario3_snowpack_system/09_storm_event_trace.png",
        "scenario3_snowpack_system/10_transit_animation.gif",
        "scenario1_toc_alkalinity/11_model_family_comparison.png",
        "scenario1_toc_alkalinity/12_lag_day_grid_search.png",
        "scenario3_snowpack_system/13_stratification_timeline.png",
        "scenario2_storm_runoff/14_depth_profiles.png",
        "scenario2_storm_runoff/15_storm_profile_comparison.png",
        "scenario1_toc_alkalinity/16_sonde_vs_gage_comparison.png",
        "scenario1_toc_alkalinity/17_classifier_comparison.png",
        "scenario3_snowpack_system/18_anomaly_detection.png",
        "scenario1_toc_alkalinity/19_tuning_comparison.png",
        "scenario3_snowpack_system/20_flow_forecast.png",
        "scenario1_toc_alkalinity/21_quantile_bands.png",
        "scenario1_toc_alkalinity/22_multioutput_comparison.png",
        "scenario1_toc_alkalinity/23_gaussian_process.png",
        "scenario1_toc_alkalinity/24_sarimax.png",
        "scenario1_toc_alkalinity/25_alkalinity_roc_curve.png",
        "scenario1_toc_alkalinity/26_dosing_alert_timeline.png",
        "scenario1_toc_alkalinity/27_lag_approach_comparison.png",
        "scenario1_toc_alkalinity/28_precip_extra_lag_grid.png",
        "scenario1_toc_alkalinity/29_hybrid_lag_model_family.png",
        "scenario1_toc_alkalinity/30_enso_experiment.png",
        "scenario1_toc_alkalinity/31_limited_window_comparison.png",
        "scenario1_toc_alkalinity/32_regularization_comparison.png",
        "scenario1_toc_alkalinity/33_cv_stability.png",
        "scenario1_toc_alkalinity/34_delta_target_comparison.png",
        "scenario1_toc_alkalinity/35_two_stage_chain.png",
        "scenario1_toc_alkalinity/36_burn_scar_comparison.png",
    ]
    for name in expected:
        assert (figures_dir / name).exists(), name

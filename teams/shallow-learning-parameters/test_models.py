"""Tests for models.py using small synthetic frames with a known signal, so
assertions can check the models actually recover it (not just "runs without
crashing").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import (
    best_lag,
    cluster_hydrologic_regimes,
    detect_anomalies,
    fit_gradient_boosting_importance,
    fit_linear_baseline,
    fit_logistic_baseline,
    fit_random_forest_importance,
    fit_svr_baseline,
    fit_threshold_classifier,
    lag_correlation_scan,
    predict_full_series,
    time_ordered_split,
)


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.date_range("2022-04-01", periods=n, freq="D")
    driver = rng.uniform(0, 10, size=n)
    noise = rng.normal(0, 0.05, size=n)
    other = rng.uniform(-5, 5, size=n)  # irrelevant feature
    target = 2.0 * driver + 1.0 + noise
    return pd.DataFrame(
        {"driver": driver, "other": other, "target": target}, index=dates
    )


def test_time_ordered_split_does_not_shuffle(synthetic_frame: pd.DataFrame) -> None:
    train, test = time_ordered_split(synthetic_frame, train_fraction=0.5)
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(synthetic_frame)


def test_linear_baseline_recovers_strong_signal(synthetic_frame: pd.DataFrame) -> None:
    model, r2 = fit_linear_baseline(synthetic_frame, "driver", "target")
    assert r2 > 0.9
    assert model.coef_[0] == pytest.approx(2.0, abs=0.2)


def test_random_forest_ranks_true_driver_above_noise(synthetic_frame: pd.DataFrame) -> None:
    result = fit_random_forest_importance(synthetic_frame, ["driver", "other"], "target")
    assert result.r2 > 0.8
    assert result.importances["driver"] > result.importances["other"]


def test_gradient_boosting_ranks_true_driver_above_noise(synthetic_frame: pd.DataFrame) -> None:
    result = fit_gradient_boosting_importance(synthetic_frame, ["driver", "other"], "target")
    assert result.r2 > 0.8
    assert result.importances["driver"] > result.importances["other"]


def test_svr_baseline_recovers_strong_signal(synthetic_frame: pd.DataFrame) -> None:
    result = fit_svr_baseline(synthetic_frame, ["driver", "other"], "target")
    assert result.r2 > 0.8
    assert result.importances["driver"] > result.importances["other"]


def test_predict_full_series_labels_train_and_test(synthetic_frame: pd.DataFrame) -> None:
    result = predict_full_series(synthetic_frame, ["driver", "other"], "target")
    assert len(result.frame) == len(synthetic_frame)
    assert set(result.frame["split"].unique()) == {"train", "test"}
    assert result.test_r2 > 0.8
    # predictions should track the actual values closely given the strong signal
    assert result.frame["predicted"].corr(result.frame["actual"]) > 0.9


def test_cluster_hydrologic_regimes_labels_every_row(synthetic_frame: pd.DataFrame) -> None:
    clean, clusters = cluster_hydrologic_regimes(synthetic_frame, ["driver", "other"], n_clusters=3)
    assert len(clusters.labels) == len(clean)
    assert clusters.coordinates.shape == (len(clean), 2)
    assert clusters.cluster_sizes.sum() == len(clean)
    assert set(clusters.labels) <= {0, 1, 2}


def test_threshold_classifier_separates_clear_signal(synthetic_frame: pd.DataFrame) -> None:
    result = fit_threshold_classifier(
        synthetic_frame, ["driver", "other"], "target", threshold=11.0, below=True
    )
    assert result.roc_auc > 0.8
    assert len(result.precision) == len(result.recall)
    assert "driver" in result.importances.index
    assert len(result.fpr) == len(result.tpr)
    assert result.fpr[0] == 0.0 and result.fpr[-1] == 1.0
    assert result.tpr[0] == 0.0 and result.tpr[-1] == 1.0
    assert result.scores.between(0, 1).all()
    assert isinstance(result.scores.index, pd.DatetimeIndex)


def test_logistic_baseline_separates_clear_signal(synthetic_frame: pd.DataFrame) -> None:
    result = fit_logistic_baseline(
        synthetic_frame, ["driver", "other"], "target", threshold=11.0, below=True
    )
    assert result.roc_auc > 0.8
    assert len(result.precision) == len(result.recall)
    assert result.importances["driver"] > result.importances["other"]
    assert len(result.fpr) == len(result.tpr)
    assert result.fpr[0] == 0.0 and result.fpr[-1] == 1.0
    assert result.tpr[0] == 0.0 and result.tpr[-1] == 1.0
    assert result.scores.between(0, 1).all()
    assert isinstance(result.scores.index, pd.DatetimeIndex)


def test_detect_anomalies_flags_an_obvious_outlier() -> None:
    rng = np.random.default_rng(2)
    n = 200
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    value = rng.normal(5, 0.2, size=n)
    value[100] = 50.0  # one glaring outlier
    df = pd.DataFrame({"value": value, "diff": np.abs(np.diff(value, prepend=value[0]))}, index=dates)

    result = detect_anomalies(df, ["value", "diff"], contamination=0.02)
    assert result.is_anomaly.loc[dates[100]]
    # the score at the outlier should be lower (more anomalous) than a typical row
    assert result.scores.loc[dates[100]] < result.scores.loc[dates[0]]


@pytest.fixture
def lagged_signal() -> tuple[pd.Series, pd.Series, int]:
    """White-noise predictor (no autocorrelation) and a target that is
    exactly the predictor from `true_lag` days earlier, plus tiny noise --
    so only lag == true_lag should show a strong correlation."""
    rng = np.random.default_rng(1)
    n = 200
    true_lag = 3
    dates = pd.date_range("2022-01-01", periods=n, freq="D")
    predictor = pd.Series(rng.normal(size=n), index=dates, name="p")
    target = predictor.shift(true_lag, freq="D").reindex(dates) + rng.normal(0, 0.01, size=n)
    return predictor, target.rename("t"), true_lag


def test_lag_correlation_scan_peaks_at_true_lag(
    lagged_signal: tuple[pd.Series, pd.Series, int],
) -> None:
    predictor, target, true_lag = lagged_signal
    scan = lag_correlation_scan(predictor, target, max_lag_days=10)

    assert len(scan) == 11
    assert scan[true_lag] > 0.99
    assert best_lag(scan) == true_lag
    # a lag far from the true one should show near-zero correlation (white noise)
    assert abs(scan[0]) < 0.3


def test_lag_correlation_scan_returns_nan_below_minimum_overlap() -> None:
    dates = pd.date_range("2022-01-01", periods=3, freq="D")
    predictor = pd.Series([1.0, 2.0, 3.0], index=dates)
    target = pd.Series([1.0, 2.0, 3.0], index=dates)

    scan = lag_correlation_scan(predictor, target, max_lag_days=2)
    assert scan.isna().all()

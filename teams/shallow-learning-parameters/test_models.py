"""Tests for models.py using small synthetic frames with a known signal, so
assertions can check the models actually recover it (not just "runs without
crashing").
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models import (
    cluster_hydrologic_regimes,
    fit_linear_baseline,
    fit_random_forest_importance,
    fit_threshold_classifier,
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

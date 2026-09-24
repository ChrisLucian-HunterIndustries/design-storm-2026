"""Tests for dosing_alerts.py. `recall_tuned_threshold` is tested on raw
arrays (no model fitting needed); the alert-builder functions are tested
against real fitted classifier/quantile results on small synthetic frames
with a known, separable signal, matching this repo's existing test style.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dosing_alerts import (
    alkalinity_alert_dates,
    build_dosing_alert,
    recall_tuned_threshold,
    toc_alert_dates,
)
from models import fit_logistic_baseline
from models_advanced import fit_quantile_regressor


def test_recall_tuned_threshold_picks_highest_threshold_meeting_floor() -> None:
    recall = np.array([1.0, 1.0, 0.8, 0.6, 0.4, 0.0])
    thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    threshold = recall_tuned_threshold(recall, thresholds, min_recall=0.8)

    assert threshold == pytest.approx(0.3)


def test_recall_tuned_threshold_raises_when_unreachable() -> None:
    recall = np.array([0.5, 0.4, 0.3, 0.0])
    thresholds = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="no threshold"):
        recall_tuned_threshold(recall, thresholds, min_recall=0.9)


@pytest.fixture
def alkalinity_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 300
    dates = pd.date_range("2022-04-01", periods=n, freq="D")
    driver = rng.uniform(0, 10, size=n)
    other = rng.uniform(-5, 5, size=n)
    # target is a clean, separable function of driver, so the classifier
    # should reach a high recall at some threshold.
    target = 80 - 4 * driver + rng.normal(0, 1.0, size=n)
    return pd.DataFrame({"driver": driver, "other": other, "target": target}, index=dates)


@pytest.fixture
def toc_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 300
    dates = pd.date_range("2022-04-01", periods=n, freq="D")
    driver = rng.uniform(0, 10, size=n)
    other = rng.uniform(-5, 5, size=n)
    target = 0.3 * driver + rng.normal(0, 0.1, size=n)
    return pd.DataFrame({"driver": driver, "other": other, "target": target}, index=dates)


def test_alkalinity_alert_dates_meets_recall_floor(alkalinity_frame: pd.DataFrame) -> None:
    result = fit_logistic_baseline(
        alkalinity_frame, ["driver", "other"], "target", threshold=60.0, below=True
    )
    alert, threshold = alkalinity_alert_dates(result, min_recall=0.8)

    is_low = (alkalinity_frame.loc[result.scores.index, "target"] < 60.0).to_numpy()
    caught = is_low & alert.to_numpy()
    achieved_recall = caught.sum() / is_low.sum()

    assert achieved_recall >= 0.8
    assert alert.index.equals(result.scores.index)
    assert isinstance(threshold, float)


def test_toc_alert_dates_fires_above_actionable_level(toc_frame: pd.DataFrame) -> None:
    result = fit_quantile_regressor(toc_frame, ["driver", "other"], "target", quantile=0.9)
    alert = toc_alert_dates(result, actionable_level=2.0)

    assert alert.index.equals(result.predictions.index)
    assert (alert == (result.predictions >= 2.0)).all()
    assert alert.any()  # the synthetic driver is strong enough to cross 2.0 sometimes


def test_build_dosing_alert_combines_both_targets(
    alkalinity_frame: pd.DataFrame, toc_frame: pd.DataFrame
) -> None:
    alk_result = fit_logistic_baseline(
        alkalinity_frame, ["driver", "other"], "target", threshold=60.0, below=True
    )
    toc_result = fit_quantile_regressor(toc_frame, ["driver", "other"], "target", quantile=0.9)

    dosing_alert = build_dosing_alert(alk_result, toc_result, min_recall=0.8, toc_actionable_level=2.0)

    assert dosing_alert.lead_time_days == {"TOC_mg_L": 2, "Alk_mg_L": 4}
    # combined_alert is a superset (union) of each individual alert
    assert (dosing_alert.combined_alert.reindex(dosing_alert.alkalinity_alert.index, fill_value=False)
            >= dosing_alert.alkalinity_alert).all()
    assert (dosing_alert.combined_alert.reindex(dosing_alert.toc_alert.index, fill_value=False)
            >= dosing_alert.toc_alert).all()

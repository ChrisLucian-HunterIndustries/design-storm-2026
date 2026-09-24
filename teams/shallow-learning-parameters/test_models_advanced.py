"""Tests for models_advanced.py using small synthetic frames with a known
signal, matching test_models.py's approach.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_advanced import (
    fit_gaussian_process,
    fit_multioutput_random_forest,
    fit_quantile_regressor,
    fit_sarimax_baseline,
    fit_tuned_gradient_boosting,
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


def test_tuned_gradient_boosting_recovers_strong_signal(synthetic_frame: pd.DataFrame) -> None:
    result = fit_tuned_gradient_boosting(synthetic_frame, ["driver", "other"], "target")
    assert result.r2 > 0.8
    assert "n_estimators" in result.best_params
    assert result.importances["driver"] > result.importances["other"]


def test_quantile_regressor_is_reasonably_calibrated(synthetic_frame: pd.DataFrame) -> None:
    result = fit_quantile_regressor(synthetic_frame, ["driver", "other"], "target", quantile=0.9)
    assert result.quantile == 0.9
    # coverage should be in the right ballpark of the requested quantile --
    # not exact, since the synthetic signal is nearly noise-free and a
    # quantile model has little real distribution to calibrate against.
    assert 0.6 <= result.coverage <= 1.0
    assert len(result.predictions) == len(result.actual)


def test_multioutput_random_forest_scores_both_targets(synthetic_frame: pd.DataFrame) -> None:
    rng = np.random.default_rng(1)
    df = synthetic_frame.assign(target2=2.0 * synthetic_frame["driver"] - 3.0 + rng.normal(0, 0.05, len(synthetic_frame)))
    result = fit_multioutput_random_forest(df, ["driver", "other"], ["target", "target2"])
    assert set(result.r2_by_target) == {"target", "target2"}
    assert result.r2_by_target["target"] > 0.7
    assert result.r2_by_target["target2"] > 0.7


def test_gaussian_process_recovers_strong_signal(synthetic_frame: pd.DataFrame) -> None:
    result = fit_gaussian_process(synthetic_frame, ["driver", "other"], "target")
    assert result.r2 > 0.7
    assert len(result.std) == len(result.predictions)
    assert (result.std >= 0).all()


def test_sarimax_baseline_uses_exog_signal(synthetic_frame: pd.DataFrame) -> None:
    result = fit_sarimax_baseline(synthetic_frame, "target", exog_cols=["driver"], order=(0, 0, 0))
    assert result.r2 > 0.7
    assert result.order == (0, 0, 0)
    assert len(result.predictions) == len(result.actual)

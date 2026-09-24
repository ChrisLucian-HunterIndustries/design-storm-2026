"""Model families beyond models.py's core set: hyperparameter tuning,
quantile/peak-focused regression, joint multi-output modeling, Gaussian
Process uncertainty bands, and a time-series-native SARIMAX baseline.

Split out of models.py to keep files under the repo's file-length gate, and
because these are all one step more involved than the "one estimator, one
target" functions there. Same conventions: time-ordered split (no
shuffling), plain DataFrames/Series in and out, dataclass results.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from models import RANDOM_STATE, time_ordered_split


@dataclass
class TunedRegressionResult:
    r2: float
    best_params: dict
    importances: pd.Series


def fit_tuned_gradient_boosting(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> TunedRegressionResult:
    """GridSearchCV over GradientBoostingRegressor hyperparameters, scored
    with TimeSeriesSplit within the training half only (so tuning itself
    never sees the held-out test rows). This is the direct follow-up to
    this catalog's earlier finding that an untuned GradientBoostingRegressor
    underperformed random forest for TOC -- guide.md section 9 uses
    GridSearchCV in Jake's own pipeline; this catalog's model-family
    comparison had not, until now."""
    clean = df.dropna(subset=[*feature_cols, target_col])
    train, test = time_ordered_split(clean)

    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.05, 0.1],
    }
    search = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        param_grid,
        cv=TimeSeriesSplit(n_splits=4),
        scoring="r2",
    )
    search.fit(train[feature_cols], train[target_col])

    predictions = search.best_estimator_.predict(test[feature_cols])
    importances = pd.Series(
        search.best_estimator_.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    return TunedRegressionResult(
        r2=r2_score(test[target_col], predictions),
        best_params=search.best_params_,
        importances=importances,
    )


@dataclass
class QuantileResult:
    quantile: float
    coverage: float  # fraction of held-out actuals at or below the prediction
    predictions: pd.Series
    actual: pd.Series


def fit_quantile_regressor(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, quantile: float = 0.9
) -> QuantileResult:
    """GradientBoostingRegressor with quantile loss: predicts the
    `quantile` percentile of target_col instead of its mean, matching
    guide.md section 10's "care more about catching peaks" (every other
    model in this catalog minimizes mean-squared error, which is not built
    toward that goal). `coverage` close to `quantile` means the predicted
    band is well-calibrated -- e.g. quantile=0.9 should have about 90% of
    held-out actuals fall at or below the prediction."""
    clean = df.dropna(subset=[*feature_cols, target_col])
    train, test = time_ordered_split(clean)

    model = GradientBoostingRegressor(
        loss="quantile", alpha=quantile, n_estimators=300, random_state=RANDOM_STATE
    )
    model.fit(train[feature_cols], train[target_col])
    predictions = pd.Series(model.predict(test[feature_cols]), index=test.index)

    coverage = float((test[target_col] <= predictions).mean())
    return QuantileResult(
        quantile=quantile, coverage=coverage, predictions=predictions, actual=test[target_col]
    )


@dataclass
class MultiOutputResult:
    r2_by_target: dict[str, float]


def fit_multioutput_random_forest(
    df: pd.DataFrame, feature_cols: list[str], target_cols: list[str]
) -> MultiOutputResult:
    """A single RandomForestRegressor fit jointly on every column in
    `target_cols` at once (sklearn's forests accept a 2D y natively, no
    `MultiOutputRegressor` wrapper needed). Answers: does sharing trees
    across TOC_mg_L and Alk_mg_L help or hurt, versus fitting each
    independently (`models.fit_random_forest_importance`)?"""
    clean = df.dropna(subset=[*feature_cols, *target_cols])
    train, test = time_ordered_split(clean)

    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    model.fit(train[feature_cols], train[target_cols])
    predictions = model.predict(test[feature_cols])

    r2_by_target = {
        target: r2_score(test[target], predictions[:, i]) for i, target in enumerate(target_cols)
    }
    return MultiOutputResult(r2_by_target=r2_by_target)


@dataclass
class GaussianProcessResult:
    r2: float
    predictions: pd.Series
    std: pd.Series  # predicted standard deviation, for uncertainty bands
    actual: pd.Series


def fit_gaussian_process(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> GaussianProcessResult:
    """GaussianProcessRegressor (RBF + white-noise kernel, standardized
    features): the one model family in this catalog that returns calibrated
    uncertainty alongside a point estimate. Arguably a better fit than any
    point-estimate model here for "give treatment staff actionable time to
    prepare" -- an uncertainty band is more actionable than an unqualified
    number."""
    clean = df.dropna(subset=[*feature_cols, target_col])
    train, test = time_ordered_split(clean)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_cols])
    x_test = scaler.transform(test[feature_cols])

    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
    model = GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=3, random_state=RANDOM_STATE
    )
    model.fit(x_train, train[target_col])

    predictions, std = model.predict(x_test, return_std=True)
    return GaussianProcessResult(
        r2=r2_score(test[target_col], predictions),
        predictions=pd.Series(predictions, index=test.index),
        std=pd.Series(std, index=test.index),
        actual=test[target_col],
    )


@dataclass
class SarimaxResult:
    r2: float
    predictions: pd.Series
    actual: pd.Series
    order: tuple[int, int, int]


def fit_sarimax_baseline(
    df: pd.DataFrame,
    target_col: str,
    exog_cols: list[str] | None = None,
    order: tuple[int, int, int] = (1, 0, 1),
) -> SarimaxResult:
    """A time-series-native alternative to this catalog's "lag as an
    engineered feature" approach used everywhere else: SARIMAX models the
    target's own autocorrelation directly (the AR/MA terms), optionally
    with exogenous regressors, instead of via a manually shifted predictor
    column. Time-ordered split; a single fit forecasts the whole held-out
    horizon at once (the normal way to evaluate a SARIMAX model, not a
    per-row `.predict()` the way sklearn estimators work here)."""
    columns = [target_col, *(exog_cols or [])]
    clean = df.dropna(subset=columns)
    train, test = time_ordered_split(clean)

    exog_train = train[exog_cols] if exog_cols else None
    exog_test = test[exog_cols] if exog_cols else None

    model = SARIMAX(
        train[target_col],
        exog=exog_train,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    forecast = fitted.get_forecast(steps=len(test), exog=exog_test)
    predictions = pd.Series(forecast.predicted_mean.values, index=test.index)

    return SarimaxResult(
        r2=r2_score(test[target_col], predictions),
        predictions=predictions,
        actual=test[target_col],
        order=order,
    )

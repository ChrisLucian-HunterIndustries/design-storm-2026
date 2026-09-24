"""Shallow-learning routines shared by analyze_parameters.py and visualize.py.

Every model here is deliberately simple (linear/logistic regression, a single
random forest, k-means) per the task's "favor sklearn" steer, and every
function takes/returns plain DataFrames/arrays so it is testable without
plotting or file I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_recall_curve, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42


def time_ordered_split(df: pd.DataFrame, train_fraction: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by position along the (already sorted) date index: earliest
    train_fraction of rows train, the rest test. No shuffling, matching
    guide.md section 8 (test rows must all be later in time than train rows)."""
    cut = int(len(df) * train_fraction)
    return df.iloc[:cut], df.iloc[cut:]


@dataclass
class RegressionResult:
    r2: float
    importances: pd.Series


def fit_random_forest_importance(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> RegressionResult:
    """Time-ordered train/test split, fit a RandomForestRegressor, return
    held-out R^2 and feature importances."""
    clean = df.dropna(subset=[*feature_cols, target_col])
    train, test = time_ordered_split(clean)

    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    model.fit(train[feature_cols], train[target_col])
    predictions = model.predict(test[feature_cols])

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    return RegressionResult(r2=r2_score(test[target_col], predictions), importances=importances)


def fit_gradient_boosting_importance(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> RegressionResult:
    """Time-ordered train/test split, fit a GradientBoostingRegressor,
    return held-out R^2 and feature importances. Boosted trees are the
    model family guide.md's own comparison (Jake's CatBoost) actually won
    with -- this is the plain-sklearn equivalent (no catboost/xgboost
    dependency needed), same features and split as the random forest above
    so the two are directly comparable."""
    clean = df.dropna(subset=[*feature_cols, target_col])
    train, test = time_ordered_split(clean)

    model = GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_STATE)
    model.fit(train[feature_cols], train[target_col])
    predictions = model.predict(test[feature_cols])

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    return RegressionResult(r2=r2_score(test[target_col], predictions), importances=importances)


def fit_linear_baseline(
    df: pd.DataFrame, feature_col: str, target_col: str
) -> tuple[LinearRegression, float]:
    """Single-feature linear baseline (e.g. turb_flow -> TOC), time-ordered
    split. Returns the fitted model and held-out R^2."""
    clean = df.dropna(subset=[feature_col, target_col])
    train, test = time_ordered_split(clean)

    x_train = train[[feature_col]]
    x_test = test[[feature_col]]

    model = LinearRegression()
    model.fit(x_train, train[target_col])
    predictions = model.predict(x_test)
    return model, r2_score(test[target_col], predictions)


def fit_svr_baseline(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> RegressionResult:
    """Support-vector regression (deck slide 9: "like support-vector
    machines"), the model family this catalog originally skipped. Features
    are standardized first since SVR is scale-sensitive, unlike the tree
    models above. SVR has no built-in `feature_importances_`, so importance
    here is permutation importance on the held-out split instead."""
    clean = df.dropna(subset=[*feature_cols, target_col])
    train, test = time_ordered_split(clean)

    model = make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.1))
    model.fit(train[feature_cols], train[target_col])
    predictions = model.predict(test[feature_cols])
    r2 = r2_score(test[target_col], predictions)

    perm = permutation_importance(
        model, test[feature_cols], test[target_col], n_repeats=20, random_state=RANDOM_STATE
    )
    importances = pd.Series(perm.importances_mean, index=feature_cols).sort_values(ascending=False)
    return RegressionResult(r2=r2, importances=importances)


@dataclass
class FullSeriesPrediction:
    frame: pd.DataFrame  # columns: actual, predicted, split ("train"/"test"); indexed by date
    test_r2: float


def predict_full_series(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> FullSeriesPrediction:
    """Fit a RandomForestRegressor on the time-ordered train split, then
    predict every row (train and test) so a viewer can plot the whole
    actual-vs-predicted timeline with the held-out portion clearly marked --
    the "projected TOC and/or alkalinity" curve the deck's web-app bullet
    asks for."""
    clean = df.dropna(subset=[*feature_cols, target_col])
    train, test = time_ordered_split(clean)

    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
    model.fit(train[feature_cols], train[target_col])

    predicted = pd.Series(model.predict(clean[feature_cols]), index=clean.index)
    split = pd.Series(["train"] * len(train) + ["test"] * len(test), index=clean.index)
    frame = pd.DataFrame({"actual": clean[target_col], "predicted": predicted, "split": split})

    test_r2 = r2_score(test[target_col], model.predict(test[feature_cols]))
    return FullSeriesPrediction(frame=frame, test_r2=test_r2)


@dataclass
class RegimeClusters:
    labels: np.ndarray
    coordinates: np.ndarray  # 2D PCA projection, one row per input row
    cluster_sizes: pd.Series


def cluster_hydrologic_regimes(
    df: pd.DataFrame, feature_cols: list[str], n_clusters: int = 3
) -> tuple[pd.DataFrame, RegimeClusters]:
    """Standardize features, project to 2D with PCA, and label each day with
    a KMeans cluster. Unsupervised: no target involved. Returns the rows
    actually used (NaNs dropped) alongside the cluster assignment."""
    clean = df.dropna(subset=feature_cols)
    scaled = StandardScaler().fit_transform(clean[feature_cols])

    coordinates = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(scaled)
    labels = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10).fit_predict(scaled)

    sizes = pd.Series(labels).value_counts().sort_index()
    return clean, RegimeClusters(labels=labels, coordinates=coordinates, cluster_sizes=sizes)


@dataclass
class ThresholdClassifierResult:
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    roc_auc: float
    importances: pd.Series


def fit_threshold_classifier(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float,
    below: bool = True,
) -> ThresholdClassifierResult:
    """Binary classifier for 'will target_col be below/above threshold',
    e.g. alkalinity < 60 (guide.md section 11). Returns a precision/recall
    curve and ROC-AUC on a time-ordered held-out split."""
    clean = df.dropna(subset=[*feature_cols, target_col])
    labels = (clean[target_col] < threshold) if below else (clean[target_col] >= threshold)
    clean = clean.assign(_label=labels.astype(int))

    train, test = time_ordered_split(clean)

    model = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    model.fit(train[feature_cols], train["_label"])
    scores = model.predict_proba(test[feature_cols])[:, 1]

    precision, recall, thresholds = precision_recall_curve(test["_label"], scores)
    auc = roc_auc_score(test["_label"], scores)
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    return ThresholdClassifierResult(
        precision=precision, recall=recall, thresholds=thresholds, roc_auc=auc, importances=importances
    )


def fit_logistic_baseline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    threshold: float,
    below: bool = True,
) -> ThresholdClassifierResult:
    """Same binary classification task as `fit_threshold_classifier`
    (e.g. alkalinity < 60), but a linear `LogisticRegression` baseline
    instead of a random forest -- features are standardized first since
    logistic regression is scale-sensitive. Same return shape, so the two
    are directly comparable."""
    clean = df.dropna(subset=[*feature_cols, target_col])
    labels = (clean[target_col] < threshold) if below else (clean[target_col] >= threshold)
    clean = clean.assign(_label=labels.astype(int))

    train, test = time_ordered_split(clean)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[feature_cols], train["_label"])
    scores = model.predict_proba(test[feature_cols])[:, 1]

    precision, recall, thresholds = precision_recall_curve(test["_label"], scores)
    auc = roc_auc_score(test["_label"], scores)
    coefficients = model.named_steps["logisticregression"].coef_[0]
    importances = pd.Series(np.abs(coefficients), index=feature_cols).sort_values(ascending=False)
    return ThresholdClassifierResult(
        precision=precision, recall=recall, thresholds=thresholds, roc_auc=auc, importances=importances
    )


def lag_correlation_scan(
    predictor: pd.Series, target: pd.Series, max_lag_days: int = 14
) -> pd.Series:
    """For each lag 0..max_lag_days, shift `predictor` forward that many days
    and correlate it against `target` on shared dates. Returns a Series
    indexed by lag (days) holding the Pearson r at that lag.

    This is Scenario 3's "trace a parameter through the system": the lag with
    the largest-magnitude correlation is an empirical estimate of how long a
    signal takes to travel from the upstream sensor to the Foothills lab
    result. It is a statistical fit, not a measured travel time -- guide.md
    section 1 is explicit that Denver Water's own hydraulic model puts the
    physical transit at about four hours, while the multi-day lag that
    actually predicts best is thought to reflect mixing/deposition in the
    reservoir. Do not present this lag as a literal travel time."""
    predictor = predictor.rename("predictor")
    target = target.rename("target")
    scores: dict[int, float] = {}
    for lag in range(max_lag_days + 1):
        shifted = predictor.shift(lag, freq="D")
        joined = pd.concat([shifted, target], axis=1, sort=False).dropna()
        scores[lag] = joined["predictor"].corr(joined["target"]) if len(joined) >= 10 else np.nan
    return pd.Series(scores, name="correlation")


def best_lag(scan: pd.Series) -> int:
    """The lag (days) with the largest-magnitude correlation in a lag scan."""
    return int(scan.abs().idxmax())


@dataclass
class AnomalyResult:
    is_anomaly: pd.Series  # bool, indexed like the input rows actually used
    scores: pd.Series  # IsolationForest's anomaly score, lower = more anomalous


def detect_anomalies(df: pd.DataFrame, feature_cols: list[str], contamination: float = 0.01) -> AnomalyResult:
    """Flag rows that look out of place given `feature_cols`, using
    IsolationForest -- unsupervised, no labeled "bad data" needed. Meant for
    sensor-fault detection (parameters.md's second, previously-unimplemented
    framing): e.g. a value column plus its day-over-day differences will
    flag a sensor stuck at an implausible constant, since a real signal
    rarely holds dead flat while jumping sharply in and back out."""
    clean = df.dropna(subset=feature_cols)
    model = IsolationForest(contamination=contamination, random_state=RANDOM_STATE)
    labels = model.fit_predict(clean[feature_cols])
    scores = model.score_samples(clean[feature_cols])
    return AnomalyResult(
        is_anomaly=pd.Series(labels == -1, index=clean.index),
        scores=pd.Series(scores, index=clean.index),
    )

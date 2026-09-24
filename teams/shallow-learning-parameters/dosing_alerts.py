"""When to add chemical after a spike, not just how well a model fits on
average -- the question this module answers directly, layered on top of
models already built in models.py/models_advanced.py rather than a new
model family:

- Alkalinity has a real yes/no operating threshold (below 60 mg/L, guide.md
  section 11). The existing classifier (`fit_logistic_baseline`/
  `fit_threshold_classifier`) is reused here, but its probability cutoff is
  chosen to hit a target *recall* instead of sklearn's default 0.5 --
  guide.md notes a false alarm just costs unused prep time, while a missed
  low-alkalinity day means the chemical needed didn't go in.
- TOC has no natural yes/no threshold, so the trigger is the *quantile*
  regressor's predicted 90th-percentile band (`fit_quantile_regressor`)
  crossing an actionable level, not the mean-fit point estimate -- this
  targets "catching peaks" the way guide.md section 10 asks for.

Both targets already carry a lag baked into their features (2 days TOC, 4
days alkalinity, guide.md section 8), so the date an alert fires already
carries that many days of lead time before the predicted value arrives at
Foothills -- no separate lead-time calculation is needed.

`build_dosing_alert` answers "how often would this have fired historically"
(a backtest over one held-out split). `score_current_conditions` answers a
different question -- "is a dose required right now" -- by refitting each
model on every labeled row and scoring only the single most recent day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models import RANDOM_STATE, ThresholdClassifierResult
from models_advanced import QuantileResult

TOC_ACTIONABLE_MG_L = 3.0  # guide.md section 9: Jake's own "above 3 mg/L" sample-weight cutoff
ALK_ACTIONABLE_MG_L = 60.0  # guide.md section 11: the below-60 classifier's own threshold
DEFAULT_LEAD_TIME_DAYS = {"TOC_mg_L": 2, "Alk_mg_L": 4}


def recall_tuned_threshold(recall: np.ndarray, thresholds: np.ndarray, min_recall: float = 0.8) -> float:
    """Pick the probability cutoff from a precision_recall_curve that keeps
    recall >= min_recall while maximizing precision. `precision_recall_curve`
    appends one final (precision=1, recall=0) point with no corresponding
    threshold, so only the aligned prefix (`thresholds`' own length) is
    usable. Recall falls as the threshold rises, so the highest threshold
    that still meets the floor is the best-precision choice satisfying it."""
    aligned_recall = recall[: len(thresholds)]
    candidates = thresholds[aligned_recall >= min_recall]
    if len(candidates) == 0:
        raise ValueError(f"no threshold in this precision/recall curve reaches recall >= {min_recall}")
    return float(candidates.max())


def alkalinity_alert_dates(
    result: ThresholdClassifierResult, min_recall: float = 0.8
) -> tuple[pd.Series, float]:
    """Boolean alert series (date-indexed) for 'add chemical for alkalinity',
    using a recall-tuned probability cutoff rather than the default 0.5.
    Returns the alert series and the threshold actually used."""
    threshold = recall_tuned_threshold(result.recall, result.thresholds, min_recall)
    return result.scores >= threshold, threshold


def toc_alert_dates(quantile_result: QuantileResult, actionable_level: float = TOC_ACTIONABLE_MG_L) -> pd.Series:
    """Boolean alert series (date-indexed) for 'add chemical for TOC': fires
    when the predicted 90th-percentile band itself crosses the actionable
    level, not just the mean-fit point estimate."""
    return quantile_result.predictions >= actionable_level


@dataclass
class DosingAlert:
    alkalinity_alert: pd.Series  # bool, date-indexed
    alkalinity_probability: pd.Series
    alkalinity_threshold: float
    toc_alert: pd.Series  # bool, date-indexed
    toc_predicted_p90: pd.Series
    combined_alert: pd.Series  # bool, either target's alert fires
    lead_time_days: dict[str, int]


def build_dosing_alert(
    alkalinity_classifier: ThresholdClassifierResult,
    toc_quantile: QuantileResult,
    min_recall: float = 0.8,
    toc_actionable_level: float = TOC_ACTIONABLE_MG_L,
    lead_time_days: dict[str, int] | None = None,
) -> DosingAlert:
    """Combine both targets' triggers into one calendar. The two inputs come
    from different lag_days frames (TOC=2, Alk=4), so their date ranges
    differ; `combined_alert` is a union over both, treating a date missing
    from one series as "that target had no alert on that date" rather than
    dropping it."""
    lead_time_days = lead_time_days or dict(DEFAULT_LEAD_TIME_DAYS)
    alk_alert, alk_threshold = alkalinity_alert_dates(alkalinity_classifier, min_recall)
    toc_alert = toc_alert_dates(toc_quantile, toc_actionable_level)

    combined = pd.concat([alk_alert.rename("alk"), toc_alert.rename("toc")], axis=1, sort=False).fillna(False)
    combined_alert = combined["alk"] | combined["toc"]

    return DosingAlert(
        alkalinity_alert=alk_alert,
        alkalinity_probability=alkalinity_classifier.scores,
        alkalinity_threshold=alk_threshold,
        toc_alert=toc_alert,
        toc_predicted_p90=toc_quantile.predictions,
        combined_alert=combined_alert,
        lead_time_days=lead_time_days,
    )


@dataclass
class CurrentDoseDecision:
    as_of_toc_date: pd.Timestamp
    toc_predicted_p90: float
    toc_dose_now: bool
    as_of_alkalinity_date: pd.Timestamp
    alkalinity_probability: float
    alkalinity_dose_now: bool
    dose_now: bool


def score_current_conditions(
    df_toc: pd.DataFrame,
    toc_features: list[str],
    df_alk: pd.DataFrame,
    alk_features: list[str],
    alkalinity_threshold: float,
    toc_actionable_level: float = TOC_ACTIONABLE_MG_L,
    alkalinity_actionable_level: float = ALK_ACTIONABLE_MG_L,
    toc_target_col: str = "TOC_mg_L",
    alk_target_col: str = "Alk_mg_L",
) -> CurrentDoseDecision:
    """Answers "is a dose required right now", not "how often would this have
    fired historically" (that's `build_dosing_alert` above, scored on one
    held-out backtest split). The distinction matters: the alert calendar's
    numbers describe a fixed past test window, while this function scores
    only the single most recent day available, refitting each model on
    every historically labeled row first -- a live decision should use all
    the history on hand, since the held-out split's only job was to validate
    the approach honestly, which the backtest above already did.

    The most recent row's own target value (if this particular snapshot
    happens to have one) is never read here, only its feature columns --
    this deliberately reproduces what a live deployment would do against a
    real-time upstream feed, where that day's lab result genuinely isn't in
    yet. `alkalinity_threshold` should be the recall-tuned cutoff from
    `alkalinity_alert_dates` on a backtest (sklearn's default 0.5 does not
    match the false-alarm/missed-detection cost asymmetry guide.md
    describes), not a hand-picked number.
    """
    toc_train = df_toc.dropna(subset=[*toc_features, toc_target_col])
    toc_latest = df_toc.dropna(subset=toc_features).iloc[[-1]]
    toc_model = GradientBoostingRegressor(
        loss="quantile", alpha=0.9, n_estimators=300, random_state=RANDOM_STATE
    )
    toc_model.fit(toc_train[toc_features], toc_train[toc_target_col])
    toc_pred = float(toc_model.predict(toc_latest[toc_features])[0])

    alk_train = df_alk.dropna(subset=[*alk_features, alk_target_col])
    alk_latest = df_alk.dropna(subset=alk_features).iloc[[-1]]
    alk_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    alk_labels = (alk_train[alk_target_col] < alkalinity_actionable_level).astype(int)
    alk_model.fit(alk_train[alk_features], alk_labels)
    alk_prob = float(alk_model.predict_proba(alk_latest[alk_features])[:, 1][0])

    toc_dose_now = toc_pred >= toc_actionable_level
    alk_dose_now = alk_prob >= alkalinity_threshold

    return CurrentDoseDecision(
        as_of_toc_date=toc_latest.index[0],
        toc_predicted_p90=toc_pred,
        toc_dose_now=toc_dose_now,
        as_of_alkalinity_date=alk_latest.index[0],
        alkalinity_probability=alk_prob,
        alkalinity_dose_now=alk_dose_now,
        dose_now=toc_dose_now or alk_dose_now,
    )

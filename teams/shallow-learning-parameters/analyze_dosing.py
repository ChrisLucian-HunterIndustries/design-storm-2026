"""Report section for the chemical-dosing alert built in dosing_alerts.py:
answers "when do we know to add the chemical after a spike" (scenario1's
own writeup) rather than reporting model fit quality in isolation.

Split out from analyze_parameters.py (already near this repo's file-length
gate) the same way analyze_advanced.py/analyze_sonde.py were -- its section
is appended by analyze_parameters.main() the same way.
"""
from __future__ import annotations

import pandas as pd

from dosing_alerts import (
    ALK_ACTIONABLE_MG_L,
    TOC_ACTIONABLE_MG_L,
    build_dosing_alert,
)
from models import fit_logistic_baseline
from models_advanced import fit_quantile_regressor

MIN_RECALL = 0.8


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def build_dosing_alert_section(
    df_toc: pd.DataFrame,
    toc_features: list[str],
    df_alk: pd.DataFrame,
    alk_features: list[str],
    min_recall: float = MIN_RECALL,
) -> list[str]:
    """Fits the alkalinity below-60 classifier (`fit_logistic_baseline`,
    the model family that won the family comparison above) and the TOC
    90th-percentile quantile regressor, both already in this catalog, then
    combines them into one dosing-alert calendar via
    `dosing_alerts.build_dosing_alert`."""
    alk_result = fit_logistic_baseline(
        df_alk, alk_features, "Alk_mg_L", threshold=ALK_ACTIONABLE_MG_L, below=True
    )
    toc_result = fit_quantile_regressor(df_toc, toc_features, "TOC_mg_L", quantile=0.9)
    alert = build_dosing_alert(alk_result, toc_result, min_recall=min_recall)

    is_low = df_alk.loc[alert.alkalinity_alert.index, "Alk_mg_L"] < ALK_ACTIONABLE_MG_L
    caught = is_low & alert.alkalinity_alert
    achieved_recall = caught.sum() / is_low.sum() if is_low.sum() else float("nan")

    combined_dates = alert.combined_alert[alert.combined_alert].index.sort_values()
    example_dates = ", ".join(d.date().isoformat() for d in combined_dates[:8])
    if len(combined_dates) > 8:
        example_dates += f", ... ({len(combined_dates) - 8} more)"

    return [
        f"Alkalinity trigger: `fit_logistic_baseline` probability of Alk_mg_L < "
        f"{ALK_ACTIONABLE_MG_L:.0f} mg/L, thresholded to hold recall >= {min_recall} "
        f"(picked cutoff = {_fmt(alert.alkalinity_threshold)}, achieved recall = "
        f"{_fmt(achieved_recall)} on the {int(is_low.sum())} held-out low-alkalinity days).\n",
        f"TOC trigger: `fit_quantile_regressor`'s predicted 90th-percentile band crossing "
        f"{TOC_ACTIONABLE_MG_L:.0f} mg/L (Jake's own sample-weight cutoff, guide.md section 9).\n",
        "| trigger | held-out days flagged | out of | lead time (days) |",
        "|---|---:|---:|---:|",
        f"| Alkalinity < {ALK_ACTIONABLE_MG_L:.0f} mg/L | {int(alert.alkalinity_alert.sum())} | "
        f"{len(alert.alkalinity_alert)} | {alert.lead_time_days['Alk_mg_L']} |",
        f"| TOC predicted p90 >= {TOC_ACTIONABLE_MG_L:.0f} mg/L | {int(alert.toc_alert.sum())} | "
        f"{len(alert.toc_alert)} | {alert.lead_time_days['TOC_mg_L']} |",
        f"| Combined (either trigger) | {int(alert.combined_alert.sum())} | "
        f"{len(alert.combined_alert)} | n/a (union of the two above) |",
        "",
        f"Combined alert dates (held-out split): {example_dates or 'none'}\n",
        "Each alert date already reflects its target's own lag (2 days TOC, 4 days alkalinity) -- "
        "the model's prediction for that date is built entirely from upstream readings that many "
        "days old, so the date it fires is the lead time itself, not a separate estimate of one.",
    ]

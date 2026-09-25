"""Demonstration, not a backtestable experiment: today's NWS forecast for
Strontia Springs Reservoir, framed as "does an incoming storm justify a
closer look" -- the same live-conditions framing
dosing_alerts.score_current_conditions already uses, for the same reason
(no historical forecast archive to backtest against, see
nws_forecast_loader.py's docstring). This is the sixth public-data idea in
this catalog, and the only one that cannot be tested with a held-out R^2.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from nws_forecast_loader import load_forecast_periods

FORECAST_PATH = Path(__file__).resolve().parent / "nws_forecast_snapshot.json"


def summarize_forecast(periods: pd.DataFrame) -> dict:
    """The single highest precip-probability period in the forecast window,
    plus how many periods were returned -- enough to answer "should we
    expect a storm soon" without pretending this can be scored with a
    held-out R^2."""
    valid = periods.dropna(subset=["precip_probability_pct"])
    if valid.empty:
        return {"max_precip_probability_pct": float("nan"), "period_name": None, "n_periods": len(periods)}
    peak = valid.loc[valid["precip_probability_pct"].idxmax()]
    return {
        "max_precip_probability_pct": peak["precip_probability_pct"],
        "period_name": peak["name"],
        "n_periods": len(periods),
    }


def build_nws_forecast_section() -> list[str]:
    periods = load_forecast_periods(FORECAST_PATH)
    summary = summarize_forecast(periods)

    lines = [
        "\n## NWS live-forecast demonstration: is a storm forecast right now? "
        "(new dataset, not in data/, not backtestable)\n",
        "The National Weather Service's gridpoint forecast API "
        "(https://api.weather.gov, Strontia Springs Reservoir centroid) is a public "
        "dataset not otherwise used in this catalog. **It cannot be tested the way "
        "every other public dataset in this catalog was** -- NWS has no historical "
        "forecast archive, so there is no way to ask \"how well would this have "
        "predicted 2023's storms\"; it can only ever be demonstrated on the current "
        "moment (fetch_nws_forecast.py, snapshot saved to nws_forecast_snapshot.json), "
        "the same live-conditions framing dosing_alerts.score_current_conditions "
        "already uses for the same reason (this repo's data ends on a fixed date with "
        "no \"sensor exists, lab result doesn't yet\" row to genuinely backtest "
        "against).\n",
        f"\nSnapshot fetched 2026-09-25: {summary['n_periods']} forecast periods. "
        f"Highest precipitation probability: **{summary['max_precip_probability_pct']:.0f}%** "
        f"({summary['period_name']}).\n",
        "\n| period | start | temperature (F) | precip probability | forecast |",
        "|---|---|---:|---:|---|",
    ]
    for row in periods.itertuples():
        prob = "n/a" if pd.isna(row.precip_probability_pct) else f"{row.precip_probability_pct:.0f}%"
        lines.append(
            f"| {row.name} | {row.start_time.strftime('%Y-%m-%d %H:%M')} | {row.temperature} | {prob} | "
            f"{row.short_forecast} |"
        )
    return lines

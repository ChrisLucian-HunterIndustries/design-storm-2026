"""Figure 37: the NWS live-forecast demonstration in analyze_nws_forecast.py.
Split out to keep files under the repo's file-length gate; see visualize.py
for the entry point (`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_nws_forecast import FORECAST_PATH
from nws_forecast_loader import load_forecast_periods


def plot_nws_forecast_precip(out_path: Path) -> None:
    """Fig 37: precipitation probability across the current NWS forecast
    window -- a live demonstration, not a backtested experiment (see
    analyze_nws_forecast.py's docstring for why no R^2 is possible here)."""
    periods = load_forecast_periods(FORECAST_PATH)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = ["tab:blue" if v >= 40 else "tab:gray" for v in periods["precip_probability_pct"].fillna(0)]
    ax.bar(periods["name"], periods["precip_probability_pct"], color=colors)
    ax.set_ylabel("precipitation probability (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", labelrotation=60, labelsize=7)
    ax.set_title(f"NWS forecast, Strontia Springs Reservoir (fetched {FORECAST_PATH.stat().st_mtime_ns and 'live'})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

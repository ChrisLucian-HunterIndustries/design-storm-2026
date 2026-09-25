"""Figure 38: the melt-out date regression in analyze_snowmelt.py. Split
out to keep files under the repo's file-length gate; see visualize.py for
the entry point (`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_snowmelt import compute_melt_out_summary


def plot_melt_out_regression(data_dir: Path, out_path: Path) -> None:
    """Fig 38: peak SWE vs. melt-out day-of-year, one point per water year
    -- too few years (4-5) for a held-out split, so this shows the
    full-sample relationship only, each point labeled by year."""
    summary = compute_melt_out_summary(data_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    if summary.empty:
        ax.text(0.5, 0.5, "No water year with both a peak and a melt-out in this record", ha="center", va="center")
    else:
        ax.scatter(summary["peak_swe"], summary["melt_out_day_of_year"], color="tab:blue", s=60, zorder=3)
        for row in summary.itertuples():
            ax.annotate(
                str(row.year), (row.peak_swe, row.melt_out_day_of_year), textcoords="offset points", xytext=(6, 4)
            )
    ax.set_xlabel("peak SWE (in)")
    ax.set_ylabel("melt-out day of year")
    ax.set_title(f"Melt-out date vs. peak snowpack ({len(summary)} water years -- too few for a held-out split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

"""Melt-out date regression -- parameters.md's own long-documented-but-
never-built idea: "predicting SWE decline/melt-out date from temperature
is a smaller, well-posed regression." Implemented here for the first time.

Real constraint, stated up front: this record spans only ~4.5 melt seasons
(2022-2026), so there are only 4-5 possible rows -- far too few for a
train/test split (every other model in this catalog uses one). The
honest scope here is a full-sample correlation/relationship, not a
held-out R^2, and that's stated explicitly in the report rather than
forcing a split that would be meaningless at this sample size.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import load_snowpack, load_weather

MELT_OUT_THRESHOLD_IN = 0.5  # SWE below this counts as "melted out"


def compute_melt_out_summary(data_dir: Path) -> pd.DataFrame:
    """One row per water year with a real peak: peak SWE, the date it
    occurred, the first date afterward SWE drops below
    MELT_OUT_THRESHOLD_IN ("melt-out"), and that date's day-of-year, plus
    the year's mean spring (Apr-Jun) TMAX as the temperature predictor."""
    snow = load_snowpack(data_dir)["SWE"]
    weather = load_weather(data_dir)

    records = []
    for year, group in snow.groupby(snow.index.year):
        if group.max() <= 0:
            continue
        peak_date = group.idxmax()
        peak_swe = group.max()

        after_peak = group.loc[peak_date:]
        melted = after_peak[after_peak <= MELT_OUT_THRESHOLD_IN]
        if melted.empty:
            continue
        melt_out_date = melted.index[0]

        spring_tmax = weather.loc[f"{year}-04-01":f"{year}-06-30", "TMAX"]
        records.append(
            {
                "year": year,
                "peak_swe": peak_swe,
                "peak_swe_date": peak_date,
                "melt_out_date": melt_out_date,
                "melt_out_day_of_year": melt_out_date.dayofyear,
                "mean_spring_tmax": spring_tmax.mean(),
            }
        )
    return pd.DataFrame.from_records(records)


def build_melt_out_section(data_dir: Path) -> list[str]:
    summary = compute_melt_out_summary(data_dir)

    lines = [
        "\n## Melt-out date regression (parameters.md: \"predicting SWE decline/melt-out "
        "date from temperature\" -- documented since the first session, never built until now)\n",
        f"Only {len(summary)} water years have a computable melt-out date in this record -- far "
        "too few for a held-out train/test split (every other model in this catalog uses one). "
        "Reporting the full-sample relationship only, not a held-out R^2, rather than forcing a "
        "split that would be meaningless at this sample size.\n",
        "\n| year | peak SWE (in) | peak date | melt-out date | melt-out day-of-year | mean spring TMAX (F) |",
        "|---:|---:|---|---|---:|---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.year} | {row.peak_swe:.1f} | {row.peak_swe_date.date()} | {row.melt_out_date.date()} | "
            f"{row.melt_out_day_of_year} | {row.mean_spring_tmax:.1f} |"
        )

    if len(summary) >= 3:
        peak_corr = summary["peak_swe"].corr(summary["melt_out_day_of_year"])
        temp_corr = summary["mean_spring_tmax"].corr(summary["melt_out_day_of_year"])
        lines.append(
            f"\nPearson r, peak SWE vs. melt-out day-of-year: **{peak_corr:.3f}** (more snow -> later "
            f"melt-out, the expected physical direction, if positive). Pearson r, mean spring TMAX vs. "
            f"melt-out day-of-year: **{temp_corr:.3f}** (warmer spring -> earlier melt-out, if negative).\n"
        )
    else:
        lines.append(f"\nOnly {len(summary)} years -- too few to compute a meaningful correlation.\n")
    return lines

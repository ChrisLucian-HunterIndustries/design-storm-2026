"""Figure 40: the five follow-up fire/inflow/outflow ideas in
analyze_fire_features.py. Split out to keep files under the repo's
file-length gate; see visualize.py for the entry point (`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_burn_scar import BASIN_PATH, MTBS_PATH
from analyze_fire_features import (
    OUTFLOW_PATH,
    RESERVOIR_LAT,
    RESERVOIR_LON,
    STORAGE_PATH,
    _fire_feature_model_table,
)
from burn_scar_loader import filter_fires_in_basin, load_mtbs_fires


def _r2_from_table_lines(lines: list[str], label: str, target: str) -> float:
    for line in lines:
        if line.startswith(f"| {target} | {label} |"):
            return float(line.rsplit("|", 2)[1].strip())
    return float("nan")


def plot_fire_feature_ideas(
    data_dir: Path, national_features: dict[str, list[str]], lag_days: dict[str, int], out_path: Path
) -> None:
    """Baseline vs. each of the five ideas vs. all combined, per target."""
    fires_in_basin = filter_fires_in_basin(load_mtbs_fires(MTBS_PATH), BASIN_PATH)
    lines = _fire_feature_model_table(
        data_dir, OUTFLOW_PATH, STORAGE_PATH, fires_in_basin, RESERVOIR_LAT, RESERVOIR_LON, national_features, lag_days
    )

    labels = [
        "baseline",
        "+ fire\npressure",
        "+ fire x\nprecip",
        "+ cumulative\nacres",
        "+ distance\nweighted",
        "+ residence\ntime",
        "+ all\ncombined",
    ]
    row_labels = [
        "baseline",
        "+ fire_pressure",
        "+ fire_precip_interaction",
        "+ cumulative_burned_acres",
        "+ distance_weighted_fire_pressure",
        "+ residence_time_days",
        "+ all combined",
    ]
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, target in zip(axes, ("TOC_mg_L", "Alk_mg_L")):
        values = [_r2_from_table_lines(lines, row_label, target) for row_label in row_labels]
        ax.bar(labels, values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(target)
    axes[0].set_ylabel("held-out R²")

    fig.suptitle("Five more fire/inflow/outflow ideas: does any beat the baseline?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

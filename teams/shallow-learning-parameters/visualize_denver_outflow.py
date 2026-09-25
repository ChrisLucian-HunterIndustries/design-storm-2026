"""Figure 39: Denver Water's own outflow (COND20CO/COND26CO), combined
with the MTBS burn-scar fire feature, from analyze_denver_outflow.py. Split
out to keep files under the repo's file-length gate; see visualize.py for
the entry point (`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_denver_outflow import OUTFLOW_PATH, _denver_outflow_model_table
from analyze_burn_scar import BASIN_PATH, MTBS_PATH
from burn_scar_loader import filter_fires_in_basin, load_mtbs_fires


def _r2_from_table_lines(lines: list[str], label: str, target: str) -> float:
    for line in lines:
        if line.startswith(f"| {target} | {label} |"):
            return float(line.rsplit("|", 2)[1].strip())
    return float("nan")


def plot_denver_outflow_comparison(
    data_dir: Path, national_features: dict[str, list[str]], lag_days: dict[str, int], out_path: Path
) -> None:
    """Baseline vs. +Outflow_CFS vs. +days_since_fire vs. both combined,
    per target."""
    fires_in_basin = filter_fires_in_basin(load_mtbs_fires(MTBS_PATH), BASIN_PATH)
    lines = _denver_outflow_model_table(data_dir, OUTFLOW_PATH, fires_in_basin, national_features, lag_days)

    labels = ["baseline", "+ Outflow_CFS", "+ days_since_fire", "+ Outflow_CFS\n+ days_since_fire"]
    colors = ["tab:gray", "tab:blue", "tab:orange", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, target in zip(axes, ("TOC_mg_L", "Alk_mg_L")):
        values = [
            _r2_from_table_lines(lines, "baseline", target),
            _r2_from_table_lines(lines, "+ Outflow_CFS", target),
            _r2_from_table_lines(lines, "+ days_since_fire", target),
            _r2_from_table_lines(lines, "+ Outflow_CFS + days_since_fire", target),
        ]
        ax.bar(labels, values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(target)
    axes[0].set_ylabel("held-out R²")

    fig.suptitle("Denver Water's own outflow + fire history: does combining them help?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

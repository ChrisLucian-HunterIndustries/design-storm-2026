"""Figure 36: the MTBS burn-scar experiment in analyze_burn_scar.py,
including its time-index-artifact controls. Split out to keep files under
the repo's file-length gate; see visualize.py for the entry point
(`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_burn_scar import (
    BASIN_PATH,
    MTBS_PATH,
    _burn_scar_model_table,
    _time_index_control_table,
)
from burn_scar_loader import filter_fires_in_basin, load_mtbs_fires


def _r2_from_table_lines(lines: list[str], label: str, target: str) -> float:
    for line in lines:
        if line.startswith(f"| {target} | {label} |"):
            return float(line.rsplit("|", 2)[1].strip())
    return float("nan")


def plot_burn_scar_comparison(
    data_dir: Path, national_features: dict[str, list[str]], lag_days: dict[str, int], out_path: Path
) -> None:
    """Does the burn-scar feature's apparent improvement survive the
    time-index controls, or is it an artifact? Shows both the real feature
    and all three date-only controls side by side, per target."""
    fires_in_basin = filter_fires_in_basin(load_mtbs_fires(MTBS_PATH), BASIN_PATH)
    model_lines = _burn_scar_model_table(data_dir, fires_in_basin, national_features, lag_days)
    control_lines = _time_index_control_table(data_dir, national_features, lag_days)

    labels = [
        "no feature\n(baseline)",
        "days since\nrecord start",
        "days since fire\n(single segment)",
        "binary\nbefore/after",
        "days_since_fire\n(real feature)",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, target in zip(axes, ("TOC_mg_L", "Alk_mg_L")):
        values = [
            _r2_from_table_lines(model_lines, "without burn-scar feature", target),
            _r2_from_table_lines(control_lines, "days since record start (no fire semantics)", target),
            _r2_from_table_lines(
                control_lines,
                f"days since 2023-03-30 only, single segment (no earlier fire history)",
                target,
            ),
            _r2_from_table_lines(control_lines, "binary before/after 2023-03-30 (no fire semantics)", target),
            _r2_from_table_lines(model_lines, "with burn-scar feature", target),
        ]
        colors = ["tab:gray", "tab:orange", "tab:orange", "tab:orange", "tab:red"]
        ax.bar(labels, values, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(target)
    axes[0].set_ylabel("held-out R²")

    fig.suptitle("Burn-scar feature vs. date-only controls (orange) -- is the jump real?")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

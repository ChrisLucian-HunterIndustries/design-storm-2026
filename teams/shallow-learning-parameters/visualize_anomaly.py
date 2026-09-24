"""Figure 18: sensor-fault anomaly detection (previously an unimplemented
idea in parameters.md). Split out of visualize.py to keep files under the
repo's file-length gate; see visualize.py for the entry point (`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_michigan_creek
from models import detect_anomalies


def plot_anomaly_detection(data_dir: Path, out_path: Path) -> None:
    """Fig 18: MichiganCreek's SWE record with IsolationForest-flagged days
    marked -- validated against the one labeled bad patch this repo already
    documents (AGENTS.md: SWE 9.0 on 2026-05-12 to 05-15, a stuck sensor)."""
    creek = load_michigan_creek(data_dir)
    features = pd.DataFrame(index=creek.index)
    features["SWE"] = creek["SWE"]
    features["diff_prev"] = creek["SWE"].diff().abs()
    features["diff_next"] = creek["SWE"].diff(-1).abs()
    features["rolling_std_5"] = creek["SWE"].rolling(5, center=True).std()

    feature_cols = ["SWE", "diff_prev", "diff_next", "rolling_std_5"]
    result = detect_anomalies(features, feature_cols, contamination=0.01)
    flagged = result.is_anomaly[result.is_anomaly].index

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(creek.index, creek["SWE"], color="tab:blue", linewidth=0.9, label="SWE")
    ax.scatter(flagged, creek.loc[flagged, "SWE"], color="tab:red", zorder=5, s=18, label="flagged anomaly")
    ax.set_ylabel("SWE (in)")
    ax.set_title("MichiganCreek.csv: IsolationForest-flagged anomalies")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

"""Figure 26: the chemical-dosing alert calendar (dosing_alerts.py /
analyze_dosing.py) -- when the model says to add chemical, not just how
well it fits on average. Split out to its own module the same way
visualize_advanced.py/visualize_sonde.py were, to keep visualize.py under
the repo's file-length gate; see visualize.py for the entry point
(`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dosing_alerts import ALK_ACTIONABLE_MG_L, TOC_ACTIONABLE_MG_L, build_dosing_alert
from models import fit_logistic_baseline
from models_advanced import fit_quantile_regressor


def plot_dosing_alert_timeline(
    df_toc: pd.DataFrame, toc_features: list[str], df_alk: pd.DataFrame, alk_features: list[str], out_path: Path
) -> None:
    """Fig 26: TOC's predicted 90th-percentile band against its actionable
    level, and alkalinity's classifier probability against its recall-tuned
    cutoff, each with the resulting alert dates marked -- the visual answer
    to "when do we know to add the chemical after a spike"."""
    alk_result = fit_logistic_baseline(
        df_alk, alk_features, "Alk_mg_L", threshold=ALK_ACTIONABLE_MG_L, below=True
    )
    toc_result = fit_quantile_regressor(df_toc, toc_features, "TOC_mg_L", quantile=0.9)
    alert = build_dosing_alert(alk_result, toc_result)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=False)

    ax = axes[0]
    ax.plot(toc_result.actual.index, toc_result.actual, color="tab:blue", linewidth=1.0, label="actual TOC_mg_L")
    ax.plot(
        alert.toc_predicted_p90.index, alert.toc_predicted_p90, color="tab:red", linewidth=1.0,
        label="predicted 90th percentile",
    )
    ax.axhline(TOC_ACTIONABLE_MG_L, color="black", linestyle=":", linewidth=1, label=f"actionable level ({TOC_ACTIONABLE_MG_L:.0f} mg/L)")
    fire_dates = alert.toc_alert[alert.toc_alert].index
    ax.scatter(fire_dates, alert.toc_predicted_p90.loc[fire_dates], color="tab:red", s=22, zorder=5, label="dosing alert")
    ax.set_title(f"TOC: predicted-p90-crosses-{TOC_ACTIONABLE_MG_L:.0f}mg/L trigger ({len(fire_dates)} alert days)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(
        alert.alkalinity_probability.index, alert.alkalinity_probability, color="tab:purple", linewidth=1.0,
        label=f"P(Alk_mg_L < {ALK_ACTIONABLE_MG_L:.0f})",
    )
    ax.axhline(
        alert.alkalinity_threshold, color="black", linestyle=":", linewidth=1,
        label=f"recall-tuned cutoff ({alert.alkalinity_threshold:.2f})",
    )
    fire_dates = alert.alkalinity_alert[alert.alkalinity_alert].index
    ax.scatter(
        fire_dates, alert.alkalinity_probability.loc[fire_dates], color="tab:purple", s=22, zorder=5,
        label="dosing alert",
    )
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Alkalinity: recall-tuned classifier trigger ({len(fire_dates)} alert days)")
    ax.legend(fontsize=8)

    fig.suptitle("Chemical-dosing alert calendar: when to add chemical after a spike (held-out split)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

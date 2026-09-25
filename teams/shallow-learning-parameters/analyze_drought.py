"""Experiment: does the U.S. Drought Monitor's weekly county drought
severity index (DSCI, a public dataset not otherwise used in this
catalog) improve TOC/alkalinity predictions -- on the full multi-year
record, AND specifically on the sonde's limited 4-month window, where
Scenario 1's "is there public data to fine-tune this until we have more
[sonde] data" question is most pressing? See drought_loader.py for
parsing/lag-safe alignment, fetch_usdm.py for how usdm_jefferson.csv was
obtained.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import build_dataset
from drought_loader import dsci_feature_for_dates, load_usdm
from models import fit_random_forest_importance

USDM_PATH = Path(__file__).resolve().parent / "usdm_jefferson.csv"


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _add_dsci_feature(df: pd.DataFrame, usdm: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DSCI_prev_week"] = dsci_feature_for_dates(usdm, out.index)
    return out


def _drought_correlation_table(df_toc: pd.DataFrame, df_alk: pd.DataFrame, usdm: pd.DataFrame) -> list[str]:
    """A direct grounding check before touching any model: is DSCI even
    correlated with either target at all, on its own?"""
    toc_dsci = dsci_feature_for_dates(usdm, df_toc.index)
    alk_dsci = dsci_feature_for_dates(usdm, df_alk.index)
    toc_corr = pd.concat([df_toc["TOC_mg_L"], toc_dsci], axis=1).corr().iloc[0, 1]
    alk_corr = pd.concat([df_alk["Alk_mg_L"], alk_dsci], axis=1).corr().iloc[0, 1]
    return [
        "| target | Pearson r with DSCI (most recently ended week) |",
        "|---|---:|",
        f"| TOC_mg_L | {_fmt(toc_corr)} |",
        f"| Alk_mg_L | {_fmt(alk_corr)} |",
    ]


def _drought_model_table(
    data_dir: Path, usdm: pd.DataFrame, toc_features: list[str], alk_features: list[str]
) -> list[str]:
    """Held-out R^2 with vs. without DSCI_prev_week, same random forest, on
    the full multi-year record (uniform lag, this catalog's long-standing
    baseline)."""
    df_toc = build_dataset(data_dir, lag_days=2)
    df_alk = build_dataset(data_dir, lag_days=4)
    toc_r2 = fit_random_forest_importance(df_toc, toc_features, "TOC_mg_L").r2
    alk_r2 = fit_random_forest_importance(df_alk, alk_features, "Alk_mg_L").r2

    df_toc_dsci = _add_dsci_feature(df_toc, usdm)
    df_alk_dsci = _add_dsci_feature(df_alk, usdm)
    toc_dsci_r2 = fit_random_forest_importance(df_toc_dsci, [*toc_features, "DSCI_prev_week"], "TOC_mg_L").r2
    alk_dsci_r2 = fit_random_forest_importance(df_alk_dsci, [*alk_features, "DSCI_prev_week"], "Alk_mg_L").r2

    return [
        "| target | features | held-out R² (full multi-year record, uniform lag) |",
        "|---|---|---:|",
        f"| TOC_mg_L | without DSCI | {_fmt(toc_r2)} |",
        f"| TOC_mg_L | with DSCI | {_fmt(toc_dsci_r2)} |",
        f"| Alk_mg_L | without DSCI | {_fmt(alk_r2)} |",
        f"| Alk_mg_L | with DSCI | {_fmt(alk_dsci_r2)} |",
    ]


def _drought_limited_window_table(
    data_dir: Path,
    sonde_readings: pd.DataFrame,
    usdm: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
) -> list[str]:
    """The question actually asked: does this new public dataset help the
    sonde's limited 4-month window specifically, where analyze_sonde's
    feature-set comparison already found every existing feature set
    (national, sonde, combined) scoring at or below a mean-only baseline?
    Adds DSCI to the "national only" feature set, restricted to the same
    window, and refits."""
    start = sonde_readings["timestamp"].min().normalize()
    end = sonde_readings["timestamp"].max().normalize()

    lines = [
        f"Same {start.date()} to {end.date()} sonde window as the earlier feature-set "
        "comparison -- does adding DSCI to the national-only feature set change the "
        "result there?\n",
        "| target | features | rows after dropna | held-out R² |",
        "|---|---|---:|---:|",
    ]
    for target in ("TOC_mg_L", "Alk_mg_L"):
        lag = lag_days[target]
        cols = national_features[target]
        national_df = build_dataset(data_dir, lag_days=lag).loc[start:end]
        national_dsci_df = _add_dsci_feature(national_df, usdm)
        dsci_cols = [*cols, "DSCI_prev_week"]

        n_rows = len(national_df.dropna(subset=[*cols, target]))
        base_r2 = fit_random_forest_importance(national_df, cols, target).r2
        n_rows_dsci = len(national_dsci_df.dropna(subset=[*dsci_cols, target]))
        dsci_r2 = fit_random_forest_importance(national_dsci_df, dsci_cols, target).r2

        lines.append(f"| {target} | national only (baseline) | {n_rows} | {_fmt(base_r2)} |")
        lines.append(f"| {target} | national + DSCI | {n_rows_dsci} | {_fmt(dsci_r2)} |")
    return lines


def build_drought_experiment_section(
    data_dir: Path,
    sonde_readings: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
) -> list[str]:
    usdm = load_usdm(USDM_PATH)
    df_toc = build_dataset(data_dir, lag_days=lag_days["TOC_mg_L"])
    df_alk = build_dataset(data_dir, lag_days=lag_days["Alk_mg_L"])

    lines = [
        "\n## U.S. Drought Monitor experiment: does county drought severity help? "
        "(new dataset, not in data/)\n",
        "The U.S. Drought Monitor's weekly county drought-severity index "
        "(https://usdmdataservices.unl.edu, Jefferson County CO -- DSCI = D0+D1+D2+D3+D4, "
        "range 0-500) is a public dataset not otherwise used anywhere in this catalog -- "
        "weekly and county-specific, a higher-frequency and more local public signal than "
        "NOAA's ONI (analyzed above). Snapshot saved to usdm_jefferson.csv (fetch_usdm.py); "
        "DSCI_prev_week uses each day's most recently *fully ended* week, avoiding "
        "lookahead (drought_loader.py).\n",
    ]
    lines += _drought_correlation_table(df_toc, df_alk, usdm)
    lines.append("")
    lines += _drought_model_table(
        data_dir, usdm, national_features["TOC_mg_L"], national_features["Alk_mg_L"]
    )
    lines.append("\n### Does it help Scenario 1's limited 4-month sonde window specifically?\n")
    lines += _drought_limited_window_table(data_dir, sonde_readings, usdm, national_features, lag_days)
    return lines

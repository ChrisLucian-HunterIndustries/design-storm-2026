"""Experiment: does a burn-scar recency feature (days since the nearest
fire's ignition date within the watershed, MTBS data) improve TOC/alkalinity
predictions? A real fire (the "403" fire, ignited 2023-03-30, 1769 acres)
was found -- via a real point-in-polygon check against the actual drainage
basin, not just a bounding box -- inside the South Platte watershed above
Strontia Springs, squarely within this record's 2022-2026 coverage. This is
general water-treatment knowledge, not something stated in this repo's own
materials: burned watersheds are documented to elevate post-storm turbidity
and TOC due to reduced soil infiltration and ash/sediment mobilization.

See burn_scar_loader.py for the fire data (fetch_mtbs.py) and the
lag-safe/lookahead-safe daily feature; fetch_mtbs.py for how mtbs_fires.json
was obtained.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from burn_scar_loader import days_since_fire_feature, filter_fires_in_basin, load_mtbs_fires
from data_loader import build_dataset
from models import fit_random_forest_importance

MTBS_PATH = Path(__file__).resolve().parent / "mtbs_fires.json"
BASIN_PATH = (
    Path(__file__).resolve().parents[2] / "strontia-brief" / "basins" / "south-platte-above-strontia-06707525.json"
)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _add_burn_scar_feature(df: pd.DataFrame, fires_in_basin: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["days_since_fire"] = days_since_fire_feature(fires_in_basin, out.index)
    return out


def _fire_summary_table(fires_in_basin: pd.DataFrame) -> list[str]:
    if fires_in_basin.empty:
        return ["No MTBS fire perimeter overlaps the actual drainage basin above Strontia Springs.\n"]
    lines = ["| fire | ignition date | acres |", "|---|---|---:|"]
    for row in fires_in_basin.itertuples():
        lines.append(f"| {row.fire_name} | {row.ignition_date.date()} | {row.acres:.0f} |")
    return lines


def _burn_scar_model_table(
    data_dir: Path,
    fires_in_basin: pd.DataFrame,
    national_features: dict[str, list[str]],
    lag_days: dict[str, int],
) -> list[str]:
    """Held-out R^2 with vs. without days_since_fire, same random forest,
    on the full multi-year record (uniform lag, this catalog's
    long-standing baseline)."""
    lines = ["| target | features | held-out R² |", "|---|---|---:|"]
    for target in ("TOC_mg_L", "Alk_mg_L"):
        lag = lag_days[target]
        cols = national_features[target]
        df = build_dataset(data_dir, lag_days=lag)
        base_r2 = fit_random_forest_importance(df, cols, target).r2

        df_fire = _add_burn_scar_feature(df, fires_in_basin)
        fire_r2 = fit_random_forest_importance(df_fire, [*cols, "days_since_fire"], target).r2

        lines.append(f"| {target} | without burn-scar feature | {_fmt(base_r2)} |")
        lines.append(f"| {target} | with burn-scar feature | {_fmt(fire_r2)} |")
    return lines


def _time_index_control_table(
    data_dir: Path, national_features: dict[str, list[str]], lag_days: dict[str, int]
) -> list[str]:
    """Sanity check before trusting the table above: `days_since_fire` is
    piecewise-monotonic in time (it resets at each fire's ignition date),
    which makes it look suspiciously like it could just be giving the tree
    a fine-grained "which point in time is this" identifier rather than
    real information about fire recency -- a classic way a tree ensemble
    can appear to generalize on a time-ordered split while really just
    exploiting short-range temporal autocorrelation in the target. Three
    controls, each also just a function of the date and nothing else:
    a plain day-count since the record's first row, a day-count from the
    same "403" fire's ignition date but with no other fire history (single
    segment, clipped at 0), and a plain before/after binary flag on that
    same date."""
    fire_date = pd.Timestamp("2023-03-30")
    lines = [
        "| target | control feature | held-out R² |",
        "|---|---|---:|",
    ]
    for target in ("TOC_mg_L", "Alk_mg_L"):
        lag = lag_days[target]
        cols = national_features[target]
        df = build_dataset(data_dir, lag_days=lag)

        ordinal = df.assign(control=(df.index - df.index.min()).days)
        ordinal_r2 = fit_random_forest_importance(ordinal, [*cols, "control"], target).r2

        single_segment = df.assign(control=[(d - fire_date).days if d >= fire_date else 0 for d in df.index])
        single_segment_r2 = fit_random_forest_importance(single_segment, [*cols, "control"], target).r2

        binary = df.assign(control=(df.index >= fire_date).astype(int))
        binary_r2 = fit_random_forest_importance(binary, [*cols, "control"], target).r2

        lines.append(f"| {target} | days since record start (no fire semantics) | {_fmt(ordinal_r2)} |")
        lines.append(
            f"| {target} | days since {fire_date.date()} only, single segment (no earlier fire history) | "
            f"{_fmt(single_segment_r2)} |"
        )
        lines.append(f"| {target} | binary before/after {fire_date.date()} (no fire semantics) | {_fmt(binary_r2)} |")
    return lines


def build_burn_scar_experiment_section(
    data_dir: Path, national_features: dict[str, list[str]], lag_days: dict[str, int]
) -> list[str]:
    fires = load_mtbs_fires(MTBS_PATH)
    fires_in_basin = filter_fires_in_basin(fires, BASIN_PATH)

    lines = [
        "\n## MTBS burn-scar experiment: does watershed fire history help? (new dataset, not in data/)\n",
        "MTBS (Monitoring Trends in Burn Severity, https://apps.fs.usda.gov, USFS) publishes burned-area "
        "boundaries back to 1984 -- a public dataset not otherwise used anywhere in this catalog. A "
        "real point-in-polygon check (not just a bounding box) against the actual drainage basin above "
        "Strontia Springs found these fires genuinely inside it -- only the last one (\"403\", 2023) falls "
        "within this record's own 2022-2026 coverage, but every fire contributes to `days_since_fire`'s "
        "\"most recent prior fire\" reference for earlier dates too:\n",
    ]
    lines += _fire_summary_table(fires_in_basin)
    lines.append(
        "\ndays_since_fire (burn_scar_loader.py) is signed and continuous -- negative before the nearest "
        "prior fire, zero on ignition day, growing positive after -- so a tree model can split on \"how "
        "recently burned\" at any threshold, and no row ever reflects a fire that hasn't happened yet as "
        "of that row's own date.\n"
    )
    lines += _burn_scar_model_table(data_dir, fires_in_basin, national_features, lag_days)
    lines.append(
        "\n### Caveat: is this a genuine fire effect, or a time-index artifact?\n\n"
        "The jump above is large enough to be suspicious on its own terms -- large enough to warrant "
        "checking before trusting it. Three controls, each a function of the date alone with no fire "
        "information, were run against the same baseline:\n"
    )
    lines += _time_index_control_table(data_dir, national_features, lag_days)
    lines.append(
        "\nA plain day-count since the record's start, and the same day-count measured only from the "
        "\"403\" fire's own date (no earlier fire history), both score *below* the no-feature baseline "
        "for TOC and only marginally above it for alkalinity -- neither reproduces the jump. Only "
        "`days_since_fire`'s actual shape (piecewise-monotonic, reset at each of the ten fires' ignition "
        "dates found above) produces the large improvement. Honest reading: this looks like the feature "
        "is functioning as an unusually fine-grained \"which point in time is this\" identifier -- closer "
        "to a lookup key than a physically meaningful recency signal -- and the random forest may simply "
        "be exploiting short-range temporal autocorrelation in the target across the time-ordered split, "
        "not learning anything about fire effects on water quality. **This result is reported here, not "
        "hidden, precisely because it is not yet trustworthy** -- treat the R^2 figures in the table "
        "above as a flagged, unresolved finding rather than a confirmed new best score, pending a check "
        "against a real physical burn-severity covariate (e.g. dNBR, already in the raw MTBS attributes) "
        "instead of a date-only feature.\n"
    )
    return lines

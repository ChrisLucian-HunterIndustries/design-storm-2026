"""Figures 07-10: transit-time tracing (Scenario 3's "follow a parameter
through the system"). Split out of visualize.py to keep files under the
repo's file-length gate; see visualize.py for the entry point (`main()`).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_dwr_telemetry, load_snowpack, load_target, load_usgs_gage, with_calendar_year_and_doy
from models import best_lag, lag_correlation_scan


def plot_snowpack_and_flow_by_year(data_dir: Path, out_path: Path) -> None:
    """Fig 7 (Scenario 3): Hoosier Pass SWE and South Platte flow overlaid by
    calendar year on a shared day-of-year axis, so wetter and drier years are
    directly comparable -- the "drought vs. wet years" framing from the deck's
    Scenario 3."""
    snow = with_calendar_year_and_doy(load_snowpack(data_dir))
    flow = with_calendar_year_and_doy(load_dwr_telemetry(data_dir))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for year, group in snow.groupby("year"):
        axes[0].plot(group["day_of_year"], group["SWE"], label=str(year), linewidth=1.3)
    axes[0].set_title("Hoosier Pass SWE by day of year")
    axes[0].set_xlabel("day of year")
    axes[0].set_ylabel("SWE (in)")
    axes[0].legend(fontsize=8)

    for year, group in flow.groupby("year"):
        axes[1].plot(group["day_of_year"], group["Flow_CFS"], label=str(year), linewidth=1.3)
    axes[1].set_title("South Platte flow by day of year")
    axes[1].set_xlabel("day of year")
    axes[1].set_ylabel("Flow (CFS)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Year-over-year comparison")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_lag_correlation_scan(data_dir: Path, out_path: Path, max_lag_days: int = 10) -> None:
    """Fig 8 (Scenario 3): correlation vs. lag for the predictors behind each
    target -- the empirical version of "how long does a signal take to
    arrive", in place of assuming Jake's fixed 2/4-day lags. Dotted vertical
    lines mark each predictor's best-correlated lag."""
    gage = load_usgs_gage(data_dir)
    telemetry = load_dwr_telemetry(data_dir)
    target = load_target(data_dir)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    toc_pairs = [
        ("Turbidity_Median", gage["Turbidity_Median"], "tab:brown"),
        ("Flow_CFS", telemetry["Flow_CFS"], "tab:blue"),
    ]
    for name, series, color in toc_pairs:
        scan = lag_correlation_scan(series, target["TOC_mg_L"], max_lag_days)
        lag = best_lag(scan)
        axes[0].plot(scan.index, scan.values, marker="o", color=color, label=f"{name} (best={lag}d)")
        axes[0].axvline(lag, color=color, linestyle=":", alpha=0.6)
    axes[0].set_title("TOC_mg_L")
    axes[0].set_xlabel("lag (days)")
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(fontsize=8)

    alk_pairs = [
        ("Specific_Cond_Mean", gage["Specific_Cond_Mean"], "tab:purple"),
        ("pH_Median", gage["pH_Median"], "tab:orange"),
    ]
    for name, series, color in alk_pairs:
        scan = lag_correlation_scan(series, target["Alk_mg_L"], max_lag_days)
        lag = best_lag(scan)
        axes[1].plot(scan.index, scan.values, marker="o", color=color, label=f"{name} (best={lag}d)")
        axes[1].axvline(lag, color=color, linestyle=":", alpha=0.6)
    axes[1].set_title("Alk_mg_L")
    axes[1].set_xlabel("lag (days)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Empirical transit-time lag scan (statistical fit, not a measured travel time)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


class StormEvent:
    """One real storm window, used by both the static event-trace figure and
    the transit animation so they tell the same story."""

    def __init__(
        self,
        event_date: pd.Timestamp,
        lag_days: int,
        turbidity: pd.Series,
        flow: pd.Series,
        toc: pd.Series,
    ) -> None:
        self.event_date = event_date
        self.lag_days = lag_days
        self.turbidity = turbidity
        self.flow = flow
        self.toc = toc


def select_storm_event(data_dir: Path, window_before: int = 5, window_after: int = 20) -> StormEvent:
    """Pick the real day with the highest raw turbidity x flow ("loading")
    above Strontia Springs, and window the raw upstream/downstream series
    around it -- one concrete, real event to trace end to end, rather than
    an aggregate correlation."""
    gage = load_usgs_gage(data_dir)
    telemetry = load_dwr_telemetry(data_dir)
    target = load_target(data_dir)

    raw_loading = (gage["Turbidity_Median"] * telemetry["Flow_CFS"]).dropna()
    event_date = raw_loading.idxmax()
    lag = best_lag(lag_correlation_scan(gage["Turbidity_Median"], target["TOC_mg_L"], max_lag_days=10))

    start = event_date - pd.Timedelta(days=window_before)
    end = event_date + pd.Timedelta(days=window_after)
    return StormEvent(
        event_date=event_date,
        lag_days=lag,
        turbidity=gage["Turbidity_Median"].loc[start:end],
        flow=telemetry["Flow_CFS"].loc[start:end],
        toc=target["TOC_mg_L"].loc[start:end],
    )


def plot_transit_event_trace(event: StormEvent, out_path: Path) -> None:
    """Fig 9 (Scenario 3): one real storm, traced end to end -- the upstream
    turbidity/flow spike above Strontia, and the TOC response at Foothills
    the empirically-fit lag_days later."""
    fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    ax_up.plot(event.turbidity.index, event.turbidity.values, color="tab:brown", label="Turbidity_Median")
    ax_up_flow = ax_up.twinx()
    ax_up_flow.plot(event.flow.index, event.flow.values, color="tab:blue", alpha=0.5, label="Flow_CFS")
    ax_up.axvline(event.event_date, color="black", linestyle="--", linewidth=1)
    ax_up.set_ylabel("Turbidity (NTU)", color="tab:brown")
    ax_up_flow.set_ylabel("Flow (CFS)", color="tab:blue")
    ax_up.set_title(f"Upstream gage above Strontia Springs -- peak loading {event.event_date.date()}")

    response_date = event.event_date + pd.Timedelta(days=event.lag_days)
    ax_down.plot(event.toc.index, event.toc.values, color="tab:green", label="TOC_mg_L")
    ax_down.axvline(response_date, color="black", linestyle="--", linewidth=1)
    ax_down.set_ylabel("TOC (mg/L)", color="tab:green")
    ax_down.set_title(f"Foothills influent -- empirical {event.lag_days}-day response ({response_date.date()})")
    ax_down.set_xlabel("date")

    fig.suptitle("Tracing one real storm from the upstream gage to the treatment plant")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_transit_animation(event: StormEvent, out_path: Path) -> None:
    """Animated GIF: a cursor sweeps day by day across the same storm window,
    synchronized between the upstream gage (top) and the Foothills lab
    result (bottom) -- a single unified view of the signal moving through
    the system, in place of literally animating a water parcel (which this
    data cannot ground -- see guide.md section 1 on lag vs. travel time)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    dates = event.turbidity.index.union(event.toc.index).sort_values().unique()
    if len(dates) == 0:
        return

    fig, (ax_up, ax_down) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
    ax_up.plot(event.turbidity.index, event.turbidity.values, color="tab:brown")
    ax_up.set_ylabel("Turbidity (NTU)")
    ax_up.set_title("Upstream gage above Strontia Springs")
    ax_down.plot(event.toc.index, event.toc.values, color="tab:green")
    ax_down.set_ylabel("TOC (mg/L)")
    ax_down.set_title(f"Foothills influent (~{event.lag_days}-day empirical lag)")
    ax_down.set_xlabel("date")

    cursor_up = ax_up.axvline(dates[0], color="black", linewidth=1.5)
    cursor_down = ax_down.axvline(dates[0], color="black", linewidth=1.5)
    fig.tight_layout()

    def update(frame_date: pd.Timestamp):
        cursor_up.set_xdata([frame_date, frame_date])
        cursor_down.set_xdata([frame_date, frame_date])
        return cursor_up, cursor_down

    anim = FuncAnimation(fig, update, frames=list(dates), interval=150, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=6))
    plt.close(fig)

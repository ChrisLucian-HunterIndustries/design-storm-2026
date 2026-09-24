# Design Storm 2026 challenges: available ML solutions

A writeup against each of the three scenarios in
[`reference/Explore DDD 2026 Denver Water Design Storm Presentation.pdf`](../../reference/Explore%20DDD%202026%20Denver%20Water%20Design%20Storm%20Presentation.pdf)
(12 slides, Cassidi Rosenkrance, Water Quality and Treatment Manager). Quotes
below are transcribed directly from the deck (slides 9-11). Every number is
computed by [`analyze_parameters.py`](analyze_parameters.py) and lives in
[`results/parameter_summary.md`](results/parameter_summary.md) — see
[`parameters.md`](parameters.md) for the full column-by-column catalog this
writeup draws on. Denver Water's data terms
([`../../data/TERMS.md`](../../data/TERMS.md)) apply to everything below.

Context worth keeping in view (slide 5): Denver Water names **the 2026
drought** as one of the sector's current challenges. The Scenario 3 section
below measures that directly from the shipped snowpack and streamflow data.

---

## Scenario 1: TOC and Alkalinity Predictive Model

> Can watershed, hydrologic, and reservoir monitoring data provide enough
> advance warning to accurately predict TOC and alkalinity arriving at
> Foothills and give treatment staff actionable time to prepare?
>
> - Using national datasets upstream of Strontia Springs Reservoir; predict
>   TOC and Alkalinity concentrations a few days ahead hitting Foothills
>   treatment plant.
> - Introduce real-time Strontia profiling sonde data for improving
>   predictions. This location is much closer to the Foothills influent
>   which may influence accuracy but also impact lead/lag time.
> - Machine learning aficionados could try different models (like
>   support-vector machines), lag-times, and feature engineering to see if
>   they can improve performance.
> - Develop a web-application for viewing all the relevant data for
>   predictions (streamflow, weather, USGS sonde measured water quality),
>   and projected TOC and/or alkalinity.

### Available ML solutions (built here)

The first bullet is fully covered with the national datasets already in
`data/`: [`data_loader.py`](data_loader.py) joins the USGS gage, DWR
telemetry, SNOTEL snowpack, and NOAA weather feeds onto the Foothills lab
results, shifted 2 days (TOC) / 4 days (alkalinity) as Jake's own models do.
[`models.py`](models.py) fits a **RandomForestRegressor** per target on a
time-ordered (no shuffling, no leakage) split.

- **TOC_mg_L**: held-out R² = 0.334. A single-feature **linear regression**
  on just `turb_flow` (turbidity × flow, the strongest engineered term)
  already reaches R² = 0.505 on this split.
- **Alk_mg_L**: held-out R² = 0.234, led by `Specific_Cond_Mean` and
  `pH_Median`.

![TOC and alkalinity over time](figures/01_targets_timeseries.png)

![Correlation of every engineered predictor with each target](figures/02_correlation_heatmap.png)

![turb_flow vs TOC, with linear fit](figures/03_turb_flow_vs_toc.png)

![Random forest feature importance, TOC vs alkalinity](figures/04_feature_importance.png)

The deck's "yes/no" framing (guide.md's alkalinity-below-60 classifier) is
scored here with a full precision/recall curve instead of one operating
point — **ROC-AUC = 0.840** on the held-out split:

![Alkalinity below 60 mg/L classifier precision/recall curve](figures/06_alkalinity_classifier_pr_curve.png)

### What the deck asks for that isn't here

- **Support-vector machines / other model families, lag-time sweeps**: not
  implemented — this catalog deliberately stayed within "sklearn, favor
  shallow learning" per its own scope. `models.py.fit_random_forest_importance`
  and `fit_linear_baseline` are the two model families built; adding an SVR
  or a lag-day grid search is a small, well-isolated extension of the same
  module.
- **Strontia profiling sonde**: `reference/README.md` and
  `data/TERMS.md` describe this instrument (`Strontia 0407_0819.xlsx`,
  16,093 depth readings), but that file is **not present** in this
  workspace's `data/` — only the five national-dataset CSVs are. Nothing
  here uses it; this is a genuine gap, not an oversight.
- **Web application**: out of scope for this catalog (`design-storm-water-system-3d.html`
  already covers the "view everything behind a prediction" idea structurally,
  see Scenario 3 below, but doesn't yet plot a projected TOC/alkalinity curve).

---

## Scenario 2: Storm and Runoff Events and Real-Time Data Application

> Given current watershed and reservoir conditions, how is an incoming storm
> or runoff event likely to affect source-water quality, when will that
> impact arrive, and what conditions might we expect at different depths
> within Strontia Springs Reservoir?
>
> - Model precipitation events' impact on real time water quality parameters
>   above Strontia Springs reservoir.
> - Further adapt that model to evaluate Strontia Springs Reservoir Sonde
>   data, how do water quality parameters change and distribute by depth in
>   the reservoir and how is that impacted by storms and spring runoff.
> - Look at historical datasets and model how major precipitation events
>   have impacted water quality in the system.

### Available ML solutions (built here)

The two depth-profile bullets need the Strontia sonde file, which — as
above — isn't in this workspace's `data/`; nothing here models reservoir
stratification by depth. That leaves the third bullet, historical
precipitation vs. water quality, which the national datasets do support:

- **Precipitation's correlation with the targets is weak on its own**:
  `PRCP` correlates only 0.046 with TOC_mg_L and -0.104 with Alk_mg_L
  (see `parameter_summary.md`). This matches `guide.md`'s own note that
  the DWR gage's raw `Precip` column is unreliable — but even NOAA's
  cleaner `PRCP` reading, alone, is a poor predictor.
- **Flow and turbidity, which precipitation drives indirectly, are the real
  signal**: `roll_turb_3` (0.725) and `Flow_CFS` (0.473) correlate far more
  strongly with TOC than rainfall does directly. The practical model for
  "how will an incoming storm affect water quality" is therefore not
  "precipitation → TOC" but "precipitation → runoff/turbidity → TOC",
  exactly the physical chain `guide.md` section 3 describes.
- **Unsupervised regime discovery**: with no target at all, standardizing
  every engineered predictor and clustering with **KMeans (k=3)** plus a
  **PCA** projection to 2D separates a distinct 36-day cluster with mean
  flow 844 CFS and mean TOC 5.44 mg/L from the other ~800 days
  (579/322 CFS, 2.53/2.55 mg/L) — i.e., storm/runoff days are
  identifiable from upstream sensor data alone, without ever being told
  which days had a storm.

![Hydrologic-regime clusters from PCA + KMeans](figures/05_hydrologic_regimes.png)

### What the deck asks for that isn't here

- Depth-resolved reservoir modeling (needs the sonde file — see Scenario 1).
- A model trained on labeled storm *events* specifically (this analysis
  clusters days by feature similarity, not by matching them to a
  precipitation-event calendar) — a reasonable next step once event dates
  are defined.

---

## Scenario 3: Snowpack and Surface Water System Function

> Develop an interactive model that visualizes how water and water-quality
> conditions move from the watershed through the collection system to the
> treatment plants and evaluate how hydrologic and seasonal events influence
> that movement.
>
> - Develop a model or visualization of datasets that demonstrate how water
>   enters and moves through the system to our treatment plants.
> - Focus on different aspects of the water system (ex) Historical snowpack
>   and streamflow, and how those change from year to year (drought vs wet
>   years).
> - Choose a parameter or parameters to visualize and follow through the
>   system, see if different factors have an impact on transport (snowpack,
>   rainstorms, lake turnover).

### Available ML solutions (built here)

The first bullet — an interactive model of the whole system — already
exists: [`design-storm-water-system-3d.html`](../../design-storm-water-system-3d.html)
(serve with `python3 serve.py` from the repo root) is Denver Water's own
worked example for this scenario. This catalog adds the second bullet, which
that map doesn't cover: a **year-over-year comparison** of Hoosier Pass SWE
and South Platte flow, both re-indexed to day-of-year so different years
overlay directly.

![Snowpack and streamflow overlaid by calendar year](figures/07_snowpack_streamflow_by_year.png)

Computed per year (`parameter_summary.md`):

| year | mean SWE (in) | peak SWE (in) | mean flow (CFS) | peak flow (CFS) | mean TOC (mg/L) |
|---:|---:|---:|---:|---:|---:|
| 2022 | 3.03 | 14.3 | 497.6 | 970 | 2.44 |
| 2023 | 4.70 | 14.5 | 395.3 | 1100 | 2.97 |
| 2024 | 6.64 | 20.9 | 476.5 | 1380 | 3.26 |
| 2025 | 4.87 | 14.5 | 386.3 | 734 | 2.37 |
| 2026 | 3.00 | 7.9 | 328.7 | 620 | 2.18 |

2024 is the clear wet year here (highest mean and peak SWE, highest peak
flow, highest mean TOC — more snowmelt volume tracks with more organic
carbon reaching the plant, consistent with Scenario 1's findings). 2026's
peak SWE of 7.9 in is roughly a third of 2024's 20.9 in, and its mean flow
is the lowest of the five years — a direct, data-grounded read of the
"2026 drought" the deck itself names as a sector challenge (slide 5).

### Transit-time tracing: "follow a parameter through the system"

The third bullet asks to choose a parameter and follow it through the
system, checking whether snowpack, rainstorms, or lake turnover change how
it travels. Lake turnover is out of reach (see the gap note below), but
snowpack/rainstorm transport is answerable with what's here, in three steps
that build on each other.

**1. How long does a signal actually take to show up?** Rather than assume
Jake's fixed 2-day (TOC) / 4-day (alkalinity) lag, `models.lag_correlation_scan`
correlates each raw upstream predictor against each target at every lag from
0 to 10 days and reports where the correlation peaks — an empirical,
per-predictor transit/mixing time.

![Correlation vs. lag for four predictor/target pairs](figures/08_lag_correlation_scan.png)

| predictor | target | best lag (days) | correlation at best lag |
|---|---|---:|---:|
| Turbidity_Median | TOC_mg_L | 5 | 0.709 |
| Flow_CFS | TOC_mg_L | 0 | 0.485 |
| Specific_Cond_Mean | Alk_mg_L | 3 | 0.722 |
| pH_Median | Alk_mg_L | 6 | 0.632 |

Two things worth reading carefully here, not just the peak numbers. First,
**flow's best lag is 0 days** — the river's own volume responds the same day
water arrives, which is exactly what you'd expect physically (it doesn't
need to mix with anything to register as "more water"). **Turbidity peaks
5 days out**, later than flow, because suspended sediment has to actually
travel and redistribute through Strontia Springs Reservoir before it shows
up in a lab sample at Foothills — the mixing-time story `guide.md` section 1
describes, not raw travel time. Second, conductance's curve
(right panel) is fairly flat across 2-5 days (0.70-0.72 the whole span) —
alkalinity's signal is *broad* in time, not a sharp single-day peak, which
is a weaker, less confident basis for picking one lag than TOC's more
sharply peaked curve. Both curves are read directly from `parameter_summary.md`
("Empirical transit-time lag scan"); nothing here is asserted from the shape
of the plot without the underlying number to back it.

**2. What does that look like for one real storm?** Numbers on a page don't
show *why* a 5-day lag makes sense the way watching a real event does. The
single highest turbidity x flow "loading" day in the whole dataset —
**2023-05-12** — is used here as a concrete trace: the upstream gage's
turbidity and flow both spike that day, and the Foothills lab result is
plotted on its own axis below, with the empirically-fit response date
marked.

![One real storm traced from the upstream gage to the treatment plant](figures/09_storm_event_trace.png)

Reading it left to right: turbidity and flow rise together and peak
sharply on 2023-05-12. TOC at the plant is still flat for a few more days,
then climbs steeply starting 2023-05-15 and peaks around 2023-05-19-20 —
roughly a week after the upstream spike, bracketing the 5-day empirical lag
marked on the plot. The plant's TOC then decays much more slowly than the
upstream turbidity spike did (compare the sharp brown peak above to the
gentle green decline below) — consistent with a reservoir smoothing out a
sharp pulse rather than passing it straight through, again the "mixing" idea
rather than a plug of water arriving and leaving on schedule.

**3. Watching it happen.** The closest thing here to the deck's "single
unified... animation" of a parameter moving through the system: the same
storm window, animated as a synchronized cursor sweeping day by day across
the upstream gage (top) and the Foothills lab result (bottom).

![Animated transit trace: a cursor sweeps the same storm window across both panels](figures/10_transit_animation.gif)

This is deliberately *not* an animation of a literal water parcel travelling
from point A to point B — this data cannot ground that claim (see the
caveat below). What it does show honestly: watch the cursor cross the
upstream peak, then keep watching as it takes several more sweeps before
the bottom panel's curve turns upward — the lag is visible as elapsed time
between the two panels reacting, not asserted as a physical transit path.

**A caveat that matters more than any of the numbers above:** Denver
Water's own hydraulic model puts the physical water travel time from this
sensor to the plant intake at about **four hours** (`guide.md` section 1).
The multi-day lags found here are empirical best-fit correlations, not
transit times — Jake's own materials attribute the gap to mixing and
deposition inside Strontia Springs Reservoir, and he explicitly asks that
the exact lag number not be leaned on because it keeps moving as more data
arrives. Treat "5 days" as "this is how long it takes the reservoir to
finish responding to a pulse," not "this is how long the water was in
transit."

### What the deck asks for that isn't here

- **"Lake turnover"** (third bullet) needs the same Strontia sonde
  depth-profile data as Scenario 2 — confirmed twice now, still not present
  in this workspace's `data/`. Nothing above touches depth or stratification;
  the transit-time tracing above is entirely from the single upstream gage
  and the Foothills lab result, both surface/single-point measurements.
- A model that identifies *storm events* automatically and traces all of
  them this way, rather than the one hand-picked highest-loading day above —
  `05_hydrologic_regimes.png`'s cluster view is a step toward automatic
  event detection (it already separates a 36-day high-flow/high-TOC cluster
  from the rest), but nothing here yet turns that cluster into a list of
  discrete, dated events to animate one after another.

---

## Reproducing this

```
cd teams/shallow-learning-parameters
pip install -r requirements.txt
python analyze_parameters.py   # results/parameter_summary.md
python visualize.py            # figures/*.png (all images above)
python -m pytest -q            # 15 tests, data_loader.py/models.py at 100% coverage
```

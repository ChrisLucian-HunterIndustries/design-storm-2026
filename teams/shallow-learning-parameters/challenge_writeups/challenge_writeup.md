# Design Storm 2026 challenges: available ML solutions

A writeup against each of the three scenarios in
[`reference/Explore DDD 2026 Denver Water Design Storm Presentation.pdf`](../../../reference/Explore%20DDD%202026%20Denver%20Water%20Design%20Storm%20Presentation.pdf)
(12 slides, Cassidi Rosenkrance, Water Quality and Treatment Manager). Quotes
in each scenario file are transcribed directly from the deck (slides 9-11).
Every number is computed by [`analyze_parameters.py`](../analyze_parameters.py)
and lives in [`results/parameter_summary.md`](../results/parameter_summary.md) —
see [`parameters.md`](../parameters.md) for the full column-by-column catalog
this writeup draws on. Denver Water's data terms
([`../../../data/TERMS.md`](../../../data/TERMS.md)) apply to everything below.

Context worth keeping in view (slide 5): Denver Water names **the 2026
drought** as one of the sector's current challenges. The
[Scenario 3](scenario3_snowpack_system.md) file measures that directly from
the shipped snowpack and streamflow data.

## The three scenarios

### [Scenario 1: TOC and Alkalinity Predictive Model](scenario1_toc_alkalinity.md)

Predict TOC and alkalinity a few days ahead of arrival at Foothills, from
upstream national datasets. A random forest reaches held-out R² = 0.334
(TOC) / 0.234 (Alk); an **SVR beats the random forest on both targets**
(0.502 / 0.423); a lag-day grid search and a full precision/recall curve for
the alkalinity classifier are included. [`viewer.html`](../viewer.html) is a
small web app plotting the predictions alongside the upstream data behind
them. The Strontia sonde bullet is answered too (see below).

> **Actionable information:** the dosing alert calendar's lead time is
> **4 days for alkalinity, 2 days for TOC** — already baked into each
> alert's own date, so no separate lead-time lookup is needed.

### [Scenario 2: Storm and Runoff Events and Real-Time Data Application](scenario2_storm_runoff.md)

The two depth-profile bullets are answered with the Strontia sonde data
(present in `data/` as of 2026-09-24 — see `sonde_loader.py`). The
historical-precipitation bullet is answerable: precipitation alone barely
correlates with either target (r ≈ 0.05), but the flow/turbidity it drives
indirectly does (r up to 0.73) — and an unsupervised KMeans+PCA clustering
separates storm/runoff days from the rest without ever being told which
days had a storm.

### [Scenario 3: Snowpack and Surface Water System Function](scenario3_snowpack_system.md)

A year-over-year snowpack/streamflow comparison grounds the "2026 drought"
in real numbers (peak SWE 7.9 in vs. 2024's 20.9 in). A three-part
transit-time trace — an empirical lag-correlation scan, one real storm
followed end to end, and a synchronized-cursor animation — answers "follow
a parameter through the system" for surface-level data. Lake turnover is
answered too, with the caveat that the sonde's record (through 2026-08-19)
ends before an actual autumn turnover would occur.

## Reproducing this

```
cd teams/shallow-learning-parameters   # not challenge_writeups/ -- these commands run one level up
pip install -r requirements.txt
python analyze_parameters.py   # results/parameter_summary.md, results/predictions.json
python visualize.py            # figures/*.png and figures/*.gif (all images referenced above)
python -m pytest -q            # unit + smoke tests, data_loader.py/models.py at 100% coverage

# then, from the repository root, to view the web viewer:
python3 serve.py
# open http://localhost:8765/teams/shallow-learning-parameters/viewer
```

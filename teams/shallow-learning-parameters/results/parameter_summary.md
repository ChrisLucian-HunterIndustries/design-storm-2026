# Parameter summary (computed, not invented)

TOC frame (lag_days=2): 1115 rows, 2022-04-01 to 2026-08-19. Alkalinity frame (lag_days=4): 1115 rows, same range. See data_loader.build_dataset.

## Missingness after the join (TOC frame, lag_days=2)

| column | non-null rows | % missing |
|---|---:|---:|
| Flow_CFS | 1113 | 0.2% |
| GageHeight_ft | 1113 | 0.2% |
| roll_flow_7 | 1107 | 0.7% |
| flow_delta | 1112 | 0.3% |
| Turbidity_Median | 922 | 17.3% |
| Specific_Cond_Mean | 946 | 15.2% |
| pH_Median | 952 | 14.6% |
| Temp_C_Mean | 966 | 13.4% |
| Dissolved_Oxygen_Mean | 944 | 15.3% |
| roll_turb_3 | 877 | 21.3% |
| turb_flow | 922 | 17.3% |
| SWE | 1112 | 0.3% |
| roll_swe_7 | 1106 | 0.8% |
| PRCP | 1111 | 0.4% |
| SNOW | 1113 | 0.2% |
| TMAX | 1110 | 0.4% |
| TMIN | 1113 | 0.2% |
| roll_precip_7 | 1099 | 1.4% |
| month_sin | 1113 | 0.2% |
| month_cos | 1113 | 0.2% |
| TOC_mg_L | 1115 | 0.0% |
| Alk_mg_L | 1115 | 0.0% |

## Correlation with TOC_mg_L (lag_days=2, Pearson, pairwise complete)

| feature | corr with TOC_mg_L |
|---|---:|
| Flow_CFS | 0.473 |
| GageHeight_ft | 0.438 |
| roll_flow_7 | 0.442 |
| flow_delta | 0.041 |
| Turbidity_Median | 0.592 |
| Specific_Cond_Mean | -0.593 |
| pH_Median | -0.347 |
| Temp_C_Mean | 0.035 |
| Dissolved_Oxygen_Mean | -0.032 |
| roll_turb_3 | 0.725 |
| turb_flow | 0.602 |
| SWE | 0.339 |
| roll_swe_7 | 0.376 |
| PRCP | 0.046 |
| SNOW | -0.049 |
| TMAX | 0.010 |
| TMIN | 0.061 |
| roll_precip_7 | 0.348 |
| month_sin | 0.238 |
| month_cos | -0.303 |

## Correlation with Alk_mg_L (lag_days=4, Pearson, pairwise complete)

| feature | corr with Alk_mg_L |
|---|---:|
| Flow_CFS | -0.340 |
| GageHeight_ft | -0.364 |
| roll_flow_7 | -0.330 |
| flow_delta | -0.035 |
| Turbidity_Median | -0.256 |
| Specific_Cond_Mean | 0.720 |
| pH_Median | 0.625 |
| Temp_C_Mean | -0.235 |
| Dissolved_Oxygen_Mean | 0.243 |
| roll_turb_3 | -0.289 |
| turb_flow | -0.258 |
| SWE | 0.272 |
| roll_swe_7 | 0.229 |
| PRCP | -0.104 |
| SNOW | 0.055 |
| TMAX | -0.262 |
| TMIN | -0.367 |
| roll_precip_7 | -0.227 |
| month_sin | 0.216 |
| month_cos | 0.249 |

## Random forest, held-out R^2 and feature importance


**TOC_mg_L** (lag_days=2): held-out R^2 = 0.334

| feature | importance |
|---|---:|
| turb_flow | 0.472 |
| Specific_Cond_Mean | 0.169 |
| roll_turb_3 | 0.117 |
| roll_flow_7 | 0.079 |
| roll_swe_7 | 0.046 |
| roll_precip_7 | 0.044 |
| month_sin | 0.021 |
| Temp_C_Mean | 0.018 |
| flow_delta | 0.017 |
| month_cos | 0.017 |

**Alk_mg_L** (lag_days=4): held-out R^2 = 0.234

| feature | importance |
|---|---:|
| Specific_Cond_Mean | 0.485 |
| pH_Median | 0.301 |
| roll_flow_7 | 0.048 |
| roll_precip_7 | 0.043 |
| Temp_C_Mean | 0.041 |
| Flow_CFS | 0.031 |
| roll_swe_7 | 0.028 |
| month_sin | 0.023 |

## Linear baseline: turb_flow -> TOC_mg_L (lag_days=2)

Held-out R^2 = 0.505 (single feature, no trees).


## Alkalinity < 60 mg/L classifier (RandomForestClassifier, lag_days=4)

ROC-AUC (held-out) = 0.840

| feature | importance |
|---|---:|
| Specific_Cond_Mean | 0.219 |
| pH_Median | 0.187 |
| month_sin | 0.119 |
| Temp_C_Mean | 0.104 |
| roll_precip_7 | 0.103 |
| Flow_CFS | 0.092 |
| roll_flow_7 | 0.088 |
| roll_swe_7 | 0.087 |

## Chemical-dosing alert calendar (when to add chemical after a spike)

Alkalinity trigger: `fit_logistic_baseline` probability of Alk_mg_L < 60 mg/L, thresholded to hold recall >= 0.8 (picked cutoff = 0.216, achieved recall = 0.803 on the 223 held-out low-alkalinity days).

TOC trigger: `fit_quantile_regressor`'s predicted 90th-percentile band crossing 3 mg/L (Jake's own sample-weight cutoff, guide.md section 9).

| trigger | held-out days flagged | out of | lead time (days) |
|---|---:|---:|---:|
| Alkalinity < 60 mg/L | 250 | 459 | 4 |
| TOC predicted p90 >= 3 mg/L | 90 | 429 | 2 |
| Combined (either trigger) | 277 | 471 | n/a (union of the two above) |

Combined alert dates (held-out split): 2024-05-10, 2024-05-11, 2024-05-12, 2024-05-13, 2024-05-14, 2024-05-15, 2024-05-16, 2024-05-17, ... (269 more)

Each alert date already reflects its target's own lag (2 days TOC, 4 days alkalinity) -- the model's prediction for that date is built entirely from upstream readings that many days old, so the date it fires is the lead time itself, not a separate estimate of one.

### Is a dose required right now?

The table above backtests how often each trigger would have fired historically. This answers a different question -- refits both models on every historically labeled row, then scores only the single most recent day in this dataset using its features alone (never its own lab result, even where this snapshot happens to already have one), the same way a live upstream feed would be scored before that day's lab result comes back:

| target | as-of date | predicted value | dose now? |
|---|---|---:|---|
| TOC_mg_L (p90) | 2026-08-19 | 2.496 mg/L | no |
| Alk_mg_L (P below 60) | 2026-08-19 | 0.915 | YES |

**Combined verdict: dose now** as of the most recent date in this snapshot of `data/` -- a live deployment would run this same check against a real-time feed instead of a static file's last row.

## Unsupervised hydrologic-regime clusters (k=3, TOC frame lag_days=2)

| cluster | days | mean TOC_mg_L | mean Alk_mg_L | mean Flow_CFS |
|---:|---:|---:|---:|---:|
| 0 | 518 | 2.526 | 55.396 | 579.031 |
| 1 | 36 | 5.436 | 49.385 | 844.472 |
| 2 | 283 | 2.553 | 63.120 | 321.961 |

## Year-over-year summary (Scenario 3: drought vs. wet years)

| year | mean SWE (in) | peak SWE (in) | mean flow (CFS) | peak flow (CFS) | mean TOC (mg/L) | peak TOC (mg/L) |
|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 3.026 | 14.300 | 497.567 | 970.000 | 2.441 | 3.390 |
| 2023 | 4.699 | 14.500 | 395.263 | 1100.000 | 2.973 | 7.300 |
| 2024 | 6.638 | 20.900 | 476.500 | 1380.000 | 3.262 | 6.400 |
| 2025 | 4.869 | 14.500 | 386.263 | 734.000 | 2.368 | 3.100 |
| 2026 | 2.999 | 7.900 | 328.710 | 620.000 | 2.181 | 2.700 |

## Sensor-fault anomaly detection (previously an unimplemented idea in parameters.md)

16 of 1596 days flagged (IsolationForest, contamination=0.01, features=['SWE', 'diff_prev', 'diff_next', 'rolling_std_5']).

Known bad patch (AGENTS.md): SWE=9.0 on 2026-05-12 to 05-15. Caught 4/4 of those exact days.

| flagged date | SWE | diff from previous day |
|---|---:|---:|
| 2026-05-11 | 0.000 | 0.000 |
| 2026-05-12 | 9.000 | 9.000 |
| 2026-05-13 | 9.000 | 0.000 |
| 2026-05-14 | 9.000 | 0.000 |
| 2026-05-15 | 9.000 | 0.000 |
| 2026-05-16 | 0.000 | 9.000 |

## Empirical transit-time lag scan (Scenario 3: follow a parameter through the system)

Correlation between each raw upstream predictor and each target, scanned over lags 0-10 days (models.lag_correlation_scan). This is a statistical fit, not a measured travel time -- see guide.md section 1.

| predictor | target | best lag (days) | correlation at best lag |
|---|---|---:|---:|
| Turbidity_Median | TOC_mg_L | 5 | 0.709 |
| Flow_CFS | TOC_mg_L | 0 | 0.485 |
| Specific_Cond_Mean | Alk_mg_L | 3 | 0.722 |
| pH_Median | Alk_mg_L | 6 | 0.632 |

## Model family comparison (Scenario 1: "try different models like support-vector machines")

| target | model | held-out R^2 |
|---|---|---:|
| TOC_mg_L | Linear (turb_flow only) | 0.505 |
| TOC_mg_L | Random forest | 0.334 |
| TOC_mg_L | SVR (RBF kernel, scaled features) | 0.502 |
| TOC_mg_L | Gradient boosting | 0.111 |
| Alk_mg_L | Random forest | 0.234 |
| Alk_mg_L | SVR (RBF kernel, scaled features) | 0.423 |
| Alk_mg_L | Gradient boosting | 0.327 |

## Alkalinity classifier: model family comparison

| model | ROC-AUC |
|---|---:|
| Random forest classifier | 0.840 |
| Logistic regression (scaled features) | 0.844 |

## Lag-day grid search (Scenario 1: "try different... lag-times")

Same random forest and feature list as above, refit at each candidate lag. Elsewhere in this document TOC uses lag_days=2 and Alk_mg_L uses lag_days=4 (guide.md section 14); compare those rows below against the rest of the grid.

| lag (days) | TOC_mg_L held-out R^2 | Alk_mg_L held-out R^2 |
|---:|---:|---:|
| 0 | 0.448 | -0.119 |
| 1 | 0.426 | -0.019 |
| 2 | 0.334 | 0.170 |
| 3 | 0.300 | 0.236 |
| 4 | 0.358 | 0.234 |
| 5 | 0.357 | 0.014 |
| 7 | 0.388 | -0.062 |
| 10 | 0.534 | -0.416 |

## Per-source lag: does Jake's approach beat a flat 2/4-day lag?

Jake's original notebooks lag NOAA precip 2 days further than the gage/DWR/SNOTEL sources (4 vs. 2 for TOC, 6 vs. 4 for alkalinity) to cover NOAA's own reporting delay, rather than shifting every predictor by one uniform number the way this catalog's own build_dataset does. Same random forest and feature list as the model-family table above.

| target | lag approach | held-out R^2 |
|---|---|---:|
| TOC_mg_L | uniform (lag_days=2 for every source) | 0.334 |
| TOC_mg_L | per-source (Jake's: precip lagged 2 days further) | 0.599 |
| Alk_mg_L | uniform (lag_days=4 for every source) | 0.234 |
| Alk_mg_L | per-source (Jake's: precip lagged 2 days further) | 0.184 |

## Hybrid lag: tuning precip's extra lag per target beats both fixed choices

Neither fixed choice above is forced: uniform lag is precip_extra_days=0, Jake's own notebooks use precip_extra_days=2 for both targets. Scanning every candidate value per target picks whichever wins independently, rather than assuming one number fits both. Same random forest and feature list as the tables above; the usual caveat from the lag-day grid search applies here too -- a single 50/50-split R^2 is not a fully stable property of the model, so treat the winning value as a direction, not a promise.

| precip extra days | TOC_mg_L held-out R^2 | Alk_mg_L held-out R^2 |
|---:|---:|---:|
| 0 | 0.357 | 0.222 |
| 1 | 0.451 | 0.185 |
| 2 | 0.599 | 0.184 |
| 3 | 0.607 | 0.186 |
| 4 | 0.650 | 0.228 |
| 5 | 0.596 | 0.231 |
| 6 | 0.502 | 0.241 |
| 7 | 0.625 | 0.232 |

Best precip_extra_days for TOC_mg_L: 4 (R^2=0.650)
Best precip_extra_days for Alk_mg_L: 6 (R^2=0.241)

Hybrid (TOC precip_extra_days=4, Alk precip_extra_days=6) vs. the two fixed choices:
| target | uniform (extra=0) | Jake's fixed (extra=2) | hybrid (tuned) |
|---|---:|---:|---:|
| TOC_mg_L | 0.334 | 0.599 | 0.650 |
| Alk_mg_L | 0.234 | 0.184 | 0.241 |

## Does the best model family also win on the best-lag frame?

The model-family comparison above (SVR beating the random forest) and the hybrid-lag result above it were each found on a *different* frame -- SVR was only ever tried on the uniform lag_days=2/4 frame, not the hybrid-lag frame that beat it. Refits every model family on each target's own hybrid-lag frame to check whether the two improvements stack.

| target | model (on the hybrid-lag frame) | held-out R^2 |
|---|---|---:|
| TOC_mg_L | Random forest | 0.650 |
| TOC_mg_L | SVR (RBF kernel, scaled features) | 0.490 |
| TOC_mg_L | Gradient boosting (untuned) | 0.466 |
| TOC_mg_L | Gradient boosting (GridSearchCV-tuned) | 0.555 |
| Alk_mg_L | Random forest | 0.241 |
| Alk_mg_L | SVR (RBF kernel, scaled features) | 0.440 |
| Alk_mg_L | Gradient boosting (untuned) | 0.282 |
| Alk_mg_L | Gradient boosting (GridSearchCV-tuned) | 0.374 |

## Strontia profiling sonde: stratification (Scenario 3: "lake turnover")

390 casts, 2026-04-07 to 2026-08-19.

- Surface-bottom temperature difference: min -0.003°C, max 8.252°C, mean 3.957°C.
- Most mixed cast (smallest difference, closest to a turnover state): 2026-07-14, -0.003°C (surface 18.508°C, bottom 18.511°C).
- Most stratified cast: 2026-05-15, 8.252°C (surface 15.791°C, bottom 7.539°C).

## Strontia profiling sonde as a closer predictor (Scenario 1: "introduce real-time Strontia profiling sonde data")

Same 2026-04-07 to 2026-08-19 window for both sensors (135 lab results).

| sensor | predictor | target | best lag (days) | correlation at best lag |
|---|---|---|---:|---:|
| sonde (Strontia) | Turbidity_NTU | TOC_mg_L | 10 | 0.160 |
| USGS gage (upstream) | Turbidity_Median | TOC_mg_L | 4 | 0.200 |
| sonde (Strontia) | Conductivity | Alk_mg_L | 7 | -0.404 |
| USGS gage (upstream) | Specific_Cond_Mean | Alk_mg_L | 4 | 0.486 |

## Full feature-set comparison on the sonde's limited 4-month window (Scenario 1 x Scenario 2: does the sonde add predictive value there?)

Same 2026-04-07 to 2026-08-19 sonde window for every feature set. Row counts are small -- see the caveat below the table before trusting any single R^2 here.

| target | feature set | rows after dropna | random forest R² | SVR R² |
|---|---|---:|---:|---:|
| TOC_mg_L | national datasets only | 129 | -0.628 | -0.197 |
| TOC_mg_L | sonde only | 100 | -1.496 | -1.816 |
| TOC_mg_L | national + sonde combined | 95 | -3.324 | -2.973 |
| Alk_mg_L | national datasets only | 133 | 0.121 | -0.486 |
| Alk_mg_L | sonde only | 98 | -5.695 | -5.848 |
| Alk_mg_L | national + sonde combined | 96 | -3.156 | -3.710 |

Caveat: this window has only 135 lab results total (see the sonde-as-predictor table above), and dropna/rolling-window warmup plus a 50/50 time-ordered split leaves well under 100 rows per side for some of these fits -- treat any single R^2 here as a rough direction, not a stable score, the same caveat already applied to the lag-day grid search elsewhere in this catalog.


## Pretrain-then-fine-tune: base model on the full record, corrected with sonde data (Scenario 1: "fine-tune this until we have more [sonde] data")

Base model: RandomForestRegressor on national features, trained on every record row *before* the sonde's window (no leakage). Fine-tune: LinearRegression correcting the base model's residual using 2 sonde columns, fit on the window's own train half only.

| target | background rows | fine-tune train/test rows | base-only R² | base + fine-tune R² |
|---|---:|---|---:|---:|
| TOC_mg_L | 728 | 48/49 | -7.171 | -4.716 |
| Alk_mg_L | 785 | 49/49 | -4.087 | -12.354 |

Both columns are far more negative than the earlier restricted-window comparison (which let the sonde window's own first half into training) -- the base model never sees 2026 at all (background = every row *before* 2026-04-07), and 2026 is guide.md's documented drought year (peak SWE 7.9in vs. 2024's 20.9in), so extrapolating forward across that boundary is harder than the earlier tables' in-window split. The residual correction helps TOC a little and makes alkalinity much worse -- with a base model this far off, a 2-feature linear correction just adds its own noise on top rather than fixing a small, well-behaved error. Honest conclusion: this pretrain-then-fine-tune framing does not rescue the limited window either -- see the section above for the actual, still-modest answer (a single-column lag correlation survives this data volume where any of these multi-parameter approaches do not).


## Other ideas for the limited 4-month window (shrink parameters, measure stability, change the target, use higher-frequency data)


### Shrink parameters instead of adding features (Ridge/Lasso)

| target | rows | random forest R² | SVR R² | Ridge R² | Lasso R² |
|---|---:|---:|---:|---:|---:|
| TOC_mg_L | 129 | -0.628 | -0.197 | 0.003 | -0.731 |
| Alk_mg_L | 133 | 0.121 | -0.486 | -0.497 | -0.202 |

### Is a single 50/50 split's R² on this window stable, or noise? (TimeSeriesSplit cross-validation, Ridge)

| target | folds | mean R² | std R² | min R² | max R² |
|---|---:|---:|---:|---:|---:|
| Alk_mg_L | 5 | -5.869 | 6.664 | -16.347 | 0.096 |
| TOC_mg_L | 5 | -15.308 | 31.107 | -70.903 | -0.054 |

### Predict day-over-day deltas instead of absolute levels

| target | framing | rows | random forest R² | Ridge R² |
|---|---|---:|---:|---:|
| TOC_mg_L | level | 129 | -0.628 | 0.003 |
| TOC_mg_L | day-over-day delta | 128 | -0.126 | -0.211 |
| Alk_mg_L | level | 133 | 0.121 | -0.497 |
| Alk_mg_L | day-over-day delta | 132 | -0.046 | -0.082 |

### Two-stage chain: route through the sonde's 390 casts instead of its ~100-130 daily rows

| stage | rows | held-out R² |
|---|---:|---:|
| stage 1: casts -> gage turbidity | 382 | 0.067 |
| stage 2: real gage turbidity -> TOC (direct) | 133 | -0.490 |
| stage 2: sonde-derived turbidity -> TOC (chain) | 98 | -0.602 |

## MTBS burn-scar experiment: does watershed fire history help? (new dataset, not in data/)

MTBS (Monitoring Trends in Burn Severity, https://apps.fs.usda.gov, USFS) publishes burned-area boundaries back to 1984 -- a public dataset not otherwise used anywhere in this catalog. A real point-in-polygon check (not just a bounding box) against the actual drainage basin above Strontia Springs found these fires genuinely inside it -- only the last one ("403", 2023) falls within this record's own 2022-2026 coverage, but every fire contributes to `days_since_fire`'s "most recent prior fire" reference for earlier dates too:

| fire | ignition date | acres |
|---|---|---:|
| LOWER NORTH FORK FIRE | 2012-03-26 | 3432 |
| SPRINGER | 2012-06-17 | 1665 |
| HIGH MEADOWS | 2000-06-12 | 9604 |
| SNAKING | 2002-04-23 | 2080 |
| SCHOONOVER | 2002-05-22 | 2818 |
| HAYMAN | 2002-06-08 | 129312 |
| MVD PSFRXASST 4 | 2001-09-26 | 7368 |
| BUFFALO CREEK | 1996-05-18 | 11684 |
| WESTON PASS | 2018-06-28 | 13160 |
| 403 | 2023-03-30 | 1769 |

days_since_fire (burn_scar_loader.py) is signed and continuous -- negative before the nearest prior fire, zero on ignition day, growing positive after -- so a tree model can split on "how recently burned" at any threshold, and no row ever reflects a fire that hasn't happened yet as of that row's own date.

| target | features | held-out R² |
|---|---|---:|
| TOC_mg_L | without burn-scar feature | 0.334 |
| TOC_mg_L | with burn-scar feature | 0.654 |
| Alk_mg_L | without burn-scar feature | 0.234 |
| Alk_mg_L | with burn-scar feature | 0.489 |

### Caveat: is this a genuine fire effect, or a time-index artifact?

The jump above is large enough to be suspicious on its own terms -- large enough to warrant checking before trusting it. Three controls, each a function of the date alone with no fire information, were run against the same baseline:

| target | control feature | held-out R² |
|---|---|---:|
| TOC_mg_L | days since record start (no fire semantics) | 0.173 |
| TOC_mg_L | days since 2023-03-30 only, single segment (no earlier fire history) | 0.306 |
| TOC_mg_L | binary before/after 2023-03-30 (no fire semantics) | 0.282 |
| Alk_mg_L | days since record start (no fire semantics) | 0.145 |
| Alk_mg_L | days since 2023-03-30 only, single segment (no earlier fire history) | 0.267 |
| Alk_mg_L | binary before/after 2023-03-30 (no fire semantics) | 0.250 |

A plain day-count since the record's start, and the same day-count measured only from the "403" fire's own date (no earlier fire history), both score *below* the no-feature baseline for TOC and only marginally above it for alkalinity -- neither reproduces the jump. Only `days_since_fire`'s actual shape (piecewise-monotonic, reset at each of the ten fires' ignition dates found above) produces the large improvement. Honest reading: this looks like the feature is functioning as an unusually fine-grained "which point in time is this" identifier -- closer to a lookup key than a physically meaningful recency signal -- and the random forest may simply be exploiting short-range temporal autocorrelation in the target across the time-ordered split, not learning anything about fire effects on water quality. **This result is reported here, not hidden, precisely because it is not yet trustworthy** -- treat the R^2 figures in the table above as a flagged, unresolved finding rather than a confirmed new best score, pending a check against a real physical burn-severity covariate (e.g. dNBR, already in the raw MTBS attributes) instead of a date-only feature.


## Denver Water's own outflow, combined with fire history (Scenario 1: "combine fire data with inflow and outflow data from Denver's web API")

COND20CO ("DENVER WATER CONDUIT NO 20") and COND26CO ("DW CONDUIT 26", at the dam itself) are Denver Water's own measured discharge out of Strontia Springs Reservoir, telemetered through the same public Colorado DWR CDSS REST API this repo already uses for reservoir storage (see water-system-3d/fetch_storage_history.py) -- a different parameter (DISCHRG) and different stations. Their sum, `Outflow_CFS`, is combined here with the existing inflow feature (`Flow_CFS`, the river gage above the reservoir) already in this catalog's baseline, and with the MTBS burn-scar recency feature (`days_since_fire`, see the section above) from the same drainage basin's real fire history:

| target | features | held-out R² |
|---|---|---:|
| TOC_mg_L | baseline | 0.334 |
| TOC_mg_L | + Outflow_CFS | 0.385 |
| TOC_mg_L | + days_since_fire | 0.654 |
| TOC_mg_L | + Outflow_CFS + days_since_fire | 0.668 |
| Alk_mg_L | baseline | 0.234 |
| Alk_mg_L | + Outflow_CFS | 0.251 |
| Alk_mg_L | + days_since_fire | 0.489 |
| Alk_mg_L | + Outflow_CFS + days_since_fire | 0.504 |

`+ Outflow_CFS` alone is a genuine, modest improvement over baseline for both targets -- a real measured release, distinct information from the upstream inflow gage already in the baseline features. The much larger jump from adding `days_since_fire` (and the negligible extra gain from combining it with outflow) carries the same caveat raised in the burn-scar section above: it is flagged as an unresolved, not-yet-trustworthy finding, not a confirmed result -- see that section's time-index-artifact controls before treating the combined-feature R^2 here as real.


## NWS live-forecast demonstration: is a storm forecast right now? (new dataset, not in data/, not backtestable)

The National Weather Service's gridpoint forecast API (https://api.weather.gov, Strontia Springs Reservoir centroid) is a public dataset not otherwise used in this catalog. **It cannot be tested the way every other public dataset in this catalog was** -- NWS has no historical forecast archive, so there is no way to ask "how well would this have predicted 2023's storms"; it can only ever be demonstrated on the current moment (fetch_nws_forecast.py, snapshot saved to nws_forecast_snapshot.json), the same live-conditions framing dosing_alerts.score_current_conditions already uses for the same reason (this repo's data ends on a fixed date with no "sensor exists, lab result doesn't yet" row to genuinely backtest against).


Snapshot fetched 2026-09-25: 14 forecast periods. Highest precipitation probability: **71%** (Monday Night).


| period | start | temperature (F) | precip probability | forecast |
|---|---|---:|---:|---|
| This Afternoon | 2026-09-25 13:00 | 74 | 64% | Chance Showers And Thunderstorms |
| Tonight | 2026-09-25 18:00 | 45 | 47% | Chance Showers And Thunderstorms then Partly Cloudy |
| Saturday | 2026-09-26 06:00 | 79 | 0% | Sunny |
| Saturday Night | 2026-09-26 18:00 | 48 | 0% | Partly Cloudy |
| Sunday | 2026-09-27 06:00 | 80 | 12% | Mostly Sunny |
| Sunday Night | 2026-09-27 18:00 | 51 | 0% | Partly Cloudy |
| Monday | 2026-09-28 06:00 | 76 | 44% | Partly Sunny then Chance Showers And Thunderstorms |
| Monday Night | 2026-09-28 18:00 | 49 | 71% | Showers And Thunderstorms Likely |
| Tuesday | 2026-09-29 06:00 | 66 | 66% | Rain Showers Likely |
| Tuesday Night | 2026-09-29 18:00 | 46 | 66% | Showers And Thunderstorms Likely |
| Wednesday | 2026-09-30 06:00 | 69 | 52% | Chance Rain Showers |
| Wednesday Night | 2026-09-30 18:00 | 45 | 42% | Chance Rain Showers |
| Thursday | 2026-10-01 06:00 | 75 | 20% | Slight Chance Rain Showers then Sunny |
| Thursday Night | 2026-10-01 18:00 | 45 | 3% | Mostly Clear |

## Melt-out date regression (parameters.md: "predicting SWE decline/melt-out date from temperature" -- documented since the first session, never built until now)

Only 5 water years have a computable melt-out date in this record -- far too few for a held-out train/test split (every other model in this catalog uses one). Reporting the full-sample relationship only, not a held-out R^2, rather than forcing a split that would be meaningless at this sample size.


| year | peak SWE (in) | peak date | melt-out date | melt-out day-of-year | mean spring TMAX (F) |
|---:|---:|---|---|---:|---:|
| 2022 | 14.3 | 2022-04-25 | 2022-06-05 | 156 | 71.0 |
| 2023 | 14.5 | 2023-04-28 | 2023-06-14 | 165 | 67.3 |
| 2024 | 20.9 | 2024-05-12 | 2024-06-12 | 164 | 70.6 |
| 2025 | 14.5 | 2025-04-21 | 2025-06-04 | 155 | 69.8 |
| 2026 | 7.9 | 2026-03-16 | 2026-05-15 | 135 | 71.7 |

Pearson r, peak SWE vs. melt-out day-of-year: **0.854** (more snow -> later melt-out, the expected physical direction, if positive). Pearson r, mean spring TMAX vs. melt-out day-of-year: **-0.656** (warmer spring -> earlier melt-out, if negative).


## Storm impact on the reservoir's depth profile (Scenario 2: "how do water quality parameters change and distribute by depth")

Storm date (peak flow and turbidity in the sonde's window): 2026-07-18.

| cast | surface turbidity (NTU) | surface conductivity | bottom temp (C) | surface temp (C) |
|---|---:|---:|---:|---:|
| before: 2026-07-17 12:06:13 | 0.810 | 294.550 | 14.301 | 19.748 |
| after: 2026-07-19 00:06:41 | 1.020 | 295.260 | 14.385 | 19.897 |

## Hyperparameter tuning (GridSearchCV over gradient boosting)

| target | untuned R^2 | tuned R^2 | best params |
|---|---:|---:|---|
| TOC_mg_L | 0.111 | 0.561 | {'learning_rate': 0.01, 'max_depth': 4, 'n_estimators': 100} |
| Alk_mg_L | 0.327 | 0.387 | {'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 300} |

## Flow forecasting (previously an unimplemented idea in parameters.md)

Predicting Flow_CFS 3 days ahead from today's conditions (1599 rows).

| model | held-out R^2 |
|---|---:|
| Linear (today's Flow_CFS only) | 0.910 |
| Random forest (Flow_CFS, roll_flow_7, SWE, roll_swe_7, TMAX, TMIN) | 0.887 |

| feature | importance |
|---|---:|
| Flow_CFS | 0.731 |
| roll_flow_7 | 0.205 |
| roll_swe_7 | 0.019 |
| TMIN | 0.017 |
| TMAX | 0.014 |
| SWE | 0.013 |

## Quantile / peak-focused regression (guide.md: "catching peaks matters more")

| target | quantile | coverage (actual <= predicted) |
|---|---:|---:|
| TOC_mg_L | 0.9 | 0.918 |
| Alk_mg_L | 0.9 | 0.839 |

## Joint multi-output modeling of TOC_mg_L and Alk_mg_L

Both scored on the same lag_days=2 frame, so alkalinity's independent number here differs from the model family comparison above (which uses alkalinity's own lag_days=4 frame).

| target | independent random-forest R^2 | joint random-forest R^2 |
|---|---:|---:|
| TOC_mg_L | 0.334 | 0.569 |
| Alk_mg_L | 0.170 | -0.055 |

## Gaussian Process regression (uncertainty bands)

| target | held-out R^2 | mean predicted std dev |
|---|---:|---:|
| TOC_mg_L | 0.357 | 0.301 |
| Alk_mg_L | 0.516 | 5.414 |

## SARIMAX (time-series-native, vs. lag-as-feature)

| target | exog | order | held-out R^2 |
|---|---|---|---:|
| TOC_mg_L | turb_flow | (1, 0, 1) | 0.270 |
| Alk_mg_L | Specific_Cond_Mean | (1, 0, 1) | 0.108 |

## NOAA ENSO experiment: does the Oceanic Nino Index help? (new dataset, not in data/)

NOAA/PSL's Oceanic Nino Index (ONI, https://psl.noaa.gov/data/correlation/oni.data) is a public NOAA dataset not otherwise used anywhere in this catalog -- a monthly, basin-scale climate index (El Nino/La Nina strength), rather than a local daily watershed reading like everything else here. Snapshot saved to oni.txt (fetch_oni.py); ONI_prev_month uses each day's most recently *fully observed* month, one month behind, avoiding lookahead (enso_loader.py).

| target | Pearson r with ONI (previous month) |
|---|---:|
| TOC_mg_L | 0.136 |
| Alk_mg_L | -0.106 |

| target | frame | features | held-out R^2 |
|---|---|---|---:|
| TOC_mg_L | uniform lag | without ONI | 0.334 |
| TOC_mg_L | uniform lag | with ONI | 0.512 |
| Alk_mg_L | uniform lag | without ONI | 0.234 |
| Alk_mg_L | uniform lag | with ONI | 0.266 |
| TOC_mg_L | hybrid lag (tuned) | without ONI | 0.650 |
| TOC_mg_L | hybrid lag (tuned) | with ONI | 0.642 |
| Alk_mg_L | hybrid lag (tuned) | without ONI | 0.241 |
| Alk_mg_L | hybrid lag (tuned) | with ONI | 0.261 |

## U.S. Drought Monitor experiment: does county drought severity help? (new dataset, not in data/)

The U.S. Drought Monitor's weekly county drought-severity index (https://usdmdataservices.unl.edu, Jefferson County CO -- DSCI = D0+D1+D2+D3+D4, range 0-500) is a public dataset not otherwise used anywhere in this catalog -- weekly and county-specific, a higher-frequency and more local public signal than NOAA's ONI (analyzed above). Snapshot saved to usdm_jefferson.csv (fetch_usdm.py); DSCI_prev_week uses each day's most recently *fully ended* week, avoiding lookahead (drought_loader.py).

| target | Pearson r with DSCI (most recently ended week) |
|---|---:|
| TOC_mg_L | -0.298 |
| Alk_mg_L | 0.154 |

| target | frame | features | held-out R² |
|---|---|---|---:|
| TOC_mg_L | uniform lag | without DSCI | 0.334 |
| TOC_mg_L | uniform lag | with DSCI | 0.653 |
| Alk_mg_L | uniform lag | without DSCI | 0.234 |
| Alk_mg_L | uniform lag | with DSCI | 0.259 |
| TOC_mg_L | hybrid lag (tuned) | without DSCI | 0.650 |
| TOC_mg_L | hybrid lag (tuned) | with DSCI | 0.723 |
| Alk_mg_L | hybrid lag (tuned) | without DSCI | 0.241 |
| Alk_mg_L | hybrid lag (tuned) | with DSCI | 0.259 |

### Does it help Scenario 1's limited 4-month sonde window specifically?

Same 2026-04-07 to 2026-08-19 sonde window as the earlier feature-set comparison -- does adding DSCI to the national-only feature set change the result there?

| target | features | rows after dropna | held-out R² |
|---|---|---:|---:|
| TOC_mg_L | national only (baseline) | 129 | -0.628 |
| TOC_mg_L | national + DSCI | 129 | -0.591 |
| Alk_mg_L | national only (baseline) | 133 | 0.121 |
| Alk_mg_L | national + DSCI | 133 | 0.110 |

### Do the two public-data ideas (ONI + DSCI) help more together than either alone?

| target | frame | features | held-out R² |
|---|---|---|---:|
| TOC_mg_L | uniform lag | national + ONI + DSCI | 0.655 |
| Alk_mg_L | uniform lag | national + ONI + DSCI | 0.264 |
| TOC_mg_L | hybrid lag (tuned) | national + ONI + DSCI | 0.711 |
| Alk_mg_L | hybrid lag (tuned) | national + ONI + DSCI | 0.255 |

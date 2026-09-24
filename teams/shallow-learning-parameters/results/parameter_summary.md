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
| Alk_mg_L | Random forest | 0.234 |
| Alk_mg_L | SVR (RBF kernel, scaled features) | 0.423 |

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

## Storm impact on the reservoir's depth profile (Scenario 2: "how do water quality parameters change and distribute by depth")

Storm date (peak flow and turbidity in the sonde's window): 2026-07-18.

| cast | surface turbidity (NTU) | surface conductivity | bottom temp (C) | surface temp (C) |
|---|---:|---:|---:|---:|
| before: 2026-07-17 18:01:59 | 0.910 | 296.520 | 18.398 | 20.014 |
| after: 2026-07-19 00:06:41 | 1.020 | 295.260 | 14.385 | 19.897 |

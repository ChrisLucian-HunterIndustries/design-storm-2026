---
name: notebook-leakage-review
description: Use when asked to review a scripts/*.ipynb (or any read-only "original") shallow-learning notebook for data leakage / sample-stratification problems. Ensures the review distinguishes real leakage from normal causal time-series behavior, and respects the read-only rule (advice only, no edits to scripts/).
---

# Notebook leakage review

`scripts/*.ipynb` in this repo are Denver Water's originals (read-only per
AGENTS.md) and reference OneDrive paths that don't resolve from this
workspace anyway — they cannot be run or edited here. Any leakage review of
them is **advice to hand back to Jake**, not a code change. Don't propose
editing `scripts/`; if a fix needs to be demonstrated in code, do it in the
matching `teams/<name>/` implementation instead (check there first — it may
already have the fix, e.g. `models.time_ordered_split` +
`TimeSeriesSplit(...)` scoped to the training half only).

## Checklist (in order of how often each actually appears in these notebooks)

1. **Feature-selection-by-full-dataset-correlation.** A `.corr()` heatmap
   computed on the whole `combined_df` (train+test span) is real leakage if
   the very next cell hand-picks a `features = [...]` list based on that
   ranking — the choice of predictors was informed by the test period's
   statistics. Fix: compute the correlation table (or do the eyeballing)
   on the train slice only.
2. **Repeated test-set peeking across models in one notebook.** Printing
   Test RMSE/R2/MAPE after every one of several models (linear → grid-searched
   RF → CatBoost) in sequence, especially when hand-tuned constants (sample
   weight thresholds, depth grids) visibly differ between notebook versions,
   is leakage via iteration even though every individual split is
   chronological. Fix: pick hyperparameters using only the existing
   `TimeSeriesSplit` CV inside the training slice; treat the test slice as
   spend-once.
3. **Hardcoded weighting/threshold constants** (e.g. `y_train < 60.0`) that
   look eyeballed against test performance — prefer deriving them from a
   train-only quantile so they can't have been chosen by looking at the test
   result.
4. **NOT leakage (verify before flagging):** `.rolling(window=N).mean()` /
   `.diff()` computed on the full continuous series before the split. These
   are backward-looking only (no `center=True` in this codebase) — using
   trailing data through "now" to predict "now" is the intended causal
   behavior for both training and a real deployment, not leakage. Don't
   flag this without checking for `center=True` or negative `.shift()` first.
5. **NOT leakage (usually):** `.shift(N, freq='D')` used to lag an upstream
   sensor before joining — this simulates real operational reporting delay
   (data becomes available N days later) and pushes source data *forward*
   in time, the opposite direction from leakage. Confirm the shift is
   positive before ruling this out.

## Output shape

State each finding as: which cell/feature list, why it's leakage (or why a
suspect isn't), and the train-only equivalent computation that would fix it
— worded so it's directly pasteable into Jake's real notebook, since we
can't paste it there ourselves.

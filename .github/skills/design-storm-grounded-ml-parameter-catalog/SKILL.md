---
name: design-storm-grounded-ml-parameter-catalog
description: Use when asked to catalog which columns in the Design Storm datasets (data/) could serve as shallow-learning inputs/outputs, or to document predictions/visualizations for this repo. Ensures numbers are computed, not invented, and code lands in teams/<name>/ per README.md conventions.
---

# Grounded ML parameter catalog for design-storm-2026

## Where things go
- `data/`, `scripts/`, `figures/`, `reference/` are Denver Water originals —
  read-only. Never edit in place.
- New analysis/code goes under `teams/<team-name>/` (per README.md's "Working
  as a small team" section), never at the repo root.
- Read `guide.md` first — it already documents Jake's feature-engineering
  recipe (rolling means, `turb_flow` loading term, month sin/cos, per-target
  lag days) and his baseline scores. Don't re-derive this from scratch; reuse
  and cite it, and note where a reproduction's numbers differ and why
  (different split, no sample weights, etc.) rather than presenting them as
  equivalent.

## Grounding numbers (AGENTS.md: "Never invent numbers")
1. Write a small, pure data-loading module (path/DataFrame in, DataFrame out)
   separate from any script that touches the real multi-year CSVs. This is
   what makes the pipeline unit-testable with tiny synthetic fixtures instead
   of the real data.
2. Write a separate `analyze_*.py` script that imports that module, runs
   against the real `data/` directory, and writes its computed output
   (correlations, R², feature importances) to a results file. Every number
   quoted in prose/docs must trace back to this file's output — re-run it
   before writing the doc, don't remember/guess a number from a prior run.
3. Only after the results file exists, write the markdown documentation,
   quoting numbers straight from it.

## Testing shape that worked well
- `test_<module>.py` against a hand-written ~10-row synthetic CSV fixture
  (same headers as the real files) for the pure transform logic (joins,
  shifts, rolling windows) — fast, exact assertions possible.
- A separate `test_scripts_smoke.py` that monkeypatches the script's
  module-level `DATA_DIR`/`RESULTS_DIR`/`FIGURES_DIR` constants and points
  them at a larger (60-90 row) *generated* synthetic dataset (rolling
  windows and KMeans need more rows than a 10-row fixture can supply) to
  smoke-test the `main()` orchestration without touching real data.
- Constant/all-zero synthetic columns (e.g. `SNOW: 0`) trigger a
  divide-by-zero `RuntimeWarning` in `DataFrame.corr()`/`corrwith()` — give
  synthetic columns real variance even when the real column is often flat.

## Visualization conventions
- matplotlib only (`matplotlib.use("Agg")` before `pyplot` import) unless the
  environment has seaborn — check with a quick `python -c "import seaborn"`
  before assuming it's available; this environment didn't have it.
- Favor scikit-learn for anything "advanced" (PCA, KMeans, RandomForest) over
  new plotting libraries — matches this repo's "favor sklearn" instruction
  pattern.
- Feature importance / correlation figures should show both targets
  (TOC_mg_L, Alk_mg_L) side by side — Jake's materials only ever did one at a
  time.

## Reviewing the Denver Water PDFs
- `pdfplumber` is installed in this environment and extracts clean text
  page-by-page (`page.extract_text()`) from
  `reference/*.pdf` — use it directly rather than paraphrasing from
  `README.md`'s quotes of the deck, even though those quotes are close.
  `pypdf`/`fitz` (PyMuPDF) are **not** installed here; don't assume either is
  available without checking first.
- When a task asks for a writeup "per challenge/scenario", quote the deck
  verbatim (transcribed from `extract_text()`) before the ML content for each
  scenario, and be explicit about which of the deck's bullets are answered by
  code in this repo vs. blocked by a missing source (e.g. the Strontia sonde
  `.xlsx` is described in `reference/README.md` and `data/TERMS.md` but is
  **not actually present** in this workspace's `data/` — say so rather than
  silently skipping the bullets that depend on it).

## Transit-time / lag analysis (Scenario 3's "follow a parameter")
- Empirically fit an upstream-to-plant lag with a correlation sweep
  (`models.lag_correlation_scan`: for lag in 0..N, shift the predictor
  forward `lag` days with `.shift(lag, freq="D")`, correlate against the
  target, keep the lag with max |r|) rather than reusing Jake's fixed 2-day
  (TOC) / 4-day (alkalinity) lags uncritically — the fitted lags here came
  out different per predictor (turbidity 5d, flow 0d, conductance 3d).
- Always caveat this kind of result with `guide.md` section 1: Denver
  Water's own hydraulic model puts the real water travel time at ~4 hours;
  the multi-day statistical lag reflects reservoir mixing/deposition, not
  transit time. State this explicitly next to any lag number so it isn't
  read as a measured travel time.
- For a "single unified animation" ask: an actual literal flowing-parcel
  animation isn't something this data can ground (no positional/depth
  data). A synchronized moving-cursor GIF across two stacked subplots
  (`matplotlib.animation.FuncAnimation` + `PillowWriter`, saved as `.gif`)
  is an honest, still genuinely useful substitute — needs `pillow` (add it
  to `requirements.txt` explicitly, don't rely on it being an implicit
  matplotlib dependency).
- Pick one concrete real event to narrate (e.g. `idxmax()` of a raw
  turbidity x flow "loading" series) rather than only aggregate
  correlations — a single dated, real storm makes the lag numbers legible
  in a way a correlation table alone doesn't.

## Committing with the RACN MCP tool
- The RACN `commit` tool does not stage changes — run `git add <paths>`
  yourself first, then call the tool. It commits whatever is currently
  staged, so stage exactly one concern's files before each call.
- Call `notation_reference` once per session to confirm current risk/intention
  enum values before classifying.
- Use `theme_mode="inline"` with a shared `theme_slug` to group a
  multi-commit unit of work in the log without creating a branch; reserve
  `theme_mode="d_shaped_merge"` (which creates/merges a branch) for cases
  the user actually wants that.
- Good concern split for a "new analysis folder" unit of work: library
  code+tests, scripts+smoke-tests, generated output (`intention="auto"`),
  hand-written docs, and any process/env changes (AGENTS.md, `.gitignore`)
  each as their own commit.
- Embed generated figures into the writeup with plain relative markdown image
  syntax (`![alt](figures/name.png)`) from the writeup file's own directory,
  so they render in an editor's markdown preview without any build step.

## Model families beyond the random forest
- `sklearn.svm.SVR` is scale-sensitive (unlike the tree models already in
  `models.py`) — wrap it in `make_pipeline(StandardScaler(), SVR(...))`.
  It has no `feature_importances_`; use
  `sklearn.inspection.permutation_importance` on the held-out split instead,
  and return it via the same `RegressionResult` dataclass the random forest
  uses so callers don't need a second result type.
- A lag-day (or any hyperparameter) grid search's R² can swing from
  strongly negative to positive across the grid on a single 50/50 split —
  this is the same instability `guide.md` section 10 already documents for
  cross-validation, showing up on a different axis. Report the range that's
  *consistently* positive/clustered as the meaningful read, not the single
  highest point in the grid.

## Building a static web viewer (Scenario 1's "web application" ask)
- Follow `design-storm-water-system-3d.html`'s own convention: a single
  hand-maintained `.html` file, no build step, plain vanilla JS, CDN
  `<script>` tags via `unpkg.com` (Chart.js + `chartjs-adapter-date-fns` for
  a time-scale x-axis worked well here), served via the repo's `serve.py`
  because it `fetch()`es JSON and won't work as a `file://` URL.
- Export the JSON payload from the *analysis* script (`analyze_parameters.py`),
  not a separate one-off script — it already has the fitted models in scope.
  Split actual-vs-predicted into `predicted_train`/`predicted_test` arrays
  explicitly (not just one `predicted` array) so the page can render the
  held-out portion in a visually distinct color and callers can't mistake
  in-sample fit for generalization.
- Verify a new `.html` page by actually serving it (`python serve.py` in the
  background) and opening it with the browser tools
  (`open_browser_page` + `screenshot_page`/accessibility snapshot) rather
  than just eyeballing the source — confirms the relative fetch path and
  chart rendering actually work. This drops a `.playwright-mcp/` screenshot
  folder in the repo root; gitignore it (`.playwright-mcp/`).

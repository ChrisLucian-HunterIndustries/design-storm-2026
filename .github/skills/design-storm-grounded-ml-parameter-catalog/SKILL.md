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
- Embed generated figures into the writeup with plain relative markdown image
  syntax (`![alt](figures/name.png)`) from the writeup file's own directory,
  so they render in an editor's markdown preview without any build step.

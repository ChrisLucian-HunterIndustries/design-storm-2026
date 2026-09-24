# Shallow-learning parameter exploration

What in `../../data/` can be a shallow-learning input or output, backed by
numbers actually computed from the files. Start with
[`parameters.md`](parameters.md).

- `data_loader.py` / `test_data_loader.py` — load and join the daily CSVs,
  engineer the features from `guide.md` section 7.
- `models.py` / `test_models.py` — the shallow-learning routines (linear,
  random-forest, and SVR regression; RandomForest classification; KMeans/PCA
  clustering; lag-correlation scanning), all scikit-learn.
- `analyze_parameters.py` — runs those models against the real data, writes
  `results/parameter_summary.md` and `results/predictions.json`.
- `visualize.py` — writes the figures (and one animation) in `figures/`.
- `viewer.html` — a small static web app plotting `results/predictions.json`
  interactively; serve with `python3 serve.py` from the repo root, then open
  `http://localhost:8765/teams/shallow-learning-parameters/viewer`.

```
pip install -r requirements.txt
python analyze_parameters.py
python visualize.py
python -m pytest -q
```

Denver Water's data terms (`../../data/TERMS.md`) apply to everything derived
here.

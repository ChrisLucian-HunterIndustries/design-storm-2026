# Shallow-learning parameter exploration

What in `../../data/` can be a shallow-learning input or output, backed by
numbers actually computed from the files. Start with
[`parameters.md`](parameters.md).

- `data_loader.py` / `test_data_loader.py` — load and join the daily CSVs,
  engineer the features from `guide.md` section 7.
- `models.py` / `test_models.py` — the shallow-learning routines (linear and
  random-forest regression, RandomForest classification, KMeans/PCA
  clustering), all scikit-learn.
- `analyze_parameters.py` — runs those models against the real data, writes
  `results/parameter_summary.md`.
- `visualize.py` — writes the figures in `figures/`.

```
pip install -r requirements.txt
python analyze_parameters.py
python visualize.py
python -m pytest -q
```

Denver Water's data terms (`../../data/TERMS.md`) apply to everything derived
here.

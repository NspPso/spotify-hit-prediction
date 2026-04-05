# Spotify Hit Prediction Project

This project builds a leakage-safe binary hit-prediction pipeline on a public Spotify tracks dataset using the target below:

- `hit = 1` when `popularity >= 7`
- `hit = 0` otherwise

The workflow compares five models:

- logistic regression
- decision tree
- random forest
- XGBoost
- multilayer perceptron (PyTorch)

## Highlights

- Built with Python, Pandas, NumPy, scikit-learn, XGBoost, and PyTorch
- Uses group-aware splitting to prevent the same song from leaking across train and test
- Benchmarks multiple models instead of relying on a single result
- Reports balanced accuracy, ROC-AUC, average precision, F1, and log loss

## Why The Pipeline Is Leakage-Safe

The dataset contains many repeated `track_id` values. The same song can appear under multiple genres, so a random row split would leak songs across train and test.

To avoid that, this project:

- splits data with `StratifiedGroupKFold`
- groups by `track_id`
- fits imputers, scalers, and encoders on training folds only
- drops `popularity` from the feature set because the target is derived from it

## Features Used

The model keeps structured Spotify metadata and audio descriptors:

- numeric: `duration_ms`, `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- categorical: `explicit`, `key`, `mode`, `time_signature`, `track_genre`

The following columns are excluded:

- `popularity` because it defines the target
- `Unnamed: 0` because it is just an index
- `track_id`, `artists`, `album_name`, `track_name` to avoid identifier memorization and high-cardinality text leakage

## Metrics

Because `popularity >= 7` creates a majority-positive target, accuracy alone is misleading. The project uses:

- primary: balanced accuracy
- secondary: ROC-AUC, average precision, F1, precision, recall, accuracy, log loss

## Quick Start

```powershell
python scripts/run_analysis.py
```

This will:

1. download the Spotify dataset if it is missing
2. inspect the dataset and save summary tables/plots
3. create a leakage-safe train/test split
4. run cross-validation on the training portion
5. fit the final models
6. save figures, tables, and a short summary report under `artifacts/`

## Optional Data Download Only

```powershell
python scripts/download_data.py
```

## Notes

- The threshold `7` is implemented exactly as requested. If you later meant `70`, change `HIT_THRESHOLD` in [config.py](C:\Users\Malcolm\OneDrive\Documents\New project\spotify_hit_project\src\spotify_hit_project\config.py).
- The MLP uses PyTorch rather than scikit-learn's `MLPClassifier` so the deep learning artifact is a true neural network implementation.
- The raw dataset is not intended to be committed in a public portfolio copy of this repo. Re-run the download script if you need a local copy.

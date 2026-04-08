from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
TABLES_DIR = ARTIFACTS_DIR / "tables"

DATA_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
RAW_DATA_PATH = DATA_RAW_DIR / "spotify_tracks.csv"

RANDOM_STATE = 42
HIT_THRESHOLD = 7
TARGET_COLUMN = "hit"
POPULARITY_COLUMN = "popularity"
GROUP_COLUMN = "track_id"

OUTER_SPLITS = 5
INNER_CV_SPLITS = 3
TEST_FOLD_INDEX = 0
MLP_MAX_EPOCHS = 35
MLP_BATCH_SIZE = 512
EMBEDDING_MLP_MAX_EPOCHS = 45
EMBEDDING_MLP_BATCH_SIZE = 512

DROP_COLUMNS = [
    "Unnamed: 0",
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
]

NUMERIC_FEATURES = [
    "duration_ms",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

CATEGORICAL_FEATURES = [
    "explicit",
    "key",
    "mode",
    "time_signature",
    "track_genre",
]

BASE_MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "mlp",
    "embedding_mlp",
]

ENSEMBLE_BLEND_MODELS = [
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "xgboost",
    "mlp",
    "embedding_mlp",
]

MODEL_NAMES = BASE_MODEL_NAMES

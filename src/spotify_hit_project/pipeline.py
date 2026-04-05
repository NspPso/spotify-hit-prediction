from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from .config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    DATA_URL,
    DROP_COLUMNS,
    FIGURES_DIR,
    GROUP_COLUMN,
    HIT_THRESHOLD,
    INNER_CV_SPLITS,
    MLP_BATCH_SIZE,
    MLP_MAX_EPOCHS,
    MODEL_NAMES,
    NUMERIC_FEATURES,
    OUTER_SPLITS,
    POPULARITY_COLUMN,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TABLES_DIR,
    TARGET_COLUMN,
    TEST_FOLD_INDEX,
)

plt.switch_backend("Agg")


@dataclass
class TrainedTorchModel:
    model: nn.Module
    device: str
    best_epoch: int


def ensure_dirs() -> None:
    for path in [RAW_DATA_PATH.parent, ARTIFACTS_DIR, FIGURES_DIR, TABLES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def download_data(force: bool = False) -> Path:
    ensure_dirs()
    if RAW_DATA_PATH.exists() and not force:
        return RAW_DATA_PATH
    df = pd.read_csv(DATA_URL)
    df.to_csv(RAW_DATA_PATH, index=False)
    return RAW_DATA_PATH


def load_data() -> pd.DataFrame:
    download_data(force=False)
    df = pd.read_csv(RAW_DATA_PATH)
    df[TARGET_COLUMN] = (df[POPULARITY_COLUMN] >= HIT_THRESHOLD).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "to_string",
                FunctionTransformer(
                    lambda frame: frame.astype(str),
                    feature_names_out="one-to-one",
                ),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    duplicate_track_rows = int(df.duplicated(subset=[GROUP_COLUMN]).sum())
    track_groups = df.groupby(GROUP_COLUMN)
    conflicting_target_groups = int(track_groups[TARGET_COLUMN].nunique().gt(1).sum())

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "unique_track_ids": int(df[GROUP_COLUMN].nunique()),
        "duplicate_track_rows": duplicate_track_rows,
        "track_groups_with_conflicting_hit_labels": conflicting_target_groups,
        "genre_count": int(df["track_genre"].nunique()),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_by_column": {
            key: int(value)
            for key, value in df.isna().sum().items()
            if int(value) > 0
        },
        "target_balance": {
            int(key): int(value)
            for key, value in df[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "target_rate": float(df[TARGET_COLUMN].mean()),
        "popularity_summary": {
            key: float(value)
            for key, value in df[POPULARITY_COLUMN].describe().to_dict().items()
        },
    }


def save_dataset_tables_and_plots(df: pd.DataFrame, summary: dict[str, Any]) -> None:
    inspection_rows: list[dict[str, Any]] = []
    for key, value in summary.items():
        if isinstance(value, dict):
            inspection_rows.append({"metric": key, "value": json.dumps(value)})
        else:
            inspection_rows.append({"metric": key, "value": value})
    pd.DataFrame(inspection_rows).to_csv(TABLES_DIR / "dataset_summary.csv", index=False)

    missing_df = (
        df.isna()
        .sum()
        .rename("missing_count")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values(by="missing_count", ascending=False)
    )
    missing_df.to_csv(TABLES_DIR / "missing_values.csv", index=False)

    popularity_stats = (
        df[POPULARITY_COLUMN]
        .describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        .rename("value")
        .reset_index()
        .rename(columns={"index": "statistic"})
    )
    popularity_stats.to_csv(TABLES_DIR / "popularity_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df[POPULARITY_COLUMN], bins=25, color="#1f77b4", edgecolor="white")
    axes[0].axvline(HIT_THRESHOLD, color="#d62728", linestyle="--", linewidth=2)
    axes[0].set_title("Popularity Distribution")
    axes[0].set_xlabel("Popularity")
    axes[0].set_ylabel("Count")

    target_counts = df[TARGET_COLUMN].value_counts().sort_index()
    axes[1].bar(
        ["not_hit (0)", "hit (1)"],
        target_counts.to_numpy(),
        color=["#8c564b", "#2ca02c"],
    )
    axes[1].set_title(f"Target Balance (hit >= {HIT_THRESHOLD})")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dataset_inspection.png", dpi=200)
    plt.close(fig)


def make_outer_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedGroupKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    split_iter = splitter.split(
        df[feature_columns()],
        df[TARGET_COLUMN],
        groups=df[GROUP_COLUMN],
    )
    for fold_index, (train_idx, test_idx) in enumerate(split_iter):
        if fold_index == TEST_FOLD_INDEX:
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[test_idx].reset_index(drop=True)
            return train_df, test_df
    raise RuntimeError("Failed to construct outer split.")


def save_split_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    overlap = set(train_df[GROUP_COLUMN]).intersection(set(test_df[GROUP_COLUMN]))
    summary = {
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "train_track_ids": int(train_df[GROUP_COLUMN].nunique()),
        "test_track_ids": int(test_df[GROUP_COLUMN].nunique()),
        "train_hit_rate": float(train_df[TARGET_COLUMN].mean()),
        "test_hit_rate": float(test_df[TARGET_COLUMN].mean()),
        "train_test_track_overlap": int(len(overlap)),
    }
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.items()]
    ).to_csv(TABLES_DIR / "split_summary.csv", index=False)
    return summary


def to_float32(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }


def negative_positive_ratio(y: np.ndarray) -> float:
    positives = float(np.sum(y == 1))
    negatives = float(np.sum(y == 0))
    if positives == 0:
        return 1.0
    return negatives / positives


def build_sklearn_model(model_name: str, y_train: np.ndarray) -> Any:
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )
    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=25,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=4,
            verbosity=0,
            scale_pos_weight=negative_positive_ratio(y_train),
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def predict_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(X)
    return np.asarray(probabilities[:, 1], dtype=np.float64)


class SpotifyMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def fit_torch_mlp_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = MLP_MAX_EPOCHS,
    batch_size: int = MLP_BATCH_SIZE,
    patience: int = 6,
) -> TrainedTorchModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpotifyMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative_positive_ratio(y_train)], dtype=torch.float32, device=device)
    )

    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    best_score = -np.inf
    best_epoch = 1
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
        score = average_precision_score(y_val, val_probs)

        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return TrainedTorchModel(model=model, device=device, best_epoch=best_epoch)


def fit_torch_mlp_fixed_epochs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int = MLP_BATCH_SIZE,
) -> TrainedTorchModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SpotifyMLP(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative_positive_ratio(y_train)], dtype=torch.float32, device=device)
    )
    train_loader = make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)

    for _ in range(max(1, epochs)):
        model.train()
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    return TrainedTorchModel(model=model, device=device, best_epoch=max(1, epochs))


def predict_torch_probabilities(
    trained_model: TrainedTorchModel,
    X: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    model = trained_model.model
    device = trained_model.device
    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for (features,) in loader:
            logits = model(features.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float64)


def cross_validate_models(train_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    splitter = StratifiedGroupKFold(
        n_splits=INNER_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    X_df = train_df[feature_columns()]
    y = train_df[TARGET_COLUMN].to_numpy()
    groups = train_df[GROUP_COLUMN].to_numpy()
    rows: list[dict[str, Any]] = []
    mlp_best_epochs: list[int] = []

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(X_df, y, groups=groups), start=1):
        X_fit_df = X_df.iloc[fit_idx]
        X_val_df = X_df.iloc[val_idx]
        y_fit = y[fit_idx]
        y_val = y[val_idx]

        preprocessor = build_preprocessor()
        X_fit = to_float32(preprocessor.fit_transform(X_fit_df))
        X_val = to_float32(preprocessor.transform(X_val_df))

        for model_name in MODEL_NAMES:
            if model_name == "mlp":
                trained_model = fit_torch_mlp_with_early_stopping(X_fit, y_fit, X_val, y_val)
                probabilities = predict_torch_probabilities(trained_model, X_val)
                extra = {"best_epoch": trained_model.best_epoch}
                mlp_best_epochs.append(trained_model.best_epoch)
            else:
                model = build_sklearn_model(model_name, y_fit)
                model.fit(X_fit, y_fit)
                probabilities = predict_probabilities(model, X_val)
                extra = {"best_epoch": np.nan}

            metrics = compute_metrics(y_val, probabilities)
            rows.append({"model": model_name, "fold": fold, **metrics, **extra})

    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(TABLES_DIR / "cv_fold_metrics.csv", index=False)

    cv_summary = (
        cv_df.groupby("model", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            average_precision_mean=("average_precision", "mean"),
            log_loss_mean=("log_loss", "mean"),
            best_epoch_median=("best_epoch", "median"),
        )
        .sort_values(by=["balanced_accuracy_mean", "roc_auc_mean"], ascending=False)
    )
    cv_summary.to_csv(TABLES_DIR / "cv_summary.csv", index=False)

    mlp_epochs = 12
    if mlp_best_epochs:
        mlp_epochs = int(max(5, round(float(np.median(mlp_best_epochs)))))
    return cv_summary, mlp_epochs


def fit_final_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mlp_epochs: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], str]:
    X_train_df = train_df[feature_columns()]
    y_train = train_df[TARGET_COLUMN].to_numpy()
    X_test_df = test_df[feature_columns()]
    y_test = test_df[TARGET_COLUMN].to_numpy()

    preprocessor = build_preprocessor()
    X_train = to_float32(preprocessor.fit_transform(X_train_df))
    X_test = to_float32(preprocessor.transform(X_test_df))

    rows: list[dict[str, Any]] = []
    probability_map: dict[str, np.ndarray] = {}

    for model_name in MODEL_NAMES:
        if model_name == "mlp":
            trained_model = fit_torch_mlp_fixed_epochs(X_train, y_train, epochs=mlp_epochs)
            probabilities = predict_torch_probabilities(trained_model, X_test)
        else:
            model = build_sklearn_model(model_name, y_train)
            model.fit(X_train, y_train)
            probabilities = predict_probabilities(model, X_test)

        probability_map[model_name] = probabilities
        rows.append({"model": model_name, **compute_metrics(y_test, probabilities)})

    results_df = pd.DataFrame(rows).sort_values(
        by=["balanced_accuracy", "roc_auc"],
        ascending=False,
    )
    results_df.to_csv(TABLES_DIR / "model_comparison.csv", index=False)

    best_model_name = str(results_df.iloc[0]["model"])
    return results_df, probability_map, best_model_name


def save_model_comparison_plot(results_df: pd.DataFrame) -> None:
    metrics_to_plot = ["balanced_accuracy", "roc_auc", "f1"]
    x = np.arange(len(results_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, metric_name in zip([-width, 0.0, width], metrics_to_plot, strict=True):
        ax.bar(
            x + offset,
            results_df[metric_name],
            width=width,
            label=metric_name.replace("_", " ").title(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"], rotation=15)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Test Set Model Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=220)
    plt.close(fig)


def save_curve_plots(y_true: np.ndarray, probability_map: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, probabilities in probability_map.items():
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc = roc_auc_score(y_true, probabilities)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=220)
    plt.close(fig)

    positive_rate = float(np.mean(y_true))
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, probabilities in probability_map.items():
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        ap = average_precision_score(y_true, probabilities)
        ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")
    ax.axhline(positive_rate, linestyle="--", color="gray", linewidth=1, label="baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=220)
    plt.close(fig)


def save_confusion_outputs(y_true: np.ndarray, probabilities: np.ndarray, model_name: str) -> None:
    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(TABLES_DIR / "best_model_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Best Model Confusion Matrix: {model_name}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "best_model_confusion_matrix.png", dpi=220)
    plt.close(fig)


def save_report_context(context: dict[str, Any]) -> None:
    (ARTIFACTS_DIR / "report_context.json").write_text(
        json.dumps(context, indent=2),
        encoding="utf-8",
    )


def save_workflow_summary(
    dataset_summary: dict[str, Any],
    split_summary: dict[str, Any],
    cv_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    best_model_name: str,
    mlp_epochs: int,
) -> None:
    best_row = test_summary.iloc[0]
    lines = [
        "# Spotify Hit Prediction Summary",
        "",
        "## Dataset Inspection",
        f"- Rows: {dataset_summary['rows']:,}",
        f"- Unique tracks: {dataset_summary['unique_track_ids']:,}",
        f"- Duplicate track rows by `track_id`: {dataset_summary['duplicate_track_rows']:,}",
        f"- Hit definition: `popularity >= {HIT_THRESHOLD}`",
        f"- Positive class rate: {dataset_summary['target_rate']:.3f}",
        "",
        "## Leakage Prevention",
        "- Outer and inner splits are group-aware on `track_id`.",
        "- The same song never appears in both train and test.",
        "- `popularity` is dropped from the feature matrix because it defines the target.",
        "- Imputation, scaling, and one-hot encoding are fit on training folds only.",
        "",
        "## Preprocessing",
        f"- Numeric features: {', '.join(NUMERIC_FEATURES)}",
        f"- Categorical features: {', '.join(CATEGORICAL_FEATURES)}",
        f"- Dropped columns: {', '.join(DROP_COLUMNS)}",
        "",
        "## Evaluation",
        "- Primary metric: balanced accuracy.",
        "- Secondary metrics: ROC-AUC, average precision, F1, precision, recall, accuracy, and log loss.",
        "- Accuracy is reported but not used as the main ranking metric because the positive class dominates.",
        "",
        "## Split Summary",
        f"- Train rows: {split_summary['train_rows']:,}",
        f"- Test rows: {split_summary['test_rows']:,}",
        f"- Train hit rate: {split_summary['train_hit_rate']:.3f}",
        f"- Test hit rate: {split_summary['test_hit_rate']:.3f}",
        f"- Train/test overlapping `track_id` values: {split_summary['train_test_track_overlap']}",
        "",
        "## Cross-Validation Snapshot",
    ]

    for row in cv_summary.to_dict("records"):
        lines.append(
            "- "
            f"{row['model']}: balanced_accuracy={row['balanced_accuracy_mean']:.3f}, "
            f"roc_auc={row['roc_auc_mean']:.3f}, "
            f"f1={row['f1_mean']:.3f}"
        )

    lines.extend(["", "## Test Set Results"])
    for row in test_summary.to_dict("records"):
        lines.append(
            "- "
            f"{row['model']}: balanced_accuracy={row['balanced_accuracy']:.3f}, "
            f"roc_auc={row['roc_auc']:.3f}, "
            f"average_precision={row['average_precision']:.3f}, "
            f"f1={row['f1']:.3f}"
        )

    lines.extend(
        [
            "",
            "## Best Model",
            f"- Best model by balanced accuracy: `{best_model_name}`",
            f"- Balanced accuracy: {best_row['balanced_accuracy']:.3f}",
            f"- ROC-AUC: {best_row['roc_auc']:.3f}",
            f"- Average precision: {best_row['average_precision']:.3f}",
            f"- F1: {best_row['f1']:.3f}",
            f"- Final MLP epochs selected from CV: {mlp_epochs}",
        ]
    )

    (ARTIFACTS_DIR / "workflow_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline() -> dict[str, Any]:
    ensure_dirs()
    set_seed()
    df = load_data()

    dataset_summary = inspect_dataset(df)
    save_dataset_tables_and_plots(df, dataset_summary)

    train_df, test_df = make_outer_split(df)
    split_summary = save_split_summary(train_df, test_df)

    cv_summary, mlp_epochs = cross_validate_models(train_df)
    test_summary, probability_map, best_model_name = fit_final_models(train_df, test_df, mlp_epochs)

    save_model_comparison_plot(test_summary)
    save_curve_plots(test_df[TARGET_COLUMN].to_numpy(), probability_map)
    save_confusion_outputs(
        test_df[TARGET_COLUMN].to_numpy(),
        probability_map[best_model_name],
        best_model_name,
    )

    context = {
        "dataset_summary": dataset_summary,
        "split_summary": split_summary,
        "cv_summary": cv_summary.to_dict("records"),
        "test_summary": test_summary.to_dict("records"),
        "best_model": best_model_name,
        "mlp_final_epochs": mlp_epochs,
        "artifacts_dir": str(ARTIFACTS_DIR),
    }
    save_report_context(context)
    save_workflow_summary(
        dataset_summary=dataset_summary,
        split_summary=split_summary,
        cv_summary=cv_summary,
        test_summary=test_summary,
        best_model_name=best_model_name,
        mlp_epochs=mlp_epochs,
    )
    return context

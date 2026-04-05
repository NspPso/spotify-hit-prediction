# Spotify Hit Prediction Summary

## Dataset Inspection
- Rows: 114,000
- Unique tracks: 89,741
- Duplicate track rows by `track_id`: 24,259
- Hit definition: `popularity >= 7`
- Positive class rate: 0.814

## Leakage Prevention
- Outer and inner splits are group-aware on `track_id`.
- The same song never appears in both train and test.
- `popularity` is dropped from the feature matrix because it defines the target.
- Imputation, scaling, and one-hot encoding are fit on training folds only.

## Preprocessing
- Numeric features: duration_ms, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
- Categorical features: explicit, key, mode, time_signature, track_genre
- Dropped columns: Unnamed: 0, track_id, artists, album_name, track_name, popularity

## Evaluation
- Primary metric: balanced accuracy.
- Secondary metrics: ROC-AUC, average precision, F1, precision, recall, accuracy, and log loss.
- Accuracy is reported but not used as the main ranking metric because the positive class dominates.

## Split Summary
- Train rows: 91,200
- Test rows: 22,800
- Train hit rate: 0.814
- Test hit rate: 0.814
- Train/test overlapping `track_id` values: 0

## Cross-Validation Snapshot
- random_forest: balanced_accuracy=0.824, roc_auc=0.910, f1=0.937
- mlp: balanced_accuracy=0.823, roc_auc=0.902, f1=0.878
- xgboost: balanced_accuracy=0.803, roc_auc=0.890, f1=0.904
- logistic_regression: balanced_accuracy=0.788, roc_auc=0.870, f1=0.860
- decision_tree: balanced_accuracy=0.633, roc_auc=0.634, f1=0.915

## Test Set Results
- random_forest: balanced_accuracy=0.838, roc_auc=0.917, average_precision=0.975, f1=0.938
- mlp: balanced_accuracy=0.827, roc_auc=0.907, average_precision=0.974, f1=0.881
- xgboost: balanced_accuracy=0.808, roc_auc=0.894, average_precision=0.971, f1=0.903
- logistic_regression: balanced_accuracy=0.789, roc_auc=0.873, average_precision=0.965, f1=0.858
- decision_tree: balanced_accuracy=0.635, roc_auc=0.636, average_precision=0.858, f1=0.915

## Best Model
- Best model by balanced accuracy: `random_forest`
- Balanced accuracy: 0.838
- ROC-AUC: 0.917
- Average precision: 0.975
- F1: 0.938
- Final MLP epochs selected from CV: 23
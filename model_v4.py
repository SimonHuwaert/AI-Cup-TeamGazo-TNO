"""
Bird-group classification (AI Cup 2026) — Model V4.

Trains a CatBoost multiclass classifier on radar track features to predict which
of 9 bird groups (or "Clutter") produced each track. Robustness comes from an
ensemble: 5 cross-validation folds x 5 random seeds = 25 models whose predicted
probabilities are averaged. Quality is measured with macro mean Average Precision
(mAP), the competition metric, computed out-of-fold (OOF).

Feature engineering lives entirely in features.py — this file imports
`add_features` from there and never computes a feature itself. It only decides
which of the resulting columns to feed the model, trains, and writes the submission.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool
import warnings

from features import add_features   # all feature engineering is defined here

warnings.filterwarnings('ignore')

# Ensemble size: N_FOLDS x N_SEEDS models are trained and their probabilities averaged.
N_SEEDS = 5
N_FOLDS = 5

# Exact column order Kaggle expects in the submission file (order matters for scoring).
KAGGLE_COLUMN_ORDER = [
    'Clutter', 'Cormorants', 'Pigeons', 'Ducks', 'Geese',
    'Gulls', 'Birds of Prey', 'Waders', 'Songbirds'
]

# Columns that are identifiers, labels, raw inputs, or bookkeeping — never fed to
# the model as features. Everything else (the engineered features from features.py
# plus the raw radar summaries like airspeed/min_z/max_z/radar_bird_size) is used.
META_COLS = {
    "track_id", "trajectory", "trajectory_time", "timestamp_start_radar_utc",
    "timestamp_end_radar_utc", "observation_id", "primary_observation_id",
    "observer_position", "observer_comment", "n_birds_observed",
    "bird_species", "timestamp", "bird_group", "is_augmented",
    "original_track_id", "parent_idx",
}


def get_features(df):
    """Return the list of column names to use as model inputs.

    Drops meta/label/raw columns and any `weather_*` columns; whatever remains —
    the engineered features attached by features.py, plus the leftover raw radar
    summaries — is what the model trains on.
    """
    return [c for c in df.columns
            if c not in META_COLS and not c.startswith('weather_')]


def calculate_map(y_true, y_pred_probs, n_classes):
    """Macro mean Average Precision — the competition metric.

    One-hot encodes the true labels, then averages the per-class AP (area under the
    precision-recall curve) equally across classes so rare classes count as much as common ones.
    """
    y_bin = label_binarize(y_true, classes=range(n_classes))
    return average_precision_score(y_bin, y_pred_probs, average="macro")


def main():
    print("=" * 60)
    print("  BIRD CLASSIFICATION — Model V4")
    print("  5-fold x 5-seed CatBoost ensemble")
    print("=" * 60)

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Build the exact same feature set on train and test, straight from features.py.
    print("\nExtracting features (features.py)...")
    train = add_features(train)
    test = add_features(test)

    # Select the feature columns and slice train/test to exactly those.
    features = get_features(train)
    print(f"Using {len(features)} features")

    X = train[features].copy()
    X_test = test[features].copy()

    # CatBoost handles categoricals natively but needs them as clean strings with no NaNs.
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_features:
        X[col] = X[col].fillna("Unknown").astype(str)
        X_test[col] = X_test[col].fillna("Unknown").astype(str)

    # Encode string class labels to integers 0..n-1; keep the mapping for output.
    le = LabelEncoder()
    y = le.fit_transform(train["bird_group"])
    class_names = le.classes_
    n_classes = len(class_names)

    print(f"Classes: {list(class_names)}")
    print(f"Samples: {len(X)}")

    # Class weights to counter imbalance. Base weight is the inverse-frequency weight
    # softened by a square root (so rare classes are up-weighted, but not extremely).
    class_counts = np.bincount(y)
    base_weights = (len(y) / (n_classes * class_counts)) ** 0.5
    # Extra manual boosts for classes that are hard/valuable to get right.
    for name, boost in [('Cormorants', 1.4), ('Birds of Prey', 1.2),
                        ('Waders', 1.2), ('Ducks', 1.1), ('Geese', 1.1)]:
        idx = np.where(class_names == name)[0][0]
        base_weights[idx] *= boost
    cw = {i: float(w) for i, w in enumerate(base_weights)}

    # Shared CatBoost hyperparameters (regularization-heavy to resist overfitting).
    CB_PARAMS = dict(
        iterations=1200,
        learning_rate=0.04,
        depth=7,
        l2_leaf_reg=10,
        min_data_in_leaf=40,
        grow_policy='Lossguide',
        max_leaves=64,
        bagging_temperature=0.3,
        rsm=0.7,
        border_count=128,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        early_stopping_rounds=100,
        task_type='CPU',
        verbose=False,
    )

    # Fixed stratified folds (same split reused across every seed for comparability).
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_splits = list(skf.split(X, y))

    # Accumulators for the final averaged predictions.
    oof_preds = np.zeros((len(X), n_classes))       # out-of-fold train predictions (for honest scoring)
    test_preds = np.zeros((len(X_test), n_classes))  # test predictions (for submission)

    print(f"\nTraining {N_FOLDS}-fold x {N_SEEDS}-seed ({N_FOLDS * N_SEEDS} models)...\n")

    last_model = None
    # Outer loop: repeat the whole CV with different random seeds and average — seed
    # averaging reduces the variance of any single random initialization.
    for seed in range(42, 42 + N_SEEDS):
        oof_seed = np.zeros((len(X), n_classes))
        test_seed = np.zeros((len(X_test), n_classes))

        # Inner loop: standard k-fold CV. Each fold trains on k-1 parts, predicts the held-out part.
        for fold, (tr_idx, va_idx) in enumerate(fold_splits):
            model = CatBoostClassifier(
                **CB_PARAMS,
                class_weights=cw,
                random_seed=seed + fold * 100,  # distinct seed per (seed, fold) model
            )
            train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_features)
            val_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_features)
            model.fit(train_pool, eval_set=val_pool)  # val_pool drives early stopping

            # Predict the validation fold once (no leakage) and add this fold's share of the test preds.
            oof_seed[va_idx] = model.predict_proba(X.iloc[va_idx])
            test_seed += model.predict_proba(X_test) / N_FOLDS
            last_model = model

        # This seed's OOF score, then fold the seed's results into the grand average.
        seed_map = calculate_map(y, oof_seed, n_classes)
        print(f"  Seed {seed}: OOF mAP = {seed_map:.4f}")

        oof_preds += oof_seed / N_SEEDS
        test_preds += test_seed / N_SEEDS

    # Overall cross-validated score of the full ensemble — the number to trust when tuning.
    final_map = calculate_map(y, oof_preds, n_classes)
    print(f"\nFinal OOF mAP: {final_map:.4f}")

    # Per-class breakdown to see which bird groups the model struggles with.
    y_bin = label_binarize(y, classes=range(n_classes))
    print("\nPer-class AP:")
    for i, name in enumerate(class_names):
        ap = average_precision_score(y_bin[:, i], oof_preds[:, i])
        print(f"  {name:<15}: {ap:.4f}")

    # Feature importances from the last trained model (rough guide to what drives predictions).
    if last_model is not None:
        print("\nTop 20 features:")
        imp = last_model.get_feature_importance()
        for fname, fi in sorted(zip(features, imp), key=lambda x: -x[1])[:20]:
            print(f"  {fname:<35}: {fi:.2f}")

    # Build the submission: one probability column per class, in Kaggle's required order.
    submission = pd.DataFrame(test_preds, columns=class_names)
    submission.insert(0, 'track_id', test['track_id'])
    submission = submission[['track_id'] + KAGGLE_COLUMN_ORDER]
    submission.to_csv("submission_v4.csv", index=False)
    print(f"\nSaved submission_v4.csv ({len(submission)} rows)")


if __name__ == "__main__":
    main()
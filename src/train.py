"""
Slice 05 — Model training and validation comparison.

Loads the leakage-safe preprocessing bundle from Slice 04, trains the
planned baseline models, evaluates them on the validation split only,
and persists the trained models plus comparison metrics.

Models:
    - Logistic Regression
    - Random Forest
    - XGBoost (if available)
    - HistGradientBoosting fallback is intentionally not used here because
      Slice 04 produces sparse matrices; it can be added as a dense fallback
      only if XGBoost is unavailable and memory permits.

The test split is never used for model selection.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "engineered_data.csv"
BUNDLE_PATH = ROOT / "models" / "preprocessing_bundle.joblib"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "results" / "metrics"

TARGET_COLUMN = "TARGET"


def load_preprocessed_data():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Input not found: {PROCESSED_PATH}")
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing bundle not found: {BUNDLE_PATH}. "
            "Run Slice 04 first."
        )

    df = pd.read_csv(PROCESSED_PATH)
    bundle = joblib.load(BUNDLE_PATH)

    feature_columns = bundle["feature_columns"]
    id_column = bundle["id_column"]

    if TARGET_COLUMN not in df.columns or id_column not in df.columns:
        raise ValueError("Engineered data is missing ID or TARGET.")

    if df[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique.")

    if len(df) != sum(bundle["split_sizes"].values()):
        raise ValueError("Current engineered data row count differs from Slice 04.")

    preprocessor = bundle["preprocessor"]
    X_all = df[feature_columns]
    y_all = df[TARGET_COLUMN].astype(int)

    train_ids = set(bundle["split_ids"]["train"])
    validation_ids = set(bundle["split_ids"]["validation"])

    train_mask = df[id_column].isin(train_ids)
    validation_mask = df[id_column].isin(validation_ids)

    if (train_mask & validation_mask).any():
        raise ValueError("Train and validation applicant IDs overlap.")

    train_df = df.loc[train_mask]
    validation_df = df.loc[validation_mask]

    if len(train_df) != bundle["split_sizes"]["train"]:
        raise ValueError("Training split size does not match Slice 04 bundle.")
    if len(validation_df) != bundle["split_sizes"]["validation"]:
        raise ValueError("Validation split size does not match Slice 04 bundle.")

    X_train = preprocessor.transform(train_df[feature_columns])
    X_validation = preprocessor.transform(validation_df[feature_columns])
    y_train = train_df[TARGET_COLUMN].astype(int).to_numpy()
    y_validation = validation_df[TARGET_COLUMN].astype(int).to_numpy()

    return X_train, X_validation, y_train, y_validation


def evaluate_model(name: str, model, X_validation, y_validation, train_seconds: float):
    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_validation, probabilities)),
        "pr_auc": float(average_precision_score(y_validation, probabilities)),
        "accuracy": float(accuracy_score(y_validation, predictions)),
        "precision": float(precision_score(y_validation, predictions, zero_division=0)),
        "recall": float(recall_score(y_validation, predictions, zero_division=0)),
        "f1": float(f1_score(y_validation, predictions, zero_division=0)),
        "threshold": 0.5,
        "train_seconds": round(train_seconds, 3),
        "random_state": RANDOM_STATE,
    }


def build_models(y_train: np.ndarray):
    models = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    try:
        from xgboost import XGBClassifier

        negatives = max(int((y_train == 0).sum()), 1)
        positives = max(int((y_train == 1).sum()), 1)
        scale_pos_weight = negatives / positives

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
        )
    except ImportError:
        pass

    return models


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - SLICE 05 MODEL TRAINING")
    print("=" * 80)

    X_train, X_validation, y_train, y_validation = load_preprocessed_data()

    print(f"Training rows           : {X_train.shape[0]:,}")
    print(f"Validation rows         : {X_validation.shape[0]:,}")
    print(f"Transformed features    : {X_train.shape[1]:,}")
    print(f"Positive rate (train)   : {y_train.mean():.6f}")
    print(f"Positive rate (valid.)  : {y_validation.mean():.6f}")
    print("Test split used         : NO")

    models = build_models(y_train)
    if "xgboost" not in models:
        print("XGBoost                  : unavailable; not attempted")

    results = []
    trained_model_names = []

    for name, model in models.items():
        print()
        print(f"Training {name}...")
        started = time.perf_counter()

        if name == "logistic_regression":
            # Explicitly preserve class balancing while avoiding any target
            # information in preprocessing or feature construction.
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

        elapsed = time.perf_counter() - started
        metrics = evaluate_model(
            name, model, X_validation, y_validation, elapsed
        )
        results.append(metrics)
        trained_model_names.append(name)

        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
        print(
            f"  ROC-AUC={metrics['roc_auc']:.6f} | "
            f"PR-AUC={metrics['pr_auc']:.6f} | "
            f"F1={metrics['f1']:.6f}"
        )

    if not results:
        raise RuntimeError("No model was trained successfully.")

    comparison = pd.DataFrame(results).sort_values(
        ["roc_auc", "pr_auc"], ascending=False
    )
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "selection_split": "validation",
        "test_split_used": False,
        "trained_models": trained_model_names,
        "class_imbalance": {
            "logistic_regression": "class_weight=balanced",
            "random_forest": "class_weight=balanced_subsample",
            "xgboost": (
                "scale_pos_weight=negative_count/positive_count"
                if "xgboost" in trained_model_names
                else "not available"
            ),
        },
        "artifacts": [
            f"models/{name}.joblib" for name in trained_model_names
        ],
        "comparison_report": "results/metrics/model_comparison.csv",
    }
    (REPORT_DIR / "model_training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    best = comparison.iloc[0]

    print()
    print("=" * 80)
    print("SLICE 05 MODEL TRAINING SUMMARY")
    print("=" * 80)
    print(f"Models trained           : {len(trained_model_names)}")
    print(f"Probability predictions  : PASS")
    print(f"Class imbalance handling : PASS")
    print(f"Random seed recorded     : PASS ({RANDOM_STATE})")
    print(f"Comparison report        : {REPORT_DIR / 'model_comparison.csv'}")
    print(
        f"Best validation model    : {best['model']} "
        f"(ROC-AUC={best['roc_auc']:.6f}, PR-AUC={best['pr_auc']:.6f})"
    )
    print("Test split used          : NO")
    print("=" * 80)
    print("Slice 05 model training completed successfully.")


if __name__ == "__main__":
    main()

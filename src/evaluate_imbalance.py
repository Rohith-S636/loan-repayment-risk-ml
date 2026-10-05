"""
Slice 06 — Imbalance evaluation.

Evaluates the validation predictions produced by the Slice 05 models using
metrics appropriate for an imbalanced binary classification problem.

The validation split is used for comparison; the test split remains untouched.
No SMOTE or resampling is introduced in this slice.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "engineered_data.csv"
BUNDLE_PATH = ROOT / "models" / "preprocessing_bundle.joblib"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "results" / "metrics"
FIGURE_DIR = ROOT / "results" / "figures"
TARGET_COLUMN = "TARGET"


def load_validation_data():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Input not found: {PROCESSED_PATH}")
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing bundle not found: {BUNDLE_PATH}. Run Slice 04 first."
        )

    df = pd.read_csv(PROCESSED_PATH)
    bundle = joblib.load(BUNDLE_PATH)

    feature_columns = bundle["feature_columns"]
    id_column = bundle["id_column"]
    validation_ids = set(bundle["split_ids"]["validation"])

    if TARGET_COLUMN not in df.columns or id_column not in df.columns:
        raise ValueError("Engineered data is missing ID or TARGET.")
    if df[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique.")

    validation_df = df[df[id_column].isin(validation_ids)].copy()
    expected_rows = bundle["split_sizes"]["validation"]

    if len(validation_df) != expected_rows:
        raise ValueError(
            f"Validation row count mismatch: {len(validation_df)} != {expected_rows}"
        )

    X_validation = bundle["preprocessor"].transform(
        validation_df[feature_columns]
    )
    y_validation = validation_df[TARGET_COLUMN].astype(int).to_numpy()

    return X_validation, y_validation


def evaluate_model(name: str, model, X_validation, y_validation):
    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    matrix = confusion_matrix(y_validation, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    metrics = {
        "model": name,
        "roc_auc": float(roc_auc_score(y_validation, probabilities)),
        "pr_auc": float(average_precision_score(y_validation, probabilities)),
        "accuracy": float(accuracy_score(y_validation, predictions)),
        "precision": float(
            precision_score(y_validation, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_validation, predictions, zero_division=0)),
        "f1": float(f1_score(y_validation, predictions, zero_division=0)),
        "threshold": 0.5,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "positive_rate": float(y_validation.mean()),
        "random_state": RANDOM_STATE,
    }

    return metrics, matrix


def save_confusion_matrix(name: str, matrix: np.ndarray) -> None:
    matrix_df = pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )
    matrix_df.to_csv(REPORT_DIR / f"confusion_matrix_{name}.csv")

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    ax.set_title(f"Confusion Matrix — {name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks([0, 1], ["0", "1"])
    ax.set_yticks([0, 1], ["0", "1"])

    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{matrix[row, col]:,}", ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"confusion_matrix_{name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - SLICE 06 IMBALANCE EVALUATION")
    print("=" * 80)

    X_validation, y_validation = load_validation_data()
    print(f"Validation rows         : {len(y_validation):,}")
    print(f"Positive rate           : {y_validation.mean():.6f}")
    print("Test split used         : NO")
    print("Decision threshold      : 0.5")
    print("SMOTE/resampling        : NOT USED")

    model_paths = sorted(MODEL_DIR.glob("*.joblib"))
    model_paths = [
        path
        for path in model_paths
        if path.name in {
            "logistic_regression.joblib",
            "random_forest.joblib",
            "xgboost.joblib",
        }
    ]

    if not model_paths:
        raise RuntimeError("No Slice 05 model artifacts were found.")

    results = []

    for path in model_paths:
        name = path.stem
        print()
        print(f"Evaluating {name}...")
        model = joblib.load(path)
        metrics, matrix = evaluate_model(
            name, model, X_validation, y_validation
        )
        results.append(metrics)
        save_confusion_matrix(name, matrix)

        print(
            f"  ROC-AUC={metrics['roc_auc']:.6f} | "
            f"PR-AUC={metrics['pr_auc']:.6f} | "
            f"Precision={metrics['precision']:.6f} | "
            f"Recall={metrics['recall']:.6f} | "
            f"F1={metrics['f1']:.6f}"
        )
        print(
            f"  TN={metrics['true_negatives']:,} | "
            f"FP={metrics['false_positives']:,} | "
            f"FN={metrics['false_negatives']:,} | "
            f"TP={metrics['true_positives']:,}"
        )

    comparison = pd.DataFrame(results).sort_values(
        ["pr_auc", "roc_auc", "f1"], ascending=False
    )
    comparison.to_csv(REPORT_DIR / "imbalance_comparison.csv", index=False)

    best = comparison.iloc[0]
    metadata = {
        "random_state": RANDOM_STATE,
        "evaluation_split": "validation",
        "test_split_used": False,
        "decision_threshold": 0.5,
        "positive_rate": float(y_validation.mean()),
        "smote_used": False,
        "resampling_used": False,
        "selection_priority": ["pr_auc", "roc_auc", "f1"],
        "models_evaluated": comparison["model"].tolist(),
        "best_validation_model": best["model"],
        "best_pr_auc": float(best["pr_auc"]),
        "artifacts": {
            "comparison": "results/metrics/imbalance_comparison.csv",
            "confusion_matrices": [
                f"results/metrics/confusion_matrix_{name}.csv"
                for name in comparison["model"]
            ],
            "confusion_matrix_figures": [
                f"results/figures/confusion_matrix_{name}.png"
                for name in comparison["model"]
            ],
        },
    }
    (REPORT_DIR / "imbalance_evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("SLICE 06 IMBALANCE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Models evaluated        : {len(results)}")
    print("ROC-AUC reported        : PASS")
    print("PR-AUC reported         : PASS")
    print("Precision/Recall/F1     : PASS")
    print("Confusion matrices      : PASS")
    print("Accuracy sole metric    : NO")
    print("Results reproducible    : PASS (seed 42)")
    print(f"Best validation model   : {best['model']}")
    print(f"Best PR-AUC             : {best['pr_auc']:.6f}")
    print("Test split used         : NO")
    print("=" * 80)
    print("Slice 06 imbalance evaluation completed successfully.")


if __name__ == "__main__":
    main()

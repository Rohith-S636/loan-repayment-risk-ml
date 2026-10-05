"""
Slice 08 — Final evaluation.

Evaluate the frozen XGBoost pipeline once on the untouched test split using
the threshold selected from validation data in Slice 07. No tuning is
performed from test results.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

RANDOM_STATE = 42
MODEL_NAME = "xgboost"

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "engineered_data.csv"
BUNDLE_PATH = ROOT / "models" / "preprocessing_bundle.joblib"
MODEL_PATH = ROOT / "models" / f"{MODEL_NAME}.joblib"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
PREDICTIONS_DIR = ROOT / "results" / "predictions"

TARGET_COLUMN = "TARGET"


def load_test_data() -> tuple[object, np.ndarray, pd.Series]:
    for path in (PROCESSED_PATH, BUNDLE_PATH, MODEL_PATH, THRESHOLD_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    df = pd.read_csv(PROCESSED_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    threshold_metadata = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))

    if threshold_metadata.get("model") != MODEL_NAME:
        raise ValueError("Frozen threshold was not selected for XGBoost.")
    if threshold_metadata.get("selection_split") != "validation":
        raise ValueError("Frozen threshold must have been selected on validation data.")
    if threshold_metadata.get("test_split_used") is not False:
        raise ValueError("Slice 07 threshold metadata indicates test data was used.")

    feature_columns = bundle["feature_columns"]
    id_column = bundle["id_column"]
    test_ids = set(bundle["split_ids"]["test"])
    expected_rows = bundle["split_sizes"]["test"]

    if TARGET_COLUMN not in df.columns or id_column not in df.columns:
        raise ValueError("Engineered data is missing ID or TARGET.")
    if df[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique.")

    test_df = df[df[id_column].isin(test_ids)].copy()

    if len(test_df) != expected_rows:
        raise ValueError(
            f"Test row count mismatch: {len(test_df)} != {expected_rows}"
        )

    actual_ids = set(test_df[id_column])
    if actual_ids != test_ids:
        raise ValueError("Test applicant IDs do not match the saved Slice 04 split.")

    X_test = bundle["preprocessor"].transform(test_df[feature_columns])
    y_test = test_df[TARGET_COLUMN].astype(int).to_numpy()

    return X_test, y_test, test_df[id_column]


def save_roc_curve(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (ROC-AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Final Test ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_test_roc_curve.png", dpi=160)
    plt.close()


def save_pr_curve(y_true: np.ndarray, probabilities: np.ndarray) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    ap = average_precision_score(y_true, probabilities)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label=f"XGBoost (PR-AUC = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Final Test Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "final_test_pr_curve.png", dpi=160)
    plt.close()


def save_confusion_matrix(
    y_true: np.ndarray, predictions: np.ndarray, threshold: float
) -> None:
    matrix = confusion_matrix(y_true, predictions)

    pd.DataFrame(
        matrix,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(METRICS_DIR / "final_test_confusion_matrix.csv")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Low Risk", "High Risk"],
    )
    display.plot(values_format="d")
    display.ax_.set_title(f"Final Test Confusion Matrix (threshold = {threshold:.2f})")
    display.figure_.tight_layout()
    display.figure_.savefig(
        FIGURES_DIR / "final_test_confusion_matrix.png",
        dpi=160,
    )
    plt.close(display.figure_)


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - SLICE 08 FINAL EVALUATION")
    print("=" * 80)

    X_test, y_test, test_ids = load_test_data()
    model = joblib.load(MODEL_PATH)
    threshold_metadata = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    threshold = float(threshold_metadata["selected_threshold"])

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    metrics = {
        "model": MODEL_NAME,
        "threshold": threshold,
        "evaluation_split": "test",
        "validation_tuning_completed_before_test": True,
        "test_rows": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "random_state": RANDOM_STATE,
    }

    pd.DataFrame([metrics]).to_csv(
        METRICS_DIR / "final_test_metrics.csv",
        index=False,
    )
    (METRICS_DIR / "final_test_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    predictions_df = pd.DataFrame(
        {
            "SK_ID_CURR": test_ids.to_numpy(),
            "actual_target": y_test,
            "predicted_probability": probabilities,
            "predicted_class": predictions,
            "threshold": threshold,
        }
    )
    predictions_df.to_csv(
        PREDICTIONS_DIR / "final_test_predictions.csv",
        index=False,
    )

    save_roc_curve(y_test, probabilities)
    save_pr_curve(y_test, probabilities)
    save_confusion_matrix(y_test, predictions, threshold)

    print(f"Test rows               : {len(y_test):,}")
    print(f"Positive rate           : {y_test.mean():.6f}")
    print(f"Model evaluated         : {MODEL_NAME}")
    print(f"Frozen threshold        : {threshold:.2f}")
    print("Validation tuning       : COMPLETED BEFORE TEST")
    print("Test data used for tuning: NO")
    print()
    print("Final test metrics")
    print(f"  ROC-AUC               : {roc_auc:.6f}")
    print(f"  PR-AUC                : {pr_auc:.6f}")
    print(f"  Precision             : {precision:.6f}")
    print(f"  Recall                : {recall:.6f}")
    print(f"  F1                    : {f1:.6f}")
    print()
    print("Confusion matrix")
    print(f"  TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
    print()
    print("=" * 80)
    print("SLICE 08 FINAL EVALUATION SUMMARY")
    print("=" * 80)
    print("Final test predictions use selected model/pipeline : PASS")
    print("ROC-AUC recorded                                      : PASS")
    print("PR-AUC recorded                                       : PASS")
    print("Precision/Recall/F1 at selected threshold             : PASS")
    print("Confusion matrix recorded                             : PASS")
    print("Final figures saved                                   : PASS")
    print("Validation tuning separated from test evaluation     : PASS")
    print("Further tuning after test inspection                 : NO")
    print("=" * 80)
    print("Slice 08 final evaluation completed successfully.")


if __name__ == "__main__":
    main()

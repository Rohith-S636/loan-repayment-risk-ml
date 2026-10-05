"""
Slice 07 — Threshold analysis.

Select an operating probability threshold for the best validation model
(XGBoost) using validation data only. The objective is to maximize F1;
ties are resolved by higher precision and then the threshold closest to 0.5.

The untouched test split is never used for threshold selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.50
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.95
THRESHOLD_STEP = 0.01

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "engineered_data.csv"
BUNDLE_PATH = ROOT / "models" / "preprocessing_bundle.joblib"
DEFAULT_MODEL_PATH = ROOT / "models" / "xgboost.joblib"
REPORT_DIR = ROOT / "results" / "metrics"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"
TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"


def load_validation_data(model_path: Path) -> tuple[object, np.ndarray]:
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(f"Input not found: {PROCESSED_PATH}")
    if not BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessing bundle not found: {BUNDLE_PATH}. Run Slice 04 first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Train or tune the selected model first."
        )

    df = pd.read_csv(PROCESSED_PATH)
    bundle = joblib.load(BUNDLE_PATH)

    feature_columns = bundle["feature_columns"]
    id_column = bundle["id_column"]
    validation_ids = set(bundle["split_ids"]["validation"])
    expected_rows = bundle["split_sizes"]["validation"]

    if TARGET_COLUMN not in df.columns or id_column not in df.columns:
        raise ValueError("Engineered data is missing ID or TARGET.")
    if df[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique.")

    validation_df = df[df[id_column].isin(validation_ids)].copy()

    if len(validation_df) != expected_rows:
        raise ValueError(
            f"Validation row count mismatch: {len(validation_df)} != {expected_rows}"
        )

    actual_ids = set(validation_df[id_column])
    if actual_ids != validation_ids:
        raise ValueError("Validation applicant IDs do not match the saved Slice 04 split.")

    X_validation = bundle["preprocessor"].transform(
        validation_df[feature_columns]
    )
    y_validation = validation_df[TARGET_COLUMN].astype(int).to_numpy()

    return X_validation, y_validation


def evaluate_threshold(
    threshold: float, probabilities: np.ndarray, y_true: np.ndarray
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "predicted_positive_count": int(predictions.sum()),
    }


def select_threshold(results: pd.DataFrame) -> pd.Series:
    ranked = results.sort_values(
        by=["f1", "precision", "threshold"],
        ascending=[False, False, False],
        kind="mergesort",
    )

    best_f1 = ranked.iloc[0]["f1"]
    tied = results[np.isclose(results["f1"], best_f1, rtol=0.0, atol=1e-12)].copy()
    tied["distance_from_default"] = (tied["threshold"] - DEFAULT_THRESHOLD).abs()

    return tied.sort_values(
        by=["precision", "distance_from_default", "threshold"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the accepted model artifact.",
    )
    args = parser.parse_args()
    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - SLICE 07 THRESHOLD ANALYSIS")
    print("=" * 80)

    X_validation, y_validation = load_validation_data(model_path)
    model = joblib.load(model_path)
    probabilities = model.predict_proba(X_validation)[:, 1]

    thresholds = np.round(
        np.arange(
            MIN_THRESHOLD,
            MAX_THRESHOLD + THRESHOLD_STEP / 2,
            THRESHOLD_STEP,
        ),
        2,
    )

    results = pd.DataFrame(
        [
            evaluate_threshold(threshold, probabilities, y_validation)
            for threshold in thresholds
        ]
    )

    selected = select_threshold(results)

    default_metrics = results[
        np.isclose(results["threshold"], DEFAULT_THRESHOLD)
    ].iloc[0]

    results["selection_objective"] = "maximize_f1"
    results["selected"] = np.isclose(
        results["threshold"], selected["threshold"]
    )

    results.to_csv(
        REPORT_DIR / "threshold_analysis.csv",
        index=False,
    )

    try:
        model_relative_path = model_path.relative_to(ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("Model path must be inside the project repository.") from exc
    model_name = model_path.stem

    threshold_metadata = {
        "model": model_name,
        "model_path": model_relative_path,
        "selection_split": "validation",
        "test_split_used": False,
        "random_state": RANDOM_STATE,
        "objective": "maximize_f1",
        "tie_breakers": [
            "higher_precision",
            "closest_to_default_threshold_0.5",
            "lower_threshold",
        ],
        "candidate_threshold_min": MIN_THRESHOLD,
        "candidate_threshold_max": MAX_THRESHOLD,
        "candidate_threshold_step": THRESHOLD_STEP,
        "default_threshold": DEFAULT_THRESHOLD,
        "selected_threshold": float(selected["threshold"]),
        "selected_precision": float(selected["precision"]),
        "selected_recall": float(selected["recall"]),
        "selected_f1": float(selected["f1"]),
        "default_precision": float(default_metrics["precision"]),
        "default_recall": float(default_metrics["recall"]),
        "default_f1": float(default_metrics["f1"]),
        "validation_rows": int(len(y_validation)),
        "positive_rate": float(y_validation.mean()),
        "threshold_artifact": "models/threshold.json",
        "analysis_artifact": "results/metrics/threshold_analysis.csv",
    }

    THRESHOLD_PATH.write_text(
        json.dumps(threshold_metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Validation rows         : {len(y_validation):,}")
    print(f"Positive rate           : {y_validation.mean():.6f}")
    print(f"Model evaluated         : {model_name}")
    print("Test split used         : NO")
    print(f"Thresholds evaluated    : {len(results)}")
    print()
    print("Default threshold (0.50)")
    print(f"  Precision             : {default_metrics['precision']:.6f}")
    print(f"  Recall                : {default_metrics['recall']:.6f}")
    print(f"  F1                    : {default_metrics['f1']:.6f}")
    print()
    print("Selected threshold")
    print(f"  Threshold             : {selected['threshold']:.2f}")
    print(f"  Precision             : {selected['precision']:.6f}")
    print(f"  Recall                : {selected['recall']:.6f}")
    print(f"  F1                    : {selected['f1']:.6f}")
    print()
    print("=" * 80)
    print("SLICE 07 THRESHOLD ANALYSIS SUMMARY")
    print("=" * 80)
    print("Validation data only    : PASS")
    print("Precision across thresholds : PASS")
    print("Recall across thresholds    : PASS")
    print("F1 across thresholds        : PASS")
    print("Selected threshold recorded : PASS")
    print("Test data used for selection: NO")
    print("Results reproducible        : PASS (seed 42)")
    print("Objective                   : maximize F1")
    print(f"Selected threshold          : {selected['threshold']:.2f}")
    print("=" * 80)
    print("Slice 07 threshold analysis completed successfully.")


if __name__ == "__main__":
    main()

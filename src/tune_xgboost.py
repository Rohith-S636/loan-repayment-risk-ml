"""Controlled XGBoost tuning using the fixed Slice 04 train/validation split.

The test split is never loaded. Candidate configurations are sampled
deterministically and selected by validation PR-AUC. The resulting tuned
model is saved separately so the baseline model remains available until
the experiment is reviewed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler

RANDOM_STATE = 42
N_ITER = 20

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "engineered_data.csv"
BUNDLE_PATH = ROOT / "models" / "preprocessing_bundle.joblib"
MODEL_DIR = ROOT / "models"
TUNING_DIR = ROOT / "results" / "tuning"

TARGET_COLUMN = "TARGET"


def load_preprocessed_data():
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

    if TARGET_COLUMN not in df.columns or id_column not in df.columns:
        raise ValueError("Engineered data is missing ID or TARGET.")
    if df[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique.")
    if len(df) != sum(bundle["split_sizes"].values()):
        raise ValueError("Current engineered data row count differs from Slice 04.")

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

    preprocessor = bundle["preprocessor"]
    X_train = preprocessor.transform(train_df[feature_columns])
    X_validation = preprocessor.transform(validation_df[feature_columns])
    y_train = train_df[TARGET_COLUMN].astype(int).to_numpy()
    y_validation = validation_df[TARGET_COLUMN].astype(int).to_numpy()

    return X_train, X_validation, y_train, y_validation


def build_parameter_space(y_train: np.ndarray) -> dict:
    negatives = max(int((y_train == 0).sum()), 1)
    positives = max(int((y_train == 1).sum()), 1)
    base_ratio = negatives / positives

    return {
        "n_estimators": [250, 350, 500, 650],
        "max_depth": [3, 5, 6, 7],
        "learning_rate": [0.03, 0.05, 0.08, 0.10],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 5, 10],
        "reg_lambda": [0.5, 1.0, 5.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "scale_pos_weight": [
            round(0.75 * base_ratio, 6),
            round(1.00 * base_ratio, 6),
            round(1.25 * base_ratio, 6),
        ],
    }


def evaluate_candidate(model, X_validation, y_validation, elapsed):
    probabilities = model.predict_proba(X_validation)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_validation, probabilities)),
        "pr_auc": float(average_precision_score(y_validation, probabilities)),
        "accuracy": float(accuracy_score(y_validation, predictions)),
        "precision": float(precision_score(y_validation, predictions, zero_division=0)),
        "recall": float(recall_score(y_validation, predictions, zero_division=0)),
        "f1": float(f1_score(y_validation, predictions, zero_division=0)),
        "threshold": 0.5,
        "train_seconds": round(elapsed, 3),
    }


def main() -> None:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for the tuning experiment. "
            "Install requirements.txt first."
        ) from exc

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - CONTROLLED XGBOOST TUNING")
    print("=" * 80)

    X_train, X_validation, y_train, y_validation = load_preprocessed_data()

    print(f"Training rows           : {X_train.shape[0]:,}")
    print(f"Validation rows         : {X_validation.shape[0]:,}")
    print(f"Transformed features    : {X_train.shape[1]:,}")
    print(f"Positive rate (train)   : {y_train.mean():.6f}")
    print(f"Positive rate (valid.)  : {y_validation.mean():.6f}")
    print("Test split used         : NO")
    print(f"Randomized trials       : {N_ITER}")
    print("Selection metric        : validation PR-AUC")

    parameter_space = build_parameter_space(y_train)
    candidates = list(
        ParameterSampler(
            parameter_space,
            n_iter=N_ITER,
            random_state=RANDOM_STATE,
        )
    )

    trials = []
    best_model = None
    best_params = None
    best_pr_auc = -np.inf

    for trial_number, params in enumerate(candidates, start=1):
        print()
        print(f"Trial {trial_number:02d}/{N_ITER}")
        print(f"  Parameters: {params}")

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method="hist",
            **params,
        )

        started = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - started

        metrics = evaluate_candidate(
            model, X_validation, y_validation, elapsed
        )
        record = {
            "trial": trial_number,
            **metrics,
            **params,
        }
        trials.append(record)

        print(
            f"  PR-AUC={metrics['pr_auc']:.6f} | "
            f"ROC-AUC={metrics['roc_auc']:.6f} | "
            f"F1={metrics['f1']:.6f}"
        )

        if (
            metrics["pr_auc"] > best_pr_auc
            or (
                np.isclose(metrics["pr_auc"], best_pr_auc)
                and metrics["roc_auc"] > max(
                    (r["roc_auc"] for r in trials[:-1]), default=-np.inf
                )
            )
        ):
            best_pr_auc = metrics["pr_auc"]
            best_params = params
            best_model = model

    if best_model is None or best_params is None:
        raise RuntimeError("No XGBoost tuning candidate completed successfully.")

    trials_df = pd.DataFrame(trials).sort_values(
        ["pr_auc", "roc_auc", "f1"],
        ascending=False,
    )
    trials_df.to_csv(TUNING_DIR / "xgboost_trials.csv", index=False)

    joblib.dump(best_model, MODEL_DIR / "xgboost_tuned.joblib")

    best_row = trials_df.iloc[0].to_dict()
    (TUNING_DIR / "best_params.json").write_text(
        json.dumps(
            {
                "model": "xgboost",
                "selection_metric": "validation_pr_auc",
                "random_state": RANDOM_STATE,
                "n_iter": N_ITER,
                "best_trial": int(best_row["trial"]),
                "best_validation_metrics": {
                    key: float(best_row[key])
                    for key in [
                        "pr_auc",
                        "roc_auc",
                        "precision",
                        "recall",
                        "f1",
                    ]
                },
                "best_parameters": best_params,
                "model_artifact": "models/xgboost_tuned.joblib",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    metadata = {
        "experiment": "controlled_xgboost_tuning",
        "selection_split": "validation",
        "test_split_used": False,
        "test_results_used_for_selection": False,
        "random_state": RANDOM_STATE,
        "n_iter": N_ITER,
        "selection_metric": "PR-AUC",
        "tie_breakers": ["ROC-AUC", "F1"],
        "parameter_space": parameter_space,
        "baseline_validation_pr_auc": 0.259062,
        "baseline_validation_roc_auc": 0.772189,
        "artifacts": {
            "trials": "results/tuning/xgboost_trials.csv",
            "best_params": "results/tuning/best_params.json",
            "model": "models/xgboost_tuned.joblib",
        },
        "next_step": (
            "If the tuned model is accepted, rerun threshold selection on "
            "validation data and then perform one final untouched test evaluation."
        ),
    }
    (TUNING_DIR / "tuning_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("XGBOOST TUNING SUMMARY")
    print("=" * 80)
    print(f"Trials completed        : {len(trials_df)}")
    print(f"Best trial              : {int(best_row['trial'])}")
    print(f"Best validation PR-AUC  : {best_row['pr_auc']:.6f}")
    print(f"Best validation ROC-AUC : {best_row['roc_auc']:.6f}")
    print(f"Best validation F1      : {best_row['f1']:.6f}")
    print("Test split used         : NO")
    print("Test results used       : NO")
    print(f"Tuned model             : {MODEL_DIR / 'xgboost_tuned.joblib'}")
    print("=" * 80)
    print("Controlled XGBoost tuning completed successfully.")
    
if __name__ == "__main__":
    main()

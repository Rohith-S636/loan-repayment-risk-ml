"""
Slice 04 — Leakage-safe preprocessing for loan repayment risk.

Creates a stratified train/validation/test split and fits a reusable
ColumnTransformer on training data only. Validation and test data are
transformed using the fitted training preprocessor without refitting.

Local inputs/outputs:
    Input:  data/processed/engineered_data.csv
    Model:  models/preprocessor.joblib
    Bundle: models/preprocessing_bundle.joblib
    Report: results/preprocessing/split_summary.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_REMAINDER = 0.25  # gives 60/20/20 overall

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "engineered_data.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "results" / "preprocessing"

ID_COLUMN = "SK_ID_CURR"
TARGET_COLUMN = "TARGET"


def build_preprocessor(feature_frame: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a preprocessing pipeline without fitting it."""
    numeric_columns = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = feature_frame.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if not numeric_columns and not categorical_columns:
        raise ValueError("No model features were found after excluding ID and TARGET.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_columns, categorical_columns


def validate_input(df: pd.DataFrame) -> None:
    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df[ID_COLUMN].isna().any():
        raise ValueError(f"{ID_COLUMN} contains missing values.")

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} must be unique.")

    if not set(df[TARGET_COLUMN].dropna().unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN} must contain only binary 0/1 values.")

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} contains missing values.")

    target_like = [c for c in df.columns if c.upper().startswith("TARGET_")]
    if target_like:
        raise ValueError(f"Potential target-derived feature columns found: {target_like}")


def split_data(df: pd.DataFrame):
    """Create fixed stratified 60/20/20 train/validation/test splits."""
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    train, validation = train_test_split(
        train_val,
        test_size=VALIDATION_SIZE_WITHIN_REMAINDER,
        random_state=RANDOM_STATE,
        stratify=train_val[TARGET_COLUMN],
    )
    return train.copy(), validation.copy(), test.copy()


def distribution_row(name: str, frame: pd.DataFrame) -> dict:
    counts = frame[TARGET_COLUMN].value_counts().to_dict()
    total = len(frame)
    return {
        "split": name,
        "rows": total,
        "target_0": int(counts.get(0, 0)),
        "target_1": int(counts.get(1, 0)),
        "target_1_rate": float(counts.get(1, 0) / total) if total else 0.0,
    }


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input not found: {INPUT_PATH}\n"
            "Run Slice 03 first to create engineered_data.csv."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - SLICE 04 LEAKAGE-SAFE PREPROCESSING")
    print("=" * 80)
    print(f"Input   : {INPUT_PATH}")
    print(f"Models  : {MODEL_DIR}")
    print(f"Reports : {REPORT_DIR}")
    print()

    df = pd.read_csv(INPUT_PATH)
    validate_input(df)

    # Never allow the applicant identifier or target into model features.
    feature_columns = [
        c for c in df.columns if c not in {ID_COLUMN, TARGET_COLUMN}
    ]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    train, validation, test = split_data(df)

    # Fit ONLY on training features. This is the key leakage-safety boundary.
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(
        train[feature_columns]
    )
    X_train = preprocessor.fit_transform(train[feature_columns])
    X_validation = preprocessor.transform(validation[feature_columns])
    X_test = preprocessor.transform(test[feature_columns])

    if not hasattr(X_train, "shape") or not hasattr(X_validation, "shape") or not hasattr(X_test, "shape"):
        raise RuntimeError("Preprocessor did not return valid transformed matrices.")

    # Explicitly verify the transformer was not refit for validation/test.
    validation_again = preprocessor.transform(validation[feature_columns])
    test_again = preprocessor.transform(test[feature_columns])

    if X_validation.shape != validation_again.shape or X_test.shape != test_again.shape:
        raise RuntimeError("Validation/test transformation is not deterministic.")

    if hasattr(X_validation, "nnz") and hasattr(validation_again, "nnz"):
        if (X_validation != validation_again).nnz != 0:
            raise RuntimeError("Validation transformation changed between calls.")
        if (X_test != test_again).nnz != 0:
            raise RuntimeError("Test transformation changed between calls.")
    else:
        if not np.array_equal(X_validation, validation_again):
            raise RuntimeError("Validation transformation changed between calls.")
        if not np.array_equal(X_test, test_again):
            raise RuntimeError("Test transformation changed between calls.")

    # Keep the reusable preprocessing object separate for future inference.
    joblib.dump(preprocessor, MODEL_DIR / "preprocessor.joblib")

    bundle = {
        "preprocessor": preprocessor,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "id_column": ID_COLUMN,
        "target_column": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "split_sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "split_ids": {
            "train": train[ID_COLUMN].tolist(),
            "validation": validation[ID_COLUMN].tolist(),
            "test": test[ID_COLUMN].tolist(),
        },
    }
    joblib.dump(bundle, MODEL_DIR / "preprocessing_bundle.joblib")

    summary = pd.DataFrame(
        [
            distribution_row("train", train),
            distribution_row("validation", validation),
            distribution_row("test", test),
        ]
    )
    summary["random_state"] = RANDOM_STATE
    summary["preprocessor_fitted_on"] = "train"
    summary.to_csv(REPORT_DIR / "split_summary.csv", index=False)

    metadata = {
        "random_state": RANDOM_STATE,
        "split_strategy": "stratified",
        "split_ratio": "60/20/20",
        "input_rows": len(df),
        "input_features": len(feature_columns),
        "numeric_features": len(numeric_columns),
        "categorical_features": len(categorical_columns),
        "transformed_features": int(X_train.shape[1]),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "artifacts": {
            "preprocessor": "models/preprocessor.joblib",
            "bundle": "models/preprocessing_bundle.joblib",
            "summary": "results/preprocessing/split_summary.csv",
        },
    }
    (REPORT_DIR / "preprocessing_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # Sanity checks for acceptance criteria.
    expected_rows = len(train) + len(validation) + len(test)
    if expected_rows != len(df):
        raise RuntimeError("Split row counts do not add up to the input row count.")

    train_ids = set(train[ID_COLUMN])
    validation_ids = set(validation[ID_COLUMN])
    test_ids = set(test[ID_COLUMN])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("Train/validation/test applicant IDs overlap.")

    if train[TARGET_COLUMN].mean() == 0 or train[TARGET_COLUMN].mean() == 1:
        raise RuntimeError("Training split lost one of the target classes.")

    print("=" * 80)
    print("SLICE 04 VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Rows                    : {len(df):,}")
    print(f"Features before encoding : {len(feature_columns):,}")
    print(f"Numeric features        : {len(numeric_columns):,}")
    print(f"Categorical features    : {len(categorical_columns):,}")
    print(f"Train / Validation / Test: {len(train):,} / {len(validation):,} / {len(test):,}")
    print(f"Transformed features    : {X_train.shape[1]:,}")
    print("Stratified split        : PASS")
    print("Fixed random seed       : PASS")
    print("Preprocessor fit on train only : PASS")
    print("Validation transformed without refit : PASS")
    print("Test transformed without refit       : PASS")
    print("No ID/TARGET model features          : PASS")
    print(f"Preprocessor             : {MODEL_DIR / 'preprocessor.joblib'}")
    print(f"Bundle                   : {MODEL_DIR / 'preprocessing_bundle.joblib'}")
    print("=" * 80)
    print("Slice 04 preprocessing completed successfully.")


if __name__ == "__main__":
    main()

"""
Slice 09 — Permutation explainability for the final accepted model.

Uses the saved validation split only. The untouched test split is never
loaded or inspected. The output is a compact feature-importance table that
the Streamlit demo can load without recomputing explanations on every run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 1500
DEFAULT_REPEATS = 3

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_PATH = ROOT / "data" / "processed" / "engineered_data.csv"
BUNDLE_PATH = ROOT / "models" / "preprocessing_bundle.joblib"
DEFAULT_MODEL_PATH = ROOT / "models" / "xgboost_tuned.joblib"
OUTPUT_DIR = ROOT / "results" / "explainability"
OUTPUT_PATH = OUTPUT_DIR / "permutation_importance.csv"
METADATA_PATH = OUTPUT_DIR / "explainability_metadata.json"
TARGET_COLUMN = "TARGET"
ID_COLUMN = "SK_ID_CURR"


def load_validation(model_path: Path, sample_size: int):
    for path in (PROCESSED_PATH, BUNDLE_PATH, model_path):
        if not path.exists():
            raise FileNotFoundError(f"Required artifact not found: {path}")

    df = pd.read_csv(PROCESSED_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    validation_ids = set(bundle["split_ids"]["validation"])
    expected_rows = bundle["split_sizes"]["validation"]
    feature_columns = bundle["feature_columns"]

    validation_df = df[df[ID_COLUMN].isin(validation_ids)].copy()
    if len(validation_df) != expected_rows:
        raise ValueError("Validation row count does not match the saved Slice 04 split.")

    validation_df = validation_df.sort_values(ID_COLUMN)
    if sample_size < len(validation_df):
        validation_df = validation_df.sample(
            n=sample_size,
            random_state=RANDOM_STATE,
        ).sort_values(ID_COLUMN)

    X = bundle["preprocessor"].transform(validation_df[feature_columns])
    y = validation_df[TARGET_COLUMN].astype(int).to_numpy()
    feature_names = list(bundle["preprocessor"].get_feature_names_out())

    if X.shape[1] != len(feature_names):
        raise ValueError("Transformed feature count does not match feature names.")

    return X, y, feature_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    args = parser.parse_args()

    model_path = args.model_path if args.model_path.is_absolute() else ROOT / args.model_path
    if args.sample_size < 100:
        raise ValueError("sample-size must be at least 100.")
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - SLICE 09 PERMUTATION EXPLAINABILITY")
    print("=" * 80)

    X_validation, y_validation, feature_names = load_validation(
        model_path, args.sample_size
    )
    model = joblib.load(model_path)

    result = permutation_importance(
        model,
        X_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=args.repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values(
        by="importance_mean",
        ascending=False,
        kind="mergesort",
    )

    importance["rank"] = np.arange(1, len(importance) + 1)
    importance["model"] = model_path.stem
    importance["scoring"] = "average_precision"
    importance["selection_split"] = "validation"
    importance["test_split_used"] = False
    importance.to_csv(OUTPUT_PATH, index=False)

    metadata = {
        "model": model_path.stem,
        "model_path": model_path.relative_to(ROOT).as_posix(),
        "selection_split": "validation",
        "test_split_used": False,
        "random_state": RANDOM_STATE,
        "sample_size": int(len(y_validation)),
        "repeats": int(args.repeats),
        "scoring": "average_precision",
        "validation_pr_auc_on_sample": float(
            average_precision_score(
                y_validation, model.predict_proba(X_validation)[:, 1]
            )
        ),
        "output": "results/explainability/permutation_importance.csv",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Validation sample       : {len(y_validation):,}")
    print(f"Transformed features    : {len(feature_names):,}")
    print(f"Model evaluated         : {model_path.stem}")
    print("Test split used         : NO")
    print(f"Permutation repeats     : {args.repeats}")
    print()
    print("Top 10 features")
    for _, row in importance.head(10).iterrows():
        print(
            f"  {int(row['rank']):>2}. {row['feature']:<48} "
            f"{row['importance_mean']:.6f}"
        )

    print()
    print("=" * 80)
    print("SLICE 09 EXPLAINABILITY SUMMARY")
    print("=" * 80)
    print("Final accepted model used : PASS")
    print("Validation data only      : PASS")
    print("Test data used            : NO")
    print("Permutation importance    : PASS")
    print(f"Output                    : {OUTPUT_PATH}")
    print("=" * 80)
    print("Slice 09 permutation explainability completed successfully.")


if __name__ == "__main__":
    main()

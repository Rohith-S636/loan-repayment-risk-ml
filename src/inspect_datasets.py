"""
Slice 01 - Dataset Validation and Inspection

Validates the seven Home Credit datasets used by the loan repayment
risk prediction project.

The script:

1. Verifies that all required raw CSV files exist.
2. Inspects row/column counts.
3. Validates required identifier and target columns.
4. Checks duplicate identifier records where applicable.
5. Checks primary/foreign-key relationships between tables.
6. Generates missing-value reports.
7. Generates a dataset summary report.
8. Writes all validation outputs to results/dataset_inspection/.

Raw datasets are expected to remain local and must not be committed to Git.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------

DATASETS: Dict[str, Dict[str, object]] = {
    "application_train": {
        "file": "application_train.csv",
        "required_columns": ["SK_ID_CURR", "TARGET"],
        "primary_key": ["SK_ID_CURR"],
    },
    "bureau": {
        "file": "bureau.csv",
        "required_columns": ["SK_ID_CURR", "SK_ID_BUREAU"],
        "primary_key": ["SK_ID_BUREAU"],
    },
    "bureau_balance": {
        "file": "bureau_balance.csv",
        "required_columns": ["SK_ID_BUREAU"],
        "primary_key": [],
    },
    "previous_application": {
        "file": "previous_application.csv",
        "required_columns": ["SK_ID_CURR", "SK_ID_PREV"],
        "primary_key": ["SK_ID_PREV"],
    },
    "POS_CASH_balance": {
        "file": "POS_CASH_balance.csv",
        "required_columns": ["SK_ID_CURR", "SK_ID_PREV"],
        "primary_key": [],
    },
    "installments_payments": {
        "file": "installments_payments.csv",
        "required_columns": ["SK_ID_CURR", "SK_ID_PREV"],
        "primary_key": [],
    },
    "credit_card_balance": {
        "file": "credit_card_balance.csv",
        "required_columns": ["SK_ID_CURR", "SK_ID_PREV"],
        "primary_key": [],
    },
}


# ---------------------------------------------------------------------------
# Relationship configuration
# ---------------------------------------------------------------------------

# Each relationship is:
# (parent dataset, parent key, child dataset, child key)
#
# Relationship status:
# - PASS    : all non-null child keys have a matching parent key.
# - WARNING : relationship can be evaluated, but some child keys are
#             unmatched. This is informational and does not fail Slice 01.
# - FAIL    : the required parent or child key column is missing.

RELATIONSHIPS: List[Tuple[str, str, str, str]] = [
    (
        "application_train",
        "SK_ID_CURR",
        "bureau",
        "SK_ID_CURR",
    ),
    (
        "bureau",
        "SK_ID_BUREAU",
        "bureau_balance",
        "SK_ID_BUREAU",
    ),
    (
        "application_train",
        "SK_ID_CURR",
        "previous_application",
        "SK_ID_CURR",
    ),
    (
        "previous_application",
        "SK_ID_PREV",
        "POS_CASH_balance",
        "SK_ID_PREV",
    ),
    (
        "previous_application",
        "SK_ID_PREV",
        "installments_payments",
        "SK_ID_PREV",
    ),
    (
        "previous_application",
        "SK_ID_PREV",
        "credit_card_balance",
        "SK_ID_PREV",
    ),
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def load_dataset(
    dataset_name: str,
    config: Dict[str, object],
) -> pd.DataFrame:
    """Load a dataset and return it as a pandas DataFrame."""

    file_name = str(config["file"])
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )

    print(f"Loading {file_name}...")

    return pd.read_csv(file_path)


def validate_required_columns(
    dataset_name: str,
    dataframe: pd.DataFrame,
    required_columns: List[str],
) -> Tuple[bool, List[str]]:
    """Validate that all required columns exist."""

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        print(
            f"[FAIL] {dataset_name}: missing columns "
            f"{missing_columns}"
        )

        return False, missing_columns

    print(
        f"[PASS] {dataset_name}: required columns present"
    )

    return True, []


def check_duplicate_keys(
    dataset_name: str,
    dataframe: pd.DataFrame,
    primary_key: List[str],
) -> Dict[str, object]:
    """
    Check duplicate records for datasets with a declared primary key.

    Historical tables such as bureau_balance, POS_CASH_balance,
    installments_payments and credit_card_balance intentionally do not
    have a unique applicant-level primary key because they contain
    multiple historical records.
    """

    if not primary_key:
        return {
            "checked": False,
            "duplicate_rows": 0,
            "status": "NOT_APPLICABLE",
        }

    missing = [
        column
        for column in primary_key
        if column not in dataframe.columns
    ]

    if missing:
        return {
            "checked": False,
            "duplicate_rows": 0,
            "status": "FAIL",
        }

    duplicate_mask = dataframe.duplicated(
        subset=primary_key,
        keep=False,
    )

    duplicate_rows = int(duplicate_mask.sum())

    if duplicate_rows == 0:
        status = "PASS"

        print(
            f"[PASS] {dataset_name}: no duplicate "
            f"{primary_key} records"
        )

    else:
        status = "WARNING"

        print(
            f"[WARNING] {dataset_name}: "
            f"{duplicate_rows:,} rows participate in duplicate "
            f"{primary_key} records"
        )

    return {
        "checked": True,
        "duplicate_rows": duplicate_rows,
        "status": status,
    }


def generate_missing_value_report(
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> Path:
    """Generate a missing-value report for one dataset."""

    missing_count = dataframe.isna().sum()

    missing_percentage = (
        missing_count / len(dataframe) * 100
        if len(dataframe) > 0
        else 0
    )

    report = pd.DataFrame(
        {
            "column": dataframe.columns,
            "missing_count": missing_count.values,
            "missing_percentage": missing_percentage.values,
            "dtype": [
                str(dataframe[column].dtype)
                for column in dataframe.columns
            ],
        }
    )

    report = report.sort_values(
        by="missing_count",
        ascending=False,
    )

    output_path = (
        OUTPUT_DIR
        / f"{dataset_name}_missing_values.csv"
    )

    report.to_csv(output_path, index=False)

    return output_path


def validate_relationship(
    parent_name: str,
    parent_key: str,
    child_name: str,
    child_key: str,
    datasets: Dict[str, pd.DataFrame],
) -> Dict[str, object]:
    """
    Validate a parent-child foreign-key relationship.

    Only non-null child keys are considered.

    Historical tables may legitimately contain repeated child-key values.

    Status semantics:
    - PASS: all checked child keys have matching parent keys.
    - WARNING: relationship is evaluable but some child keys are unmatched.
    - FAIL: required parent or child key column is missing.
    """

    parent_df = datasets[parent_name]
    child_df = datasets[child_name]

    if parent_key not in parent_df.columns:
        return {
            "parent": parent_name,
            "parent_key": parent_key,
            "child": child_name,
            "child_key": child_key,
            "child_rows_checked": 0,
            "unmatched_rows": 0,
            "unmatched_percentage": 0.0,
            "status": "FAIL",
        }

    if child_key not in child_df.columns:
        return {
            "parent": parent_name,
            "parent_key": parent_key,
            "child": child_name,
            "child_key": child_key,
            "child_rows_checked": 0,
            "unmatched_rows": 0,
            "unmatched_percentage": 0.0,
            "status": "FAIL",
        }

    parent_keys = set(
        parent_df[parent_key].dropna().unique()
    )

    child_keys = child_df[child_key].dropna()

    unmatched_mask = ~child_keys.isin(parent_keys)

    unmatched_rows = int(unmatched_mask.sum())
    checked_rows = int(len(child_keys))

    unmatched_percentage = (
        unmatched_rows / checked_rows * 100
        if checked_rows > 0
        else 0.0
    )

    if unmatched_rows == 0:
        status = "PASS"

        print(
            f"[PASS] {child_name}.{child_key} -> "
            f"{parent_name}.{parent_key}"
        )

    else:
        status = "WARNING"

        print(
            f"[WARNING] {child_name}.{child_key} -> "
            f"{parent_name}.{parent_key}: "
            f"{unmatched_rows:,} unmatched rows "
            f"({unmatched_percentage:.4f}%)"
        )

    return {
        "parent": parent_name,
        "parent_key": parent_key,
        "child": child_name,
        "child_key": child_key,
        "child_rows_checked": checked_rows,
        "unmatched_rows": unmatched_rows,
        "unmatched_percentage": round(
            unmatched_percentage,
            4,
        ),
        "status": status,
    }


def build_dataset_summary(
    datasets: Dict[str, pd.DataFrame],
    validation_results: Dict[str, Dict[str, object]],
) -> pd.DataFrame:
    """Build the overall dataset inspection summary."""

    rows = []

    for dataset_name, dataframe in datasets.items():
        config = DATASETS[dataset_name]

        missing_cells = int(
            dataframe.isna().sum().sum()
        )

        rows.append(
            {
                "dataset": dataset_name,
                "file": config["file"],
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
                "numeric_columns": len(
                    dataframe.select_dtypes(
                        include="number"
                    ).columns
                ),
                "categorical_columns": len(
                    dataframe.select_dtypes(
                        exclude="number"
                    ).columns
                ),
                "missing_cells": missing_cells,
                "required_columns_valid": validation_results[
                    dataset_name
                ]["required_columns_valid"],
                "duplicate_key_status": validation_results[
                    dataset_name
                ]["duplicate_key_status"],
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------


def main() -> None:
    """Run complete Slice 01 dataset validation."""

    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - SLICE 01 DATASET VALIDATION")
    print("=" * 80)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # -----------------------------------------------------------------------
    # 1. Check all required files exist
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("1. CHECKING REQUIRED DATASETS")
    print("-" * 80)

    missing_files = []

    for dataset_name, config in DATASETS.items():
        file_path = DATA_DIR / str(config["file"])

        if file_path.exists():
            print(
                f"[PASS] {dataset_name}: "
                f"{file_path.name}"
            )
        else:
            print(
                f"[FAIL] {dataset_name}: "
                f"{file_path.name}"
            )

            missing_files.append(file_path)

    if missing_files:
        raise FileNotFoundError(
            "One or more required datasets are missing:\n"
            + "\n".join(str(path) for path in missing_files)
        )

    # -----------------------------------------------------------------------
    # 2. Load datasets
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("2. LOADING DATASETS")
    print("-" * 80)

    datasets: Dict[str, pd.DataFrame] = {}

    for dataset_name, config in DATASETS.items():
        datasets[dataset_name] = load_dataset(
            dataset_name,
            config,
        )

    # -----------------------------------------------------------------------
    # 3. Validate required columns and keys
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("3. VALIDATING REQUIRED COLUMNS AND KEYS")
    print("-" * 80)

    validation_results: Dict[str, Dict[str, object]] = {}

    for dataset_name, dataframe in datasets.items():
        config = DATASETS[dataset_name]

        required_columns_valid, _ = validate_required_columns(
            dataset_name,
            dataframe,
            list(config["required_columns"]),
        )

        duplicate_result = check_duplicate_keys(
            dataset_name,
            dataframe,
            list(config["primary_key"]),
        )

        validation_results[dataset_name] = {
            "required_columns_valid": required_columns_valid,
            "duplicate_key_status": duplicate_result["status"],
            "duplicate_key_rows": duplicate_result[
                "duplicate_rows"
            ],
        }

    # -----------------------------------------------------------------------
    # 4. Generate missing-value reports
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("4. GENERATING MISSING-VALUE REPORTS")
    print("-" * 80)

    for dataset_name, dataframe in datasets.items():
        output_path = generate_missing_value_report(
            dataset_name,
            dataframe,
        )

        print(
            f"[PASS] {dataset_name}: "
            f"{output_path.name}"
        )

    # -----------------------------------------------------------------------
    # 5. Validate table relationships
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("5. VALIDATING TABLE RELATIONSHIPS")
    print("-" * 80)

    relationship_results = []

    for relationship in RELATIONSHIPS:
        result = validate_relationship(
            *relationship,
            datasets,
        )

        relationship_results.append(result)

    relationship_report = pd.DataFrame(
        relationship_results
    )

    relationship_output = (
        OUTPUT_DIR / "relationship_validation.csv"
    )

    relationship_report.to_csv(
        relationship_output,
        index=False,
    )

    print(
        f"\nRelationship report saved to: "
        f"{relationship_output}"
    )

    # -----------------------------------------------------------------------
    # 6. Generate overall dataset summary
    # -----------------------------------------------------------------------

    print("\n" + "-" * 80)
    print("6. GENERATING DATASET SUMMARY")
    print("-" * 80)

    summary = build_dataset_summary(
        datasets,
        validation_results,
    )

    summary_output = (
        OUTPUT_DIR / "dataset_summary.csv"
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print(
        f"Dataset summary saved to: "
        f"{summary_output}"
    )

    # -----------------------------------------------------------------------
    # 7. Final validation status
    # -----------------------------------------------------------------------

    required_columns_pass = all(
        result["required_columns_valid"]
        for result in validation_results.values()
    )

    relationships_fail = any(
        result["status"] == "FAIL"
        for result in relationship_results
    )

    relationships_warn = any(
        result["status"] == "WARNING"
        for result in relationship_results
    )

    # Relationship warnings are informational.
    # Only an unevaluable relationship is a hard failure.
    relationships_pass = not relationships_fail

    all_checks_pass = (
        required_columns_pass
        and relationships_pass
    )

    if relationships_fail:
        relationships_status = "FAIL"
    elif relationships_warn:
        relationships_status = "WARNING"
    else:
        relationships_status = "PASS"

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(
        f"Datasets checked       : {len(datasets)}"
    )

    print(
        f"Required columns       : "
        f"{'PASS' if required_columns_pass else 'FAIL'}"
    )

    print(
        f"Relationships          : "
        f"{relationships_status}"
    )

    print(
        f"Overall Slice 01       : "
        f"{'PASS' if all_checks_pass else 'FAIL'}"
    )

    print("=" * 80)

    if not all_checks_pass:
        raise RuntimeError(
            "Slice 01 validation failed. "
            "Review the validation output before continuing."
        )

    print(
        "\nSlice 01 dataset validation completed successfully."
    )


if __name__ == "__main__":
    main()
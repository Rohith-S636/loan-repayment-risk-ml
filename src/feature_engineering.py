"""
Slice 03 - Feature Engineering

Create a compact, domain-driven feature table from the Slice 02
applicant-level aggregates plus selected application_train variables.

Design goals:
- financial affordability ratios
- bureau credit/debt/overdue ratios
- previous-application approval/refusal behaviour
- POS/CASH delinquency features
- installment repayment/delay features
- credit-card utilization/delinquency features
- safe division without inf values
- no target-derived features
- one row per SK_ID_CURR
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "results" / "feature_engineering"

INPUT_FILE = PROCESSED_DIR / "modeling_data.csv"
OUTPUT_FILE = PROCESSED_DIR / "engineered_data.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


APPLICATION_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide safely; zero denominators and non-finite results become NaN."""
    denominator = denominator.replace(0, np.nan)
    result = numerator.div(denominator)
    return result.replace([np.inf, -np.inf], np.nan)


def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    table: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{table}: missing required columns: {missing}")


def add_ratio(
    dataframe: pd.DataFrame,
    name: str,
    numerator: str,
    denominator: str,
) -> bool:
    """Add a ratio only when both source columns are present."""
    if numerator not in dataframe.columns or denominator not in dataframe.columns:
        return False
    dataframe[name] = safe_divide(
        dataframe[numerator],
        dataframe[denominator],
    )
    return True


def add_application_features(
    modeling: pd.DataFrame,
    application: pd.DataFrame,
) -> list[str]:
    """Join selected current-application variables and derive affordability features."""
    available = [
        column for column in APPLICATION_FEATURES
        if column in application.columns
    ]

    selected = application[["SK_ID_CURR", *available]].copy()

    duplicate_ids = selected["SK_ID_CURR"].duplicated().any()
    if duplicate_ids:
        raise ValueError("application_train.SK_ID_CURR must be unique")

    before = len(modeling)
    modeling.merge(
        selected,
        on="SK_ID_CURR",
        how="left",
        validate="one_to_one",
        suffixes=("", "_APPLICATION"),
    )

    # Merge in-place through column assignment to keep the modeling table stable.
    application_indexed = selected.set_index("SK_ID_CURR")
    for column in available:
        modeling[column] = modeling["SK_ID_CURR"].map(application_indexed[column])

    if len(modeling) != before or not modeling["SK_ID_CURR"].is_unique:
        raise RuntimeError("Application feature join changed applicant population")

    created: list[str] = []

    ratios = [
        ("APP_CREDIT_TO_INCOME", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
        ("APP_ANNUITY_TO_INCOME", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        ("APP_GOODS_PRICE_TO_INCOME", "AMT_GOODS_PRICE", "AMT_INCOME_TOTAL"),
        ("APP_CREDIT_TO_GOODS_PRICE", "AMT_CREDIT", "AMT_GOODS_PRICE"),
        ("APP_ANNUITY_TO_CREDIT", "AMT_ANNUITY", "AMT_CREDIT"),
    ]

    for name, numerator, denominator in ratios:
        if add_ratio(modeling, name, numerator, denominator):
            created.append(name)

    if "DAYS_BIRTH" in modeling.columns:
        modeling["APP_AGE_YEARS"] = (
            modeling["DAYS_BIRTH"].abs() / 365.25
        )
        created.append("APP_AGE_YEARS")

    if "DAYS_EMPLOYED" in modeling.columns:
        employed = modeling["DAYS_EMPLOYED"].where(
            modeling["DAYS_EMPLOYED"] < 0
        )
        modeling["APP_EMPLOYED_YEARS"] = employed.abs() / 365.25
        created.append("APP_EMPLOYED_YEARS")

    if "AMT_INCOME_TOTAL" in modeling.columns and "CNT_FAM_MEMBERS" in modeling.columns:
        modeling["APP_INCOME_PER_FAMILY_MEMBER"] = safe_divide(
            modeling["AMT_INCOME_TOTAL"],
            modeling["CNT_FAM_MEMBERS"],
        )
        created.append("APP_INCOME_PER_FAMILY_MEMBER")

    return created


def add_bureau_features(dataframe: pd.DataFrame) -> list[str]:
    created: list[str] = []

    ratio_specs = [
        (
            "BUREAU_DEBT_TO_CREDIT",
            "BUREAU_AMT_CREDIT_SUM_DEBT_SUM",
            "BUREAU_AMT_CREDIT_SUM_SUM",
        ),
        (
            "BUREAU_OVERDUE_TO_CREDIT",
            "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM",
            "BUREAU_AMT_CREDIT_SUM_SUM",
        ),
        (
            "BUREAU_OVERDUE_ACCOUNTS_RATIO",
            "BUREAU_AMT_CREDIT_SUM_OVERDUE_MEAN",
            "BUREAU_AMT_CREDIT_SUM_MEAN",
        ),
    ]

    for name, numerator, denominator in ratio_specs:
        if add_ratio(dataframe, name, numerator, denominator):
            created.append(name)

    if "BUREAU_CREDIT_ACTIVE_ACTIVE_COUNT" in dataframe.columns:
        if "BUREAU_ACCOUNT_COUNT" in dataframe.columns:
            dataframe["BUREAU_ACTIVE_ACCOUNT_RATIO"] = safe_divide(
                dataframe["BUREAU_CREDIT_ACTIVE_ACTIVE_COUNT"],
                dataframe["BUREAU_ACCOUNT_COUNT"],
            )
            created.append("BUREAU_ACTIVE_ACCOUNT_RATIO")

    if "BUREAU_CREDIT_ACTIVE_CLOSED_COUNT" in dataframe.columns:
        if "BUREAU_ACCOUNT_COUNT" in dataframe.columns:
            dataframe["BUREAU_CLOSED_ACCOUNT_RATIO"] = safe_divide(
                dataframe["BUREAU_CREDIT_ACTIVE_CLOSED_COUNT"],
                dataframe["BUREAU_ACCOUNT_COUNT"],
            )
            created.append("BUREAU_CLOSED_ACCOUNT_RATIO")

    return created


def add_previous_features(dataframe: pd.DataFrame) -> list[str]:
    created: list[str] = []

    approved = "PREV_STATUS_APPROVED_COUNT"
    refused = "PREV_STATUS_REFUSED_COUNT"
    total = "PREV_APPLICATION_COUNT"

    if approved in dataframe.columns and total in dataframe.columns:
        dataframe["PREV_APPROVAL_RATIO"] = safe_divide(
            dataframe[approved],
            dataframe[total],
        )
        created.append("PREV_APPROVAL_RATIO")

    if refused in dataframe.columns and total in dataframe.columns:
        dataframe["PREV_REFUSAL_RATIO"] = safe_divide(
            dataframe[refused],
            dataframe[total],
        )
        created.append("PREV_REFUSAL_RATIO")

    if (
        "PREV_STATUS_CANCELED_COUNT" in dataframe.columns
        and total in dataframe.columns
    ):
        dataframe["PREV_CANCELLATION_RATIO"] = safe_divide(
            dataframe["PREV_STATUS_CANCELED_COUNT"],
            dataframe[total],
        )
        created.append("PREV_CANCELLATION_RATIO")

    if (
        "PREV_AMT_CREDIT_MEAN" in dataframe.columns
        and "PREV_AMT_APPLICATION_MEAN" in dataframe.columns
    ):
        if add_ratio(
            dataframe,
            "PREV_CREDIT_TO_APPLICATION_RATIO",
            "PREV_AMT_CREDIT_MEAN",
            "PREV_AMT_APPLICATION_MEAN",
        ):
            created.append("PREV_CREDIT_TO_APPLICATION_RATIO")

    return created


def add_pos_features(dataframe: pd.DataFrame) -> list[str]:
    created: list[str] = []

    if "POS_DPD_MAX" in dataframe.columns:
        dataframe["POS_SEVERE_DPD_FLAG"] = (
            dataframe["POS_DPD_MAX"].fillna(0) >= 30
        ).astype("int8")
        created.append("POS_SEVERE_DPD_FLAG")

    if "POS_DPD_DEF_MAX" in dataframe.columns:
        dataframe["POS_SEVERE_DPD_DEF_FLAG"] = (
            dataframe["POS_DPD_DEF_MAX"].fillna(0) >= 30
        ).astype("int8")
        created.append("POS_SEVERE_DPD_DEF_FLAG")

    if (
        "POS_DPD_MEAN" in dataframe.columns
        and "POS_DPD_DEF_MEAN" in dataframe.columns
    ):
        dataframe["POS_DPD_TO_DEFAULT_DPD_RATIO"] = safe_divide(
            dataframe["POS_DPD_MEAN"],
            dataframe["POS_DPD_DEF_MEAN"],
        )
        created.append("POS_DPD_TO_DEFAULT_DPD_RATIO")

    return created


def add_installment_features(dataframe: pd.DataFrame) -> list[str]:
    created: list[str] = []

    if "PAYMENT_DELAY_MAX" in dataframe.columns:
        dataframe["INST_LATE_30D_FLAG"] = (
            dataframe["PAYMENT_DELAY_MAX"].fillna(0) >= 30
        ).astype("int8")
        created.append("INST_LATE_30D_FLAG")

    if "PAYMENT_DELAY_MEAN" in dataframe.columns:
        dataframe["INST_AVG_DELAY_POSITIVE"] = (
            dataframe["PAYMENT_DELAY_MEAN"].clip(lower=0)
        )
        created.append("INST_AVG_DELAY_POSITIVE")

    if (
        "PAYMENT_AMOUNT_SUM" in dataframe.columns
        and "INSTALMENT_AMOUNT_SUM" in dataframe.columns
    ):
        if add_ratio(
            dataframe,
            "INST_PAYMENT_COVERAGE_RATIO",
            "PAYMENT_AMOUNT_SUM",
            "INSTALMENT_AMOUNT_SUM",
        ):
            created.append("INST_PAYMENT_COVERAGE_RATIO")

    return created


def add_credit_card_features(dataframe: pd.DataFrame) -> list[str]:
    created: list[str] = []

    if "CC_UTILIZATION_RATIO" in dataframe.columns:
        dataframe["CC_HIGH_UTILIZATION_FLAG"] = (
            dataframe["CC_UTILIZATION_RATIO"].fillna(0) >= 0.75
        ).astype("int8")
        created.append("CC_HIGH_UTILIZATION_FLAG")

    if "CC_DPD_MAX" in dataframe.columns:
        dataframe["CC_LATE_30D_FLAG"] = (
            dataframe["CC_DPD_MAX"].fillna(0) >= 30
        ).astype("int8")
        created.append("CC_LATE_30D_FLAG")

    if (
        "CC_PAYMENT_MEAN" in dataframe.columns
        and "CC_MIN_PAYMENT_MEAN" in dataframe.columns
    ):
        if add_ratio(
            dataframe,
            "CC_PAYMENT_TO_MIN_PAYMENT_RATIO",
            "CC_PAYMENT_MEAN",
            "CC_MIN_PAYMENT_MEAN",
        ):
            created.append("CC_PAYMENT_TO_MIN_PAYMENT_RATIO")

    return created


def build_features() -> tuple[pd.DataFrame, list[str]]:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Slice 02 output not found: {INPUT_FILE}. "
            "Run src/aggregate_history.py first."
        )

    modeling = pd.read_csv(INPUT_FILE)
    require_columns(
        modeling,
        ["SK_ID_CURR", "TARGET"],
        "modeling_data",
    )

    if not modeling["SK_ID_CURR"].is_unique:
        raise RuntimeError("modeling_data must contain one row per SK_ID_CURR")

    application_path = RAW_DIR / "application_train.csv"
    if not application_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {application_path}"
        )

    application = pd.read_csv(
        application_path,
        usecols=lambda column: column in {
            "SK_ID_CURR",
            *APPLICATION_FEATURES,
        },
    )

    require_columns(
        application,
        ["SK_ID_CURR"],
        "application_train",
    )

    created: list[str] = []
    created.extend(add_application_features(modeling, application))
    created.extend(add_bureau_features(modeling))
    created.extend(add_previous_features(modeling))
    created.extend(add_pos_features(modeling))
    created.extend(add_installment_features(modeling))
    created.extend(add_credit_card_features(modeling))

    # Remove accidental non-finite values while retaining missing values
    # for Slice 04's leakage-safe preprocessing.
    numeric_columns = modeling.select_dtypes(include=[np.number]).columns
    modeling[numeric_columns] = modeling[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    target_columns = [
        column for column in modeling.columns
        if column.upper().startswith("TARGET_")
    ]
    if target_columns:
        raise RuntimeError(
            f"Target-derived columns found: {target_columns}"
        )

    if len(modeling) != len(application) or not modeling["SK_ID_CURR"].is_unique:
        raise RuntimeError("Feature engineering changed applicant population")

    return modeling, created


def write_reports(
    engineered: pd.DataFrame,
    created_features: list[str],
) -> None:
    feature_report = pd.DataFrame(
        {
            "feature": created_features,
            "created": True,
            "source_stage": "slice_03",
        }
    )
    feature_report.to_csv(
        REPORT_DIR / "feature_catalog.csv",
        index=False,
    )

    missing_report = pd.DataFrame(
        {
            "column": engineered.columns,
            "dtype": [
                str(engineered[column].dtype)
                for column in engineered.columns
            ],
            "missing_count": [
                int(engineered[column].isna().sum())
                for column in engineered.columns
            ],
            "missing_percentage": [
                round(
                    engineered[column].isna().mean() * 100,
                    4,
                )
                for column in engineered.columns
            ],
        }
    )
    missing_report.to_csv(
        REPORT_DIR / "feature_missingness.csv",
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "rows": len(engineered),
                "columns": len(engineered.columns),
                "new_features": len(created_features),
                "unique_applicants": engineered["SK_ID_CURR"].nunique(),
                "target_present": "TARGET" in engineered.columns,
                "target_derived_columns": 0,
                "non_finite_numeric_values": int(
                    np.isinf(
                        engineered.select_dtypes(include=[np.number])
                    ).sum().sum()
                ),
            }
        ]
    )
    summary.to_csv(
        REPORT_DIR / "feature_engineering_summary.csv",
        index=False,
    )


def main() -> None:
    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - SLICE 03 FEATURE ENGINEERING")
    print("=" * 80)
    print(f"Input         : {INPUT_FILE}")
    print(f"Output        : {OUTPUT_FILE}")
    print(f"Reports       : {REPORT_DIR}")

    engineered, created_features = build_features()

    engineered.to_csv(OUTPUT_FILE, index=False)
    write_reports(engineered, created_features)

    print("\n" + "=" * 80)
    print("SLICE 03 VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Rows                    : {len(engineered):,}")
    print(f"Columns                 : {len(engineered.columns):,}")
    print(f"New features            : {len(created_features):,}")
    print(
        "One row per applicant   : "
        f"{'PASS' if engineered['SK_ID_CURR'].is_unique else 'FAIL'}"
    )
    print(
        "TARGET preserved        : "
        f"{'PASS' if 'TARGET' in engineered.columns else 'FAIL'}"
    )
    print("TARGET used in features : NO")
    print(
        "Non-finite values       : "
        f"{int(np.isinf(engineered.select_dtypes(include=[np.number])).sum().sum())}"
    )
    print(f"Output                  : {OUTPUT_FILE}")
    print("=" * 80)

    if not engineered["SK_ID_CURR"].is_unique:
        raise RuntimeError("Slice 03 validation failed: duplicate applicants")

    if "TARGET" not in engineered.columns:
        raise RuntimeError("Slice 03 validation failed: TARGET missing")

    if any(column.upper().startswith("TARGET_") for column in engineered.columns):
        raise RuntimeError("Slice 03 validation failed: target-derived feature found")

    print("Slice 03 feature engineering completed successfully.")


if __name__ == "__main__":
    main()

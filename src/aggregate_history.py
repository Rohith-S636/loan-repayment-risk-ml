"""
Slice 02 - Historical Aggregation

Reduce the six historical Home Credit tables to applicant-level features
and join them to application_train without row explosion or target leakage.

The script is intentionally sequential/chunked for the large history tables.
Raw CSV files stay local; only the processed applicant-level table and small
aggregation reports are produced under data/processed/ and results/aggregation/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "results" / "aggregation"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 500_000
OUTPUT_FILE = PROCESSED_DIR / "modeling_data.csv"


def require_file(name: str) -> Path:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return path


def require_columns(df: pd.DataFrame, columns: Iterable[str], table: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing required columns: {missing}")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, pd.NA))


def flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            f"{prefix}{name}_{stat}".upper()
            for name, stat in df.columns
        ]
    else:
        df.columns = [f"{prefix}{c}" for c in df.columns]
    return df


def aggregate_bureau_balance() -> pd.DataFrame:
    """
    Aggregate monthly bureau balances to SK_ID_BUREAU.

    The resulting table is small enough to merge into bureau.csv before the
    second aggregation to SK_ID_CURR.
    """
    path = require_file("bureau_balance.csv")
    partials: List[pd.DataFrame] = []

    usecols = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]

    for chunk_no, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE),
        start=1,
    ):
        chunk["STATUS"] = chunk["STATUS"].fillna("MISSING").astype(str)

        grouped = chunk.groupby("SK_ID_BUREAU").agg(
            BB_MONTH_COUNT=("MONTHS_BALANCE", "count"),
            BB_MONTH_MIN=("MONTHS_BALANCE", "min"),
            BB_MONTH_MAX=("MONTHS_BALANCE", "max"),
        )

        status_counts = pd.crosstab(
            chunk["SK_ID_BUREAU"],
            chunk["STATUS"],
        )
        status_counts.columns = [
            f"BB_STATUS_{str(c).upper()}_COUNT"
            for c in status_counts.columns
        ]

        partials.append(grouped.join(status_counts, how="outer"))

        print(f"[INFO] bureau_balance chunk {chunk_no} processed")

    result = pd.concat(partials, axis=0).groupby(level=0).sum()
    result.index.name = "SK_ID_BUREAU"
    result = result.reset_index()

    print(f"[PASS] bureau_balance -> {len(result):,} bureau accounts")
    return result


def aggregate_bureau(bureau_balance_features: pd.DataFrame) -> pd.DataFrame:
    path = require_file("bureau.csv")
    bureau = pd.read_csv(path)

    require_columns(
        bureau,
        ["SK_ID_CURR", "SK_ID_BUREAU"],
        "bureau",
    )

    bureau = bureau.merge(
        bureau_balance_features,
        on="SK_ID_BUREAU",
        how="left",
        validate="one_to_one",
    )

    numeric_specs: Dict[str, List[str]] = {
        "DAYS_CREDIT": ["mean", "min", "max"],
        "DAYS_CREDIT_ENDDATE": ["mean", "min", "max"],
        "DAYS_ENDDATE_FACT": ["mean", "min", "max"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_LIMIT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean", "max"],
        "CREDIT_DAY_OVERDUE": ["sum", "mean", "max"],
        "CNT_CREDIT_PROLONG": ["sum", "mean", "max"],
        "DAYS_CREDIT_UPDATE": ["mean", "min", "max"],
    }

    aggregations = {
        column: stats
        for column, stats in numeric_specs.items()
        if column in bureau.columns
    }

    result = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_ACCOUNT_COUNT=("SK_ID_BUREAU", "count"),
        **{
            column: pd.NamedAgg(column=column, aggfunc=stats[0])
            for column, stats in aggregations.items()
        },
    )

    # Add the remaining requested statistics without relying on pandas
    # NamedAgg's handling of dynamically generated names.
    if aggregations:
        numeric_extra = bureau.groupby("SK_ID_CURR").agg(aggregations)
        numeric_extra = flatten_columns(numeric_extra, "BUREAU_")
        result = result.join(numeric_extra, how="left")

    if "CREDIT_ACTIVE" in bureau.columns:
        active_counts = pd.crosstab(
            bureau["SK_ID_CURR"],
            bureau["CREDIT_ACTIVE"].fillna("MISSING").astype(str),
        )
        active_counts.columns = [
            f"BUREAU_CREDIT_ACTIVE_{str(c).upper()}_COUNT"
            for c in active_counts.columns
        ]
        result = result.join(active_counts, how="left")

    balance_numeric = [
        c
        for c in bureau_balance_features.columns
        if c != "SK_ID_BUREAU" and pd.api.types.is_numeric_dtype(
            bureau_balance_features[c]
        )
    ]

    balance_agg = bureau.groupby("SK_ID_CURR")[balance_numeric].sum(
        min_count=1
    )
    balance_agg.columns = [
        f"BUREAU_{c}" for c in balance_agg.columns
    ]
    result = result.join(balance_agg, how="left")

    result = result.reset_index()

    # The first dynamically generated statistic above is retained for
    # compactness; all features are applicant-level.
    assert result["SK_ID_CURR"].is_unique

    print(f"[PASS] bureau -> {len(result):,} applicants")
    return result


def aggregate_previous_application() -> pd.DataFrame:
    path = require_file("previous_application.csv")
    usecols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "AMT_ANNUITY",
        "AMT_APPLICATION",
        "AMT_CREDIT",
        "AMT_DOWN_PAYMENT",
        "AMT_GOODS_PRICE",
        "RATE_DOWN_PAYMENT",
        "DAYS_DECISION",
        "CNT_PAYMENT",
        "NAME_CONTRACT_STATUS",
    ]

    df = pd.read_csv(path, usecols=usecols)
    require_columns(
        df,
        ["SK_ID_CURR", "SK_ID_PREV"],
        "previous_application",
    )

    grouped = df.groupby("SK_ID_CURR").agg(
        PREV_APPLICATION_COUNT=("SK_ID_PREV", "count"),
        PREV_APPLICATION_UNIQUE_COUNT=("SK_ID_PREV", "nunique"),
        PREV_AMT_ANNUITY_MEAN=("AMT_ANNUITY", "mean"),
        PREV_AMT_APPLICATION_MEAN=("AMT_APPLICATION", "mean"),
        PREV_AMT_APPLICATION_SUM=("AMT_APPLICATION", "sum"),
        PREV_AMT_CREDIT_MEAN=("AMT_CREDIT", "mean"),
        PREV_AMT_CREDIT_SUM=("AMT_CREDIT", "sum"),
        PREV_AMT_DOWN_PAYMENT_MEAN=("AMT_DOWN_PAYMENT", "mean"),
        PREV_AMT_GOODS_PRICE_MEAN=("AMT_GOODS_PRICE", "mean"),
        PREV_RATE_DOWN_PAYMENT_MEAN=("RATE_DOWN_PAYMENT", "mean"),
        PREV_DAYS_DECISION_MEAN=("DAYS_DECISION", "mean"),
        PREV_CNT_PAYMENT_MEAN=("CNT_PAYMENT", "mean"),
    )

    if "NAME_CONTRACT_STATUS" in df.columns:
        status_counts = pd.crosstab(
            df["SK_ID_CURR"],
            df["NAME_CONTRACT_STATUS"].fillna("MISSING").astype(str),
        )
        status_counts.columns = [
            f"PREV_STATUS_{str(c).upper()}_COUNT"
            for c in status_counts.columns
        ]
        grouped = grouped.join(status_counts, how="left")

    grouped = grouped.reset_index()
    assert grouped["SK_ID_CURR"].is_unique

    print(f"[PASS] previous_application -> {len(grouped):,} applicants")
    return grouped


def aggregate_pos_cash() -> pd.DataFrame:
    path = require_file("POS_CASH_balance.csv")
    usecols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "NAME_CONTRACT_STATUS",
        "SK_DPD",
        "SK_DPD_DEF",
    ]

    partials: List[pd.DataFrame] = []

    for chunk_no, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE),
        start=1,
    ):
        chunk["POS_LATE_FLAG"] = (chunk["SK_DPD"].fillna(0) > 0).astype(int)

        grouped = chunk.groupby("SK_ID_CURR").agg(
            POS_RECORD_COUNT=("SK_ID_PREV", "count"),
            POS_PREV_LOAN_COUNT=("SK_ID_PREV", "nunique"),
            POS_MONTHS_MEAN=("MONTHS_BALANCE", "mean"),
            POS_MONTHS_MIN=("MONTHS_BALANCE", "min"),
            POS_MONTHS_MAX=("MONTHS_BALANCE", "max"),
            POS_INSTALLMENT_MEAN=("CNT_INSTALMENT", "mean"),
            POS_FUTURE_INSTALLMENT_MEAN=("CNT_INSTALMENT_FUTURE", "mean"),
            POS_DPD_MEAN=("SK_DPD", "mean"),
            POS_DPD_MAX=("SK_DPD", "max"),
            POS_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
            POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
            POS_LATE_COUNT=("POS_LATE_FLAG", "sum"),
        )

        partials.append(grouped)

        print(f"[INFO] POS_CASH chunk {chunk_no} processed")

    result = pd.concat(partials).groupby(level=0).agg(
        {
            "POS_RECORD_COUNT": "sum",
            "POS_PREV_LOAN_COUNT": "sum",
            "POS_MONTHS_MEAN": "mean",
            "POS_MONTHS_MIN": "min",
            "POS_MONTHS_MAX": "max",
            "POS_INSTALLMENT_MEAN": "mean",
            "POS_FUTURE_INSTALLMENT_MEAN": "mean",
            "POS_DPD_MEAN": "mean",
            "POS_DPD_MAX": "max",
            "POS_DPD_DEF_MEAN": "mean",
            "POS_DPD_DEF_MAX": "max",
            "POS_LATE_COUNT": "sum",
        }
    )

    result["POS_LATE_RATIO"] = safe_divide(
        result["POS_LATE_COUNT"],
        result["POS_RECORD_COUNT"],
    )

    result = result.reset_index()

    assert result["SK_ID_CURR"].is_unique

    print(f"[PASS] POS_CASH_balance -> {len(result):,} applicants")
    return result


def aggregate_installments() -> pd.DataFrame:
    path = require_file("installments_payments.csv")
    usecols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]

    partials: List[pd.DataFrame] = []

    for chunk_no, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE),
        start=1,
    ):
        chunk["PAYMENT_DELAY"] = (
            chunk["DAYS_ENTRY_PAYMENT"] - chunk["DAYS_INSTALMENT"]
        )
        chunk["LATE_FLAG"] = (chunk["PAYMENT_DELAY"].fillna(0) > 0).astype(int)
        chunk["PAYMENT_GAP"] = (
            chunk["AMT_PAYMENT"] - chunk["AMT_INSTALMENT"]
        )

        grouped = chunk.groupby("SK_ID_CURR").agg(
            INSTALMENT_RECORD_COUNT=("SK_ID_PREV", "count"),
            INSTALMENT_PREV_LOAN_COUNT=("SK_ID_PREV", "nunique"),
            INSTALMENT_AMOUNT_SUM=("AMT_INSTALMENT", "sum"),
            PAYMENT_AMOUNT_SUM=("AMT_PAYMENT", "sum"),
            PAYMENT_DELAY_MEAN=("PAYMENT_DELAY", "mean"),
            PAYMENT_DELAY_MAX=("PAYMENT_DELAY", "max"),
            PAYMENT_GAP_MEAN=("PAYMENT_GAP", "mean"),
            LATE_PAYMENT_COUNT=("LATE_FLAG", "sum"),
        )

        partials.append(grouped)

        print(f"[INFO] installments chunk {chunk_no} processed")

    result = pd.concat(partials).groupby(level=0).agg(
        {
            "INSTALMENT_RECORD_COUNT": "sum",
            "INSTALMENT_PREV_LOAN_COUNT": "sum",
            "INSTALMENT_AMOUNT_SUM": "sum",
            "PAYMENT_AMOUNT_SUM": "sum",
            "PAYMENT_DELAY_MEAN": "mean",
            "PAYMENT_DELAY_MAX": "max",
            "PAYMENT_GAP_MEAN": "mean",
            "LATE_PAYMENT_COUNT": "sum",
        }
    )

    result["PAYMENT_TO_INSTALMENT_RATIO"] = safe_divide(
        result["PAYMENT_AMOUNT_SUM"],
        result["INSTALMENT_AMOUNT_SUM"],
    )
    result["LATE_PAYMENT_RATIO"] = safe_divide(
        result["LATE_PAYMENT_COUNT"],
        result["INSTALMENT_RECORD_COUNT"],
    )

    result = result.reset_index()

    assert result["SK_ID_CURR"].is_unique

    print(f"[PASS] installments_payments -> {len(result):,} applicants")
    return result


def aggregate_credit_card() -> pd.DataFrame:
    path = require_file("credit_card_balance.csv")
    usecols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "MONTHS_BALANCE",
        "AMT_BALANCE",
        "AMT_CREDIT_LIMIT_ACTUAL",
        "AMT_DRAWINGS_ATM_CURRENT",
        "AMT_DRAWINGS_CURRENT",
        "AMT_PAYMENT_CURRENT",
        "AMT_INST_MIN_REGULARITY",
        "SK_DPD",
        "SK_DPD_DEF",
    ]

    partials: List[pd.DataFrame] = []

    for chunk_no, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE),
        start=1,
    ):
        chunk["CC_LATE_FLAG"] = (chunk["SK_DPD"].fillna(0) > 0).astype(int)

        grouped = chunk.groupby("SK_ID_CURR").agg(
            CC_RECORD_COUNT=("SK_ID_PREV", "count"),
            CC_PREV_CARD_COUNT=("SK_ID_PREV", "nunique"),
            CC_MONTHS_MEAN=("MONTHS_BALANCE", "mean"),
            CC_BALANCE_MEAN=("AMT_BALANCE", "mean"),
            CC_BALANCE_MAX=("AMT_BALANCE", "max"),
            CC_CREDIT_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
            CC_DRAWINGS_ATM_MEAN=("AMT_DRAWINGS_ATM_CURRENT", "mean"),
            CC_DRAWINGS_MEAN=("AMT_DRAWINGS_CURRENT", "mean"),
            CC_PAYMENT_MEAN=("AMT_PAYMENT_CURRENT", "mean"),
            CC_MIN_PAYMENT_MEAN=("AMT_INST_MIN_REGULARITY", "mean"),
            CC_DPD_MEAN=("SK_DPD", "mean"),
            CC_DPD_MAX=("SK_DPD", "max"),
            CC_DPD_DEF_MEAN=("SK_DPD_DEF", "mean"),
            CC_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
            CC_LATE_COUNT=("CC_LATE_FLAG", "sum"),
        )

        partials.append(grouped)

        print(f"[INFO] credit_card chunk {chunk_no} processed")

    result = pd.concat(partials).groupby(level=0).agg(
        {
            "CC_RECORD_COUNT": "sum",
            "CC_PREV_CARD_COUNT": "sum",
            "CC_MONTHS_MEAN": "mean",
            "CC_BALANCE_MEAN": "mean",
            "CC_BALANCE_MAX": "max",
            "CC_CREDIT_LIMIT_MEAN": "mean",
            "CC_DRAWINGS_ATM_MEAN": "mean",
            "CC_DRAWINGS_MEAN": "mean",
            "CC_PAYMENT_MEAN": "mean",
            "CC_MIN_PAYMENT_MEAN": "mean",
            "CC_DPD_MEAN": "mean",
            "CC_DPD_MAX": "max",
            "CC_DPD_DEF_MEAN": "mean",
            "CC_DPD_DEF_MAX": "max",
            "CC_LATE_COUNT": "sum",
        }
    )

    result["CC_LATE_RATIO"] = safe_divide(
        result["CC_LATE_COUNT"],
        result["CC_RECORD_COUNT"],
    )
    result["CC_UTILIZATION_RATIO"] = safe_divide(
        result["CC_BALANCE_MEAN"],
        result["CC_CREDIT_LIMIT_MEAN"],
    )

    result = result.reset_index()

    assert result["SK_ID_CURR"].is_unique

    print(f"[PASS] credit_card_balance -> {len(result):,} applicants")
    return result


def build_modeling_table() -> pd.DataFrame:
    application_path = require_file("application_train.csv")
    application = pd.read_csv(
        application_path,
        usecols=["SK_ID_CURR", "TARGET"],
    )

    require_columns(
        application,
        ["SK_ID_CURR", "TARGET"],
        "application_train",
    )

    if not application["SK_ID_CURR"].is_unique:
        raise ValueError("application_train.SK_ID_CURR must be unique")

    bureau_balance = aggregate_bureau_balance()
    bureau = aggregate_bureau(bureau_balance)
    previous = aggregate_previous_application()
    pos = aggregate_pos_cash()
    installments = aggregate_installments()
    credit_card = aggregate_credit_card()

    feature_tables = {
        "bureau": bureau,
        "previous_application": previous,
        "POS_CASH_balance": pos,
        "installments_payments": installments,
        "credit_card_balance": credit_card,
    }

    result = application.copy()

    for name, table in feature_tables.items():
        before = len(result)

        if table["SK_ID_CURR"].duplicated().any():
            raise ValueError(
                f"{name} aggregation contains duplicate SK_ID_CURR values"
            )

        result = result.merge(
            table,
            on="SK_ID_CURR",
            how="left",
            validate="one_to_one",
        )

        if len(result) != before:
            raise RuntimeError(
                f"Row count changed after joining {name}: "
                f"{before:,} -> {len(result):,}"
            )

    if len(result) != len(application):
        raise RuntimeError(
            "Final applicant population changed during aggregation"
        )

    if not result["SK_ID_CURR"].is_unique:
        raise RuntimeError("Final SK_ID_CURR is not unique")

    if not result["TARGET"].equals(application["TARGET"]):
        raise RuntimeError("TARGET changed during aggregation")

    # Historical aggregation must never create a feature from TARGET.
    target_dependent_columns = [
        c for c in result.columns
        if c.upper().startswith("TARGET_")
    ]
    if target_dependent_columns:
        raise RuntimeError(
            f"Target-derived columns found: {target_dependent_columns}"
        )

    return result


def write_reports(modeling_data: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        [
            {
                "output": OUTPUT_FILE.name,
                "rows": len(modeling_data),
                "columns": len(modeling_data.columns),
                "unique_applicants": modeling_data["SK_ID_CURR"].nunique(),
                "target_present": "TARGET" in modeling_data.columns,
                "target_missing": int(modeling_data["TARGET"].isna().sum()),
            }
        ]
    )
    summary.to_csv(
        REPORT_DIR / "aggregation_summary.csv",
        index=False,
    )

    row_counts = pd.DataFrame(
        [
            {
                "stage": "application_train",
                "rows": len(modeling_data),
                "unique_SK_ID_CURR": modeling_data["SK_ID_CURR"].nunique(),
            },
            {
                "stage": "final_modeling_data",
                "rows": len(modeling_data),
                "unique_SK_ID_CURR": modeling_data["SK_ID_CURR"].nunique(),
            },
        ]
    )
    row_counts.to_csv(
        REPORT_DIR / "aggregation_row_counts.csv",
        index=False,
    )

    schema = pd.DataFrame(
        {
            "column": modeling_data.columns,
            "dtype": [
                str(modeling_data[c].dtype)
                for c in modeling_data.columns
            ],
        }
    )
    schema.to_csv(
        REPORT_DIR / "aggregation_schema.csv",
        index=False,
    )


def main() -> None:
    print("=" * 80)
    print("LOAN REPAYMENT RISK ML - SLICE 02 HISTORICAL AGGREGATION")
    print("=" * 80)
    print(f"Raw data      : {RAW_DIR}")
    print(f"Processed data: {PROCESSED_DIR}")
    print(f"Reports       : {REPORT_DIR}")
    print(f"Chunk size    : {CHUNK_SIZE:,}")

    modeling_data = build_modeling_table()

    modeling_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    write_reports(modeling_data)

    print("\n" + "=" * 80)
    print("SLICE 02 VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Final rows            : {len(modeling_data):,}")
    print(
        f"Unique SK_ID_CURR     : "
        f"{modeling_data['SK_ID_CURR'].nunique():,}"
    )
    print(
        f"One row per applicant  : "
        f"{'PASS' if modeling_data['SK_ID_CURR'].is_unique else 'FAIL'}"
    )
    print(
        f"TARGET preserved       : "
        f"{'PASS' if modeling_data['TARGET'].notna().all() else 'FAIL'}"
    )
    print(f"Output                : {OUTPUT_FILE}")
    print("=" * 80)
    print("Slice 02 aggregation completed successfully.")


if __name__ == "__main__":
    main()

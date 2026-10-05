"""
Slice 02 - Historical Aggregation

Reduce the six historical Home Credit tables to applicant-level features
and join them to application_train without row explosion or target leakage.

Large historical tables are processed sequentially in chunks. Raw CSV files
remain local; the final applicant-level table is written to data/processed/.
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


def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    table: str,
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{table}: missing required columns: {missing}")


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return numerator.div(denominator.where(denominator != 0))


def flatten_columns(
    dataframe: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    if isinstance(dataframe.columns, pd.MultiIndex):
        dataframe.columns = [
            f"{prefix}{name}_{stat}".upper()
            for name, stat in dataframe.columns
        ]
    else:
        dataframe.columns = [
            f"{prefix}{column}".upper()
            for column in dataframe.columns
        ]
    return dataframe


def aggregate_bureau_balance() -> pd.DataFrame:
    """Reduce bureau_balance from monthly rows to one row per bureau loan."""

    path = require_file("bureau_balance.csv")
    partials: List[pd.DataFrame] = []

    usecols = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]

    for chunk_no, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE),
        start=1,
    ):
        chunk["STATUS"] = chunk["STATUS"].fillna("MISSING").astype(str)

        grouped = chunk.groupby("SK_ID_BUREAU").agg(
            BB_MONTH_COUNT=("MONTHS_BALANCE", "size"),
        )

        status_counts = pd.crosstab(
            chunk["SK_ID_BUREAU"],
            chunk["STATUS"],
        )
        status_counts.columns = [
            f"BB_STATUS_{str(value).upper()}_COUNT"
            for value in status_counts.columns
        ]

        partials.append(grouped.join(status_counts, how="outer"))
        print(f"[INFO] bureau_balance chunk {chunk_no} processed")

    result = (
        pd.concat(partials, axis=0)
        .groupby(level=0)
        .sum(min_count=1)
        .reset_index()
    )
    result = result.rename(columns={"index": "SK_ID_BUREAU"})

    if not result["SK_ID_BUREAU"].is_unique:
        raise RuntimeError("bureau_balance aggregation is not unique by SK_ID_BUREAU")

    print(f"[PASS] bureau_balance -> {len(result):,} bureau accounts")
    return result


def aggregate_bureau(
    bureau_balance_features: pd.DataFrame,
) -> pd.DataFrame:
    """Reduce bureau plus bureau-balance features to one row per applicant."""

    bureau = pd.read_csv(require_file("bureau.csv"))

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
    )

    if aggregations:
        numeric_features = bureau.groupby("SK_ID_CURR").agg(aggregations)
        numeric_features = flatten_columns(numeric_features, "BUREAU_")
        result = result.join(numeric_features, how="left")

    if "CREDIT_ACTIVE" in bureau.columns:
        active_counts = pd.crosstab(
            bureau["SK_ID_CURR"],
            bureau["CREDIT_ACTIVE"].fillna("MISSING").astype(str),
        )
        active_counts.columns = [
            f"BUREAU_CREDIT_ACTIVE_{str(value).upper()}_COUNT"
            for value in active_counts.columns
        ]
        result = result.join(active_counts, how="left")

    balance_columns = [
        column
        for column in bureau_balance_features.columns
        if column != "SK_ID_BUREAU"
    ]

    if balance_columns:
        balance_features = bureau.groupby("SK_ID_CURR")[
            balance_columns
        ].sum(min_count=1)
        balance_features.columns = [
            f"BUREAU_{column}".upper()
            for column in balance_features.columns
        ]
        result = result.join(balance_features, how="left")

    result = result.reset_index()

    if not result["SK_ID_CURR"].is_unique:
        raise RuntimeError("bureau aggregation is not unique by SK_ID_CURR")

    print(f"[PASS] bureau -> {len(result):,} applicants")
    return result


def aggregate_previous_application() -> pd.DataFrame:
    """Reduce previous applications to one row per applicant."""

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

    dataframe = pd.read_csv(
        require_file("previous_application.csv"),
        usecols=usecols,
    )

    require_columns(
        dataframe,
        ["SK_ID_CURR", "SK_ID_PREV"],
        "previous_application",
    )

    result = dataframe.groupby("SK_ID_CURR").agg(
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

    status_counts = pd.crosstab(
        dataframe["SK_ID_CURR"],
        dataframe["NAME_CONTRACT_STATUS"].fillna("MISSING").astype(str),
    )
    status_counts.columns = [
        f"PREV_STATUS_{str(value).upper()}_COUNT"
        for value in status_counts.columns
    ]
    result = result.join(status_counts, how="left").reset_index()

    if not result["SK_ID_CURR"].is_unique:
        raise RuntimeError(
            "previous_application aggregation is not unique by SK_ID_CURR"
        )

    print(
        f"[PASS] previous_application -> {len(result):,} applicants"
    )
    return result


def aggregate_pos_cash() -> pd.DataFrame:
    """Chunk-aggregate POS/CASH history directly to applicant level."""

    usecols = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "MONTHS_BALANCE",
        "CNT_INSTALMENT",
        "CNT_INSTALMENT_FUTURE",
        "SK_DPD",
        "SK_DPD_DEF",
    ]

    partials: List[pd.DataFrame] = []

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            require_file("POS_CASH_balance.csv"),
            usecols=usecols,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk["POS_LATE_FLAG"] = (
            chunk["SK_DPD"].fillna(0) > 0
        ).astype("int8")

        grouped = chunk.groupby("SK_ID_CURR").agg(
            POS_RECORD_COUNT=("SK_ID_PREV", "size"),
            POS_MONTHS_SUM=("MONTHS_BALANCE", "sum"),
            POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            POS_MONTHS_MIN=("MONTHS_BALANCE", "min"),
            POS_MONTHS_MAX=("MONTHS_BALANCE", "max"),
            POS_INSTALLMENT_SUM=("CNT_INSTALMENT", "sum"),
            POS_INSTALLMENT_COUNT=("CNT_INSTALMENT", "count"),
            POS_FUTURE_INSTALLMENT_SUM=(
                "CNT_INSTALMENT_FUTURE",
                "sum",
            ),
            POS_FUTURE_INSTALLMENT_COUNT=(
                "CNT_INSTALMENT_FUTURE",
                "count",
            ),
            POS_DPD_SUM=("SK_DPD", "sum"),
            POS_DPD_COUNT=("SK_DPD", "count"),
            POS_DPD_MAX=("SK_DPD", "max"),
            POS_DPD_DEF_SUM=("SK_DPD_DEF", "sum"),
            POS_DPD_DEF_COUNT=("SK_DPD_DEF", "count"),
            POS_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
            POS_LATE_COUNT=("POS_LATE_FLAG", "sum"),
        )

        partials.append(grouped)
        print(f"[INFO] POS_CASH chunk {chunk_no} processed")

    combined = pd.concat(partials).groupby(level=0).agg(
        {
            "POS_RECORD_COUNT": "sum",
            "POS_MONTHS_SUM": "sum",
            "POS_MONTHS_COUNT": "sum",
            "POS_MONTHS_MIN": "min",
            "POS_MONTHS_MAX": "max",
            "POS_INSTALLMENT_SUM": "sum",
            "POS_INSTALLMENT_COUNT": "sum",
            "POS_FUTURE_INSTALLMENT_SUM": "sum",
            "POS_FUTURE_INSTALLMENT_COUNT": "sum",
            "POS_DPD_SUM": "sum",
            "POS_DPD_COUNT": "sum",
            "POS_DPD_MAX": "max",
            "POS_DPD_DEF_SUM": "sum",
            "POS_DPD_DEF_COUNT": "sum",
            "POS_DPD_DEF_MAX": "max",
            "POS_LATE_COUNT": "sum",
        }
    )

    combined["POS_MONTHS_MEAN"] = safe_divide(
        combined["POS_MONTHS_SUM"],
        combined["POS_MONTHS_COUNT"],
    )
    combined["POS_INSTALLMENT_MEAN"] = safe_divide(
        combined["POS_INSTALLMENT_SUM"],
        combined["POS_INSTALLMENT_COUNT"],
    )
    combined["POS_FUTURE_INSTALLMENT_MEAN"] = safe_divide(
        combined["POS_FUTURE_INSTALLMENT_SUM"],
        combined["POS_FUTURE_INSTALLMENT_COUNT"],
    )
    combined["POS_DPD_MEAN"] = safe_divide(
        combined["POS_DPD_SUM"],
        combined["POS_DPD_COUNT"],
    )
    combined["POS_DPD_DEF_MEAN"] = safe_divide(
        combined["POS_DPD_DEF_SUM"],
        combined["POS_DPD_DEF_COUNT"],
    )
    combined["POS_LATE_RATIO"] = safe_divide(
        combined["POS_LATE_COUNT"],
        combined["POS_RECORD_COUNT"],
    )

    drop_columns = [
        "POS_MONTHS_SUM",
        "POS_MONTHS_COUNT",
        "POS_INSTALLMENT_SUM",
        "POS_INSTALLMENT_COUNT",
        "POS_FUTURE_INSTALLMENT_SUM",
        "POS_FUTURE_INSTALLMENT_COUNT",
        "POS_DPD_SUM",
        "POS_DPD_COUNT",
        "POS_DPD_DEF_SUM",
        "POS_DPD_DEF_COUNT",
    ]
    result = combined.drop(columns=drop_columns).reset_index()

    if not result["SK_ID_CURR"].is_unique:
        raise RuntimeError("POS_CASH aggregation is not unique by SK_ID_CURR")

    print(f"[PASS] POS_CASH_balance -> {len(result):,} applicants")
    return result


def aggregate_installments() -> pd.DataFrame:
    """Chunk-aggregate installment repayment history to applicants."""

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
        pd.read_csv(
            require_file("installments_payments.csv"),
            usecols=usecols,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk["PAYMENT_DELAY"] = (
            chunk["DAYS_ENTRY_PAYMENT"]
            - chunk["DAYS_INSTALMENT"]
        )
        chunk["LATE_FLAG"] = (
            chunk["PAYMENT_DELAY"].fillna(0) > 0
        ).astype("int8")
        chunk["PAYMENT_GAP"] = (
            chunk["AMT_PAYMENT"]
            - chunk["AMT_INSTALMENT"]
        )

        grouped = chunk.groupby("SK_ID_CURR").agg(
            INSTALMENT_RECORD_COUNT=("SK_ID_PREV", "size"),
            INSTALMENT_AMOUNT_SUM=("AMT_INSTALMENT", "sum"),
            PAYMENT_AMOUNT_SUM=("AMT_PAYMENT", "sum"),
            PAYMENT_DELAY_SUM=("PAYMENT_DELAY", "sum"),
            PAYMENT_DELAY_COUNT=("PAYMENT_DELAY", "count"),
            PAYMENT_DELAY_MAX=("PAYMENT_DELAY", "max"),
            PAYMENT_GAP_SUM=("PAYMENT_GAP", "sum"),
            PAYMENT_GAP_COUNT=("PAYMENT_GAP", "count"),
            LATE_PAYMENT_COUNT=("LATE_FLAG", "sum"),
        )

        partials.append(grouped)
        print(f"[INFO] installments chunk {chunk_no} processed")

    combined = pd.concat(partials).groupby(level=0).agg(
        {
            "INSTALMENT_RECORD_COUNT": "sum",
            "INSTALMENT_AMOUNT_SUM": "sum",
            "PAYMENT_AMOUNT_SUM": "sum",
            "PAYMENT_DELAY_SUM": "sum",
            "PAYMENT_DELAY_COUNT": "sum",
            "PAYMENT_DELAY_MAX": "max",
            "PAYMENT_GAP_SUM": "sum",
            "PAYMENT_GAP_COUNT": "sum",
            "LATE_PAYMENT_COUNT": "sum",
        }
    )

    combined["PAYMENT_DELAY_MEAN"] = safe_divide(
        combined["PAYMENT_DELAY_SUM"],
        combined["PAYMENT_DELAY_COUNT"],
    )
    combined["PAYMENT_GAP_MEAN"] = safe_divide(
        combined["PAYMENT_GAP_SUM"],
        combined["PAYMENT_GAP_COUNT"],
    )
    combined["PAYMENT_TO_INSTALMENT_RATIO"] = safe_divide(
        combined["PAYMENT_AMOUNT_SUM"],
        combined["INSTALMENT_AMOUNT_SUM"],
    )
    combined["LATE_PAYMENT_RATIO"] = safe_divide(
        combined["LATE_PAYMENT_COUNT"],
        combined["INSTALMENT_RECORD_COUNT"],
    )

    drop_columns = [
        "PAYMENT_DELAY_SUM",
        "PAYMENT_DELAY_COUNT",
        "PAYMENT_GAP_SUM",
        "PAYMENT_GAP_COUNT",
    ]
    result = combined.drop(columns=drop_columns).reset_index()

    if not result["SK_ID_CURR"].is_unique:
        raise RuntimeError(
            "installments aggregation is not unique by SK_ID_CURR"
        )

    print(
        f"[PASS] installments_payments -> {len(result):,} applicants"
    )
    return result


def aggregate_credit_card() -> pd.DataFrame:
    """Chunk-aggregate credit-card history to applicants."""

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
        pd.read_csv(
            require_file("credit_card_balance.csv"),
            usecols=usecols,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        chunk["CC_LATE_FLAG"] = (
            chunk["SK_DPD"].fillna(0) > 0
        ).astype("int8")

        grouped = chunk.groupby("SK_ID_CURR").agg(
            CC_RECORD_COUNT=("SK_ID_PREV", "size"),
            CC_MONTHS_SUM=("MONTHS_BALANCE", "sum"),
            CC_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
            CC_BALANCE_SUM=("AMT_BALANCE", "sum"),
            CC_BALANCE_COUNT=("AMT_BALANCE", "count"),
            CC_BALANCE_MAX=("AMT_BALANCE", "max"),
            CC_CREDIT_LIMIT_SUM=(
                "AMT_CREDIT_LIMIT_ACTUAL",
                "sum",
            ),
            CC_CREDIT_LIMIT_COUNT=(
                "AMT_CREDIT_LIMIT_ACTUAL",
                "count",
            ),
            CC_DRAWINGS_ATM_SUM=(
                "AMT_DRAWINGS_ATM_CURRENT",
                "sum",
            ),
            CC_DRAWINGS_ATM_COUNT=(
                "AMT_DRAWINGS_ATM_CURRENT",
                "count",
            ),
            CC_DRAWINGS_SUM=("AMT_DRAWINGS_CURRENT", "sum"),
            CC_DRAWINGS_COUNT=("AMT_DRAWINGS_CURRENT", "count"),
            CC_PAYMENT_SUM=("AMT_PAYMENT_CURRENT", "sum"),
            CC_PAYMENT_COUNT=("AMT_PAYMENT_CURRENT", "count"),
            CC_MIN_PAYMENT_SUM=(
                "AMT_INST_MIN_REGULARITY",
                "sum",
            ),
            CC_MIN_PAYMENT_COUNT=(
                "AMT_INST_MIN_REGULARITY",
                "count",
            ),
            CC_DPD_SUM=("SK_DPD", "sum"),
            CC_DPD_COUNT=("SK_DPD", "count"),
            CC_DPD_MAX=("SK_DPD", "max"),
            CC_DPD_DEF_SUM=("SK_DPD_DEF", "sum"),
            CC_DPD_DEF_COUNT=("SK_DPD_DEF", "count"),
            CC_DPD_DEF_MAX=("SK_DPD_DEF", "max"),
            CC_LATE_COUNT=("CC_LATE_FLAG", "sum"),
        )

        partials.append(grouped)
        print(f"[INFO] credit_card chunk {chunk_no} processed")

    combined = pd.concat(partials).groupby(level=0).agg(
        {
            "CC_RECORD_COUNT": "sum",
            "CC_MONTHS_SUM": "sum",
            "CC_MONTHS_COUNT": "sum",
            "CC_BALANCE_SUM": "sum",
            "CC_BALANCE_COUNT": "sum",
            "CC_BALANCE_MAX": "max",
            "CC_CREDIT_LIMIT_SUM": "sum",
            "CC_CREDIT_LIMIT_COUNT": "sum",
            "CC_DRAWINGS_ATM_SUM": "sum",
            "CC_DRAWINGS_ATM_COUNT": "sum",
            "CC_DRAWINGS_SUM": "sum",
            "CC_DRAWINGS_COUNT": "sum",
            "CC_PAYMENT_SUM": "sum",
            "CC_PAYMENT_COUNT": "sum",
            "CC_MIN_PAYMENT_SUM": "sum",
            "CC_MIN_PAYMENT_COUNT": "sum",
            "CC_DPD_SUM": "sum",
            "CC_DPD_COUNT": "sum",
            "CC_DPD_MAX": "max",
            "CC_DPD_DEF_SUM": "sum",
            "CC_DPD_DEF_COUNT": "sum",
            "CC_DPD_DEF_MAX": "max",
            "CC_LATE_COUNT": "sum",
        }
    )

    combined["CC_MONTHS_MEAN"] = safe_divide(
        combined["CC_MONTHS_SUM"],
        combined["CC_MONTHS_COUNT"],
    )
    combined["CC_BALANCE_MEAN"] = safe_divide(
        combined["CC_BALANCE_SUM"],
        combined["CC_BALANCE_COUNT"],
    )
    combined["CC_CREDIT_LIMIT_MEAN"] = safe_divide(
        combined["CC_CREDIT_LIMIT_SUM"],
        combined["CC_CREDIT_LIMIT_COUNT"],
    )
    combined["CC_DRAWINGS_ATM_MEAN"] = safe_divide(
        combined["CC_DRAWINGS_ATM_SUM"],
        combined["CC_DRAWINGS_ATM_COUNT"],
    )
    combined["CC_DRAWINGS_MEAN"] = safe_divide(
        combined["CC_DRAWINGS_SUM"],
        combined["CC_DRAWINGS_COUNT"],
    )
    combined["CC_PAYMENT_MEAN"] = safe_divide(
        combined["CC_PAYMENT_SUM"],
        combined["CC_PAYMENT_COUNT"],
    )
    combined["CC_MIN_PAYMENT_MEAN"] = safe_divide(
        combined["CC_MIN_PAYMENT_SUM"],
        combined["CC_MIN_PAYMENT_COUNT"],
    )
    combined["CC_DPD_MEAN"] = safe_divide(
        combined["CC_DPD_SUM"],
        combined["CC_DPD_COUNT"],
    )
    combined["CC_DPD_DEF_MEAN"] = safe_divide(
        combined["CC_DPD_DEF_SUM"],
        combined["CC_DPD_DEF_COUNT"],
    )
    combined["CC_LATE_RATIO"] = safe_divide(
        combined["CC_LATE_COUNT"],
        combined["CC_RECORD_COUNT"],
    )
    combined["CC_UTILIZATION_RATIO"] = safe_divide(
        combined["CC_BALANCE_MEAN"],
        combined["CC_CREDIT_LIMIT_MEAN"],
    )

    drop_columns = [
        "CC_MONTHS_SUM",
        "CC_MONTHS_COUNT",
        "CC_BALANCE_SUM",
        "CC_BALANCE_COUNT",
        "CC_CREDIT_LIMIT_SUM",
        "CC_CREDIT_LIMIT_COUNT",
        "CC_DRAWINGS_ATM_SUM",
        "CC_DRAWINGS_ATM_COUNT",
        "CC_DRAWINGS_SUM",
        "CC_DRAWINGS_COUNT",
        "CC_PAYMENT_SUM",
        "CC_PAYMENT_COUNT",
        "CC_MIN_PAYMENT_SUM",
        "CC_MIN_PAYMENT_COUNT",
        "CC_DPD_SUM",
        "CC_DPD_COUNT",
        "CC_DPD_DEF_SUM",
        "CC_DPD_DEF_COUNT",
    ]
    result = combined.drop(columns=drop_columns).reset_index()

    if not result["SK_ID_CURR"].is_unique:
        raise RuntimeError(
            "credit_card aggregation is not unique by SK_ID_CURR"
        )

    print(
        f"[PASS] credit_card_balance -> {len(result):,} applicants"
    )
    return result


def build_modeling_table() -> tuple[pd.DataFrame, int]:
    """Build the final one-row-per-applicant historical feature table."""

    application = pd.read_csv(
        require_file("application_train.csv"),
        usecols=["SK_ID_CURR", "TARGET"],
    )

    require_columns(
        application,
        ["SK_ID_CURR", "TARGET"],
        "application_train",
    )

    if not application["SK_ID_CURR"].is_unique:
        raise ValueError("application_train.SK_ID_CURR must be unique")

    base_rows = len(application)
    base_targets = application.set_index("SK_ID_CURR")["TARGET"]

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

    if len(result) != base_rows:
        raise RuntimeError(
            "Final applicant population changed during aggregation"
        )

    if not result["SK_ID_CURR"].is_unique:
        raise RuntimeError("Final SK_ID_CURR is not unique")

    final_targets = result.set_index("SK_ID_CURR")["TARGET"]
    if not final_targets.equals(base_targets):
        raise RuntimeError("TARGET changed during aggregation")

    target_dependent_columns = [
        column
        for column in result.columns
        if column.upper().startswith("TARGET_")
    ]
    if target_dependent_columns:
        raise RuntimeError(
            f"Target-derived columns found: {target_dependent_columns}"
        )

    return result, base_rows


def write_reports(
    modeling_data: pd.DataFrame,
    base_rows: int,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "output": OUTPUT_FILE.name,
                "rows": len(modeling_data),
                "columns": len(modeling_data.columns),
                "unique_applicants": (
                    modeling_data["SK_ID_CURR"].nunique()
                ),
                "target_present": "TARGET" in modeling_data.columns,
                "target_missing": int(
                    modeling_data["TARGET"].isna().sum()
                ),
                "row_count_preserved": len(modeling_data) == base_rows,
                "target_used_for_aggregation": False,
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
                "rows": base_rows,
            },
            {
                "stage": "final_modeling_data",
                "rows": len(modeling_data),
            },
        ]
    )
    row_counts["row_count_preserved"] = (
        row_counts["rows"] == base_rows
    )
    row_counts.to_csv(
        REPORT_DIR / "aggregation_row_counts.csv",
        index=False,
    )

    schema = pd.DataFrame(
        {
            "column": modeling_data.columns,
            "dtype": [
                str(modeling_data[column].dtype)
                for column in modeling_data.columns
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
    print(f"Raw data       : {RAW_DIR}")
    print(f"Processed data : {PROCESSED_DIR}")
    print(f"Reports        : {REPORT_DIR}")
    print(f"Chunk size     : {CHUNK_SIZE:,}")

    modeling_data, base_rows = build_modeling_table()

    modeling_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    write_reports(
        modeling_data,
        base_rows,
    )

    one_row_per_applicant = modeling_data["SK_ID_CURR"].is_unique
    target_preserved = (
        modeling_data["TARGET"].notna().all()
    )
    row_count_preserved = len(modeling_data) == base_rows

    print("\n" + "=" * 80)
    print("SLICE 02 VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Application rows       : {base_rows:,}")
    print(f"Final rows              : {len(modeling_data):,}")
    print(
        "One row per applicant   : "
        f"{'PASS' if one_row_per_applicant else 'FAIL'}"
    )
    print(
        "Row count preserved     : "
        f"{'PASS' if row_count_preserved else 'FAIL'}"
    )
    print(
        "TARGET preserved        : "
        f"{'PASS' if target_preserved else 'FAIL'}"
    )
    print("TARGET used in features : NO")
    print(f"Output                  : {OUTPUT_FILE}")
    print("=" * 80)

    if not (
        one_row_per_applicant
        and row_count_preserved
        and target_preserved
    ):
        raise RuntimeError("Slice 02 validation failed.")

    print("Slice 02 aggregation completed successfully.")


if __name__ == "__main__":
    main()

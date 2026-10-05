
from pathlib import Path
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "application_train.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "installments_payments.csv",
    "credit_card_balance.csv",
]

# Number of rows read at a time.
# This prevents large files from being loaded completely.
CHUNK_SIZE = 100_000


# ============================================================
# Helper functions
# ============================================================

def format_bytes(size):
    """Convert bytes to a readable file size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def inspect_csv(file_path):
    """Inspect one CSV without loading the entire file."""

    print("\n" + "=" * 80)
    print(f"DATASET: {file_path.name}")
    print("=" * 80)

    file_size = file_path.stat().st_size

    print(f"File size: {format_bytes(file_size)}")

    # --------------------------------------------------------
    # Read header
    # --------------------------------------------------------

    header = pd.read_csv(file_path, nrows=0)

    columns = list(header.columns)

    print(f"Number of columns: {len(columns)}")
    print("\nColumns:")
    for i, column in enumerate(columns, start=1):
        print(f"  {i:3}. {column}")

    # --------------------------------------------------------
    # Read a small sample for data types / examples
    # --------------------------------------------------------

    sample = pd.read_csv(file_path, nrows=5)

    print("\nData types:")
    for column, dtype in sample.dtypes.items():
        print(f"  {column}: {dtype}")

    print("\nFirst 5 rows:")
    print(sample.to_string(index=False))

    # --------------------------------------------------------
    # Full-file chunk inspection
    # --------------------------------------------------------

    total_rows = 0
    missing_counts = pd.Series(dtype="int64")

    numeric_columns = []
    categorical_columns = []

    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):

        total_rows += len(chunk)

        # Missing values
        chunk_missing = chunk.isna().sum()

        if missing_counts.empty:
            missing_counts = chunk_missing.astype("int64")
        else:
            missing_counts = missing_counts.add(
                chunk_missing,
                fill_value=0
            ).astype("int64")

        # Detect numeric/categorical columns from first chunk
        if not numeric_columns and not categorical_columns:
            numeric_columns = list(
                chunk.select_dtypes(include="number").columns
            )

            categorical_columns = list(
                chunk.select_dtypes(exclude="number").columns
            )

    print(f"\nTotal rows: {total_rows:,}")

    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical/non-numeric columns: {len(categorical_columns)}")

    # --------------------------------------------------------
    # Missing-value report
    # --------------------------------------------------------

    missing_report = pd.DataFrame({
        "column": missing_counts.index,
        "missing_count": missing_counts.values,
    })

    missing_report["missing_percentage"] = (
        missing_report["missing_count"] / total_rows * 100
    )

    missing_report = missing_report.sort_values(
        "missing_percentage",
        ascending=False
    )

    print("\nTop missing-value columns:")

    top_missing = missing_report.head(15)

    for _, row in top_missing.iterrows():
        print(
            f"  {row['column']}: "
            f"{int(row['missing_count']):,} missing "
            f"({row['missing_percentage']:.2f}%)"
        )

    # --------------------------------------------------------
    # Identifier columns
    # --------------------------------------------------------

    identifier_columns = [
        column
        for column in columns
        if column.startswith("SK_ID")
    ]

    print("\nIdentifier columns:")

    if identifier_columns:
        for column in identifier_columns:
            print(f"  - {column}")
    else:
        print("  None detected")

    # --------------------------------------------------------
    # Save missing-value report
    # --------------------------------------------------------

    missing_output = OUTPUT_DIR / (
        file_path.stem + "_missing_values.csv"
    )

    missing_report.to_csv(
        missing_output,
        index=False
    )

    return {
        "dataset": file_path.name,
        "file_size": format_bytes(file_size),
        "rows": total_rows,
        "columns": len(columns),
        "numeric_columns": len(numeric_columns),
        "categorical_columns": len(categorical_columns),
        "identifier_columns": ", ".join(identifier_columns),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("HOME CREDIT DATASET INSPECTION")
    print("=" * 80)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    if not DATA_DIR.exists():
        print("\nERROR: data/raw directory does not exist.")
        print("Create it and place the CSV files there.")
        return

    summaries = []

    for dataset_name in DATASETS:

        file_path = DATA_DIR / dataset_name

        if not file_path.exists():
            print("\n" + "-" * 80)
            print(f"NOT FOUND: {dataset_name}")
            print("-" * 80)
            print("Skipping this dataset.")
            continue

        try:
            summary = inspect_csv(file_path)
            summaries.append(summary)

        except Exception as error:
            print("\nERROR while inspecting:")
            print(f"  {dataset_name}")
            print(f"  {type(error).__name__}: {error}")

    # --------------------------------------------------------
    # Save overall summary
    # --------------------------------------------------------

    if summaries:

        summary_df = pd.DataFrame(summaries)

        summary_file = OUTPUT_DIR / "dataset_summary.csv"

        summary_df.to_csv(
            summary_file,
            index=False
        )

        print("\n" + "=" * 80)
        print("OVERALL DATASET SUMMARY")
        print("=" * 80)

        print(summary_df.to_string(index=False))

        print("\nSaved:")
        print(f"  {summary_file}")

    # --------------------------------------------------------
    # Final instructions
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)

    print("\nIndividual missing-value reports:")
    print(f"  {OUTPUT_DIR}")

    print("\nDo NOT upload the CSV datasets to GitHub.")
    print("The inspection script only reads the datasets.")
    print("It does not modify or delete them.")


if __name__ == "__main__":
    main()

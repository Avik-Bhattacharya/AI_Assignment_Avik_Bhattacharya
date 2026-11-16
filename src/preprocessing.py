"""
Customer Support Ticket Classification System
Data Preprocessing Pipeline (preprocessing.py)

Cleans and prepares raw tickets from data/tickets.csv in memory without
requiring a permanent intermediate CSV file.
"""

from pathlib import Path
import re
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE_PATH = PROJECT_ROOT / "data" / "tickets.csv"


def load_dataset(file_path: Path = INPUT_FILE_PATH) -> pd.DataFrame:
    """Load the raw customer support tickets CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input dataset not found at '{file_path}'.")

    print("=" * 70)
    print("STEP 1: LOADING DATASET")
    print("=" * 70)
    print(f"Loading data from: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Successfully loaded {len(df)} records.")
    print("\nFirst 5 rows of the raw dataset:")
    print("-" * 70)
    print(df.head().to_string(index=False))

    return df


def display_dataset_info(df: pd.DataFrame) -> None:
    """Display dataset dimensions, column names, data types, and summary stats."""
    print("\n" + "=" * 70)
    print("STEP 2: DATASET INFORMATION")
    print("=" * 70)
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print(f"\nColumn Names ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")

    print("\nData Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:<20}: {dtype}")

    print("\nBasic Statistical Summary:")
    print("-" * 70)
    print(df.describe(include="all").to_string())


def handle_missing_values(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Check and handle missing values across all columns."""
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: MISSING VALUE ANALYSIS & HANDLING")
        print("=" * 70)

    missing_series = df.isnull().sum()
    if verbose:
        print("Missing value count per column:")
        for col, count in missing_series.items():
            print(f"  - {col:<20}: {count} missing")

    total_missing = missing_series.sum()
    if verbose:
        print(f"\nTotal missing values found: {total_missing}")

    initial_count = len(df)
    cleaned_df = df.copy()

    # Drop records missing text or category: models cannot learn without features or target labels
    desc_missing = cleaned_df["ticket_description"].isnull() | (
        cleaned_df["ticket_description"].astype(str).str.strip() == ""
    )
    if desc_missing.sum() > 0:
        if verbose:
            print(f"[ACTION] Removing {desc_missing.sum()} record(s) with missing ticket description.")
        cleaned_df = cleaned_df[~desc_missing]

    category_missing = cleaned_df["category"].isnull() | (
        cleaned_df["category"].astype(str).str.strip() == ""
    )
    if category_missing.sum() > 0:
        if verbose:
            print(f"[ACTION] Removing {category_missing.sum()} record(s) with missing target category.")
        cleaned_df = cleaned_df[~category_missing]

    # For secondary fields, fill with 'Unknown' instead of guessing values
    for col in ["priority", "status"]:
        if col in cleaned_df.columns:
            missing_count = cleaned_df[col].isnull().sum()
            if missing_count > 0:
                if verbose:
                    print(f"[ACTION] Filling {missing_count} missing value(s) in '{col}' with 'Unknown'.")
                cleaned_df[col] = cleaned_df[col].fillna("Unknown")

    removed_records = initial_count - len(cleaned_df)
    if verbose:
        print(f"Records remaining after missing value handling: {len(cleaned_df)} (Removed: {removed_records})")

    return cleaned_df


def handle_duplicates(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Identify and remove exact duplicate records and duplicate ticket IDs."""
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 4: DUPLICATE ANALYSIS & HANDLING")
        print("=" * 70)

    exact_duplicates = df.duplicated().sum()
    if verbose:
        print(f"Exact duplicate records found: {exact_duplicates}")

    cleaned_df = df.copy()
    if exact_duplicates > 0:
        if verbose:
            print(f"[ACTION] Removing {exact_duplicates} exact duplicate record(s).")
        cleaned_df = cleaned_df.drop_duplicates()

    duplicate_ids = cleaned_df.duplicated(subset=["ticket_id"]).sum()
    if verbose:
        print(f"Duplicate ticket IDs found: {duplicate_ids}")

    if duplicate_ids > 0:
        if verbose:
            print(f"[ACTION] Removing {duplicate_ids} duplicate ticket ID record(s), keeping first occurrence.")
        cleaned_df = cleaned_df.drop_duplicates(subset=["ticket_id"], keep="first")

    if verbose:
        print(f"Records remaining after duplicate handling: {len(cleaned_df)}")
    return cleaned_df


def clean_text(text: str) -> str:
    """
    Clean raw ticket description text for NLP classification.
    Lowercases text, strips punctuation, and normalizes spaces.
    Words like 'not', 'cannot', 'unable' are preserved for sentiment/negation signal.
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.lower()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def apply_text_cleaning(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply clean_text() to the ticket_description column."""
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 5: TEXT CLEANING")
        print("=" * 70)
        print("Applying clean_text() to 'ticket_description' column...")
        print("Important classification keywords ('not', 'cannot', 'unable') are preserved.")

    cleaned_df = df.copy()
    cleaned_df["ticket_description"] = cleaned_df["ticket_description"].apply(clean_text)

    if verbose:
        print("Text cleaning completed successfully.")
    return cleaned_df


def preprocess_dataset(
    df: pd.DataFrame = None,
    file_path: Path = INPUT_FILE_PATH,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Preprocess the ticket dataset in memory:
    1. Loads data from file_path if df is not provided.
    2. Handles missing values.
    3. Handles duplicate rows and ticket IDs.
    4. Cleans and normalizes ticket descriptions.
    Returns the cleaned DataFrame without saving any file to disk.
    """
    if df is None:
        df = load_dataset(file_path)

    df_missing = handle_missing_values(df, verbose=verbose)
    df_dedup = handle_duplicates(df_missing, verbose=verbose)
    df_clean = apply_text_cleaning(df_dedup, verbose=verbose)
    df_clean = df_clean.reset_index(drop=True)
    return df_clean


def final_verification(df: pd.DataFrame) -> None:
    """Display final dataset shape, remaining missing values, duplicates, and sample rows."""
    print("\n" + "=" * 70)
    print("STEP 6: FINAL IN-MEMORY VERIFICATION")
    print("=" * 70)
    print(f"Final Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    remaining_missing = df.isnull().sum().sum()
    print(f"Remaining Missing Values: {remaining_missing}")

    remaining_duplicates = df.duplicated().sum()
    print(f"Remaining Exact Duplicates: {remaining_duplicates}")

    remaining_id_duplicates = df.duplicated(subset=["ticket_id"]).sum()
    print(f"Remaining Duplicate Ticket IDs: {remaining_id_duplicates}")

    print("\nFirst 5 Processed Records:")
    print("-" * 70)
    print(df.head().to_string(index=False))
    print("=" * 70)


def main():
    print("Customer Support Ticket Classification System")
    print("Data Preprocessing Script Started...\n")

    df = load_dataset(INPUT_FILE_PATH)
    display_dataset_info(df)
    df_cleaned = preprocess_dataset(df, verbose=True)
    final_verification(df_cleaned)

    print("\nPreprocessing pipeline completed successfully in memory!")
    print(f"Total ready records: {len(df_cleaned)} (No intermediate file created).")


if __name__ == "__main__":
    main()

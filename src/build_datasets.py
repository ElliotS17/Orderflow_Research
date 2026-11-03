from pathlib import Path
import warnings

import databento as db
import pandas as pd
import pyarrow  # Required for Parquet output


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

EXCLUDED_DATE = "20261101"
TARGET_SYMBOL = "NQZ6"

ORIGINAL_OUTPUT = PROCESSED_DIR / "trades_original.parquet"
CLEAN_OUTPUT = PROCESSED_DIR / "trades_deduplicated.parquet"


# Suppress a known Databento/Pandas deprecation warning.
warnings.filterwarnings(
    "ignore",
    message=r".*inplace keyword in DataFrame\.set_index is deprecated.*"
)


# --------------------------------------------------
# Find input files
# --------------------------------------------------

files = sorted(RAW_DIR.glob("*.trades.dbn.zst"))
files = [f for f in files if EXCLUDED_DATE not in f.name]

if len(files) != 17:
    raise RuntimeError(
        f"Expected 17 primary-sample files, found {len(files)}. "
        "Check data/raw/ before continuing."
    )

print(f"Found {len(files)} input files.")
print(f"Target contract: {TARGET_SYMBOL}")


# --------------------------------------------------
# Load and filter trade records
# --------------------------------------------------

frames = []

for path in files:
    print(f"\nLoading {path.name}...")

    store = db.DBNStore.from_file(path)
    df = store.to_df()

    # Ensure ts_recv is available as a regular column.
    if df.index.name is not None and df.index.name not in df.columns:
        df = df.reset_index()

    required_columns = {
        "ts_recv", "ts_event", "price", "size",
        "side", "sequence", "symbol"
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise RuntimeError(
            f"{path.name} is missing expected columns: {missing_columns}"
        )

    # Keep only the NQZ6 outright contract.
    df = df[df["symbol"] == TARGET_SYMBOL].copy()

    # Preserve the original file for traceability.
    df["source_file"] = path.name

    if df.empty:
        raise RuntimeError(f"No {TARGET_SYMBOL} records found in {path.name}")

    print(f"  NQZ6 records: {len(df):,}")
    frames.append(df)


# --------------------------------------------------
# Combine records
# --------------------------------------------------

original = pd.concat(frames, ignore_index=True)

# Every column except source_file represents a record field.
# source_file is provenance, not part of record identity.
record_columns = [
    column for column in original.columns
    if column != "source_file"
]

print("\nCombining complete sample...")
print(f"Original record count: {len(original):,}")


# --------------------------------------------------
# Remove exact repeated records
# --------------------------------------------------

clean = original.drop_duplicates(
    subset=record_columns,
    keep="first"
).copy()

removed = len(original) - len(clean)

print("\n========== DUPLICATE REMOVAL ==========")
print(f"Original records: {len(original):,}")
print(f"Clean records: {len(clean):,}")
print(f"Excess repeated copies removed: {removed:,}")
print(f"Percentage removed: {removed / len(original) * 100:.4f}%")


# --------------------------------------------------
# Sort chronologically
# --------------------------------------------------

original = original.sort_values(
    ["ts_event", "ts_recv"],
    kind="stable"
).reset_index(drop=True)

clean = clean.sort_values(
    ["ts_event", "ts_recv"],
    kind="stable"
).reset_index(drop=True)


# --------------------------------------------------
# Validation
# --------------------------------------------------

required_sides = {"B", "A", "N"}
observed_sides = set(original["side"].dropna().unique())

if not observed_sides.issubset(required_sides):
    raise RuntimeError(f"Unexpected trade sides found: {observed_sides}")

if clean.duplicated(subset=record_columns).any():
    raise RuntimeError("Exact duplicates remain in the cleaned dataset.")

if len(original) != len(clean) + removed:
    raise RuntimeError("Record-count reconciliation failed.")

print("\n========== DATA VALIDATION ==========")

for name, data in [
    ("ORIGINAL", original),
    ("DEDUPLICATED", clean),
]:
    size = data["size"].astype("int64")

    buy_volume = int(size[data["side"] == "B"].sum())
    sell_volume = int(size[data["side"] == "A"].sum())
    unknown_volume = int(size[data["side"] == "N"].sum())

    print(f"\n{name}")
    print(f"Records: {len(data):,}")
    print(f"Total contracts: {int(size.sum()):,}")
    print(f"Buyer-initiated volume: {buy_volume:,}")
    print(f"Seller-initiated volume: {sell_volume:,}")
    print(f"Unknown-side volume: {unknown_volume:,}")
    print(f"Minimum price: {data['price'].min()}")
    print(f"Maximum price: {data['price'].max()}")
    print(f"Exact duplicate excess: "
          f"{int(data.duplicated(subset=record_columns).sum()):,}")

print("\n========== DAILY RECORD COUNTS ==========")
daily_counts = (
    original.groupby("source_file")
    .size()
    .rename("original_records")
    .to_frame()
)

daily_counts["clean_records"] = (
    clean.groupby("source_file").size()
)

daily_counts["removed"] = (
    daily_counts["original_records"] - daily_counts["clean_records"]
)

print(daily_counts.to_string())


# --------------------------------------------------
# Save datasets
# --------------------------------------------------

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

original.to_parquet(ORIGINAL_OUTPUT, index=False)
clean.to_parquet(CLEAN_OUTPUT, index=False)

print("\n========== FILES SAVED ==========")
print(f"Original: {ORIGINAL_OUTPUT}")
print(f"Cleaned:  {CLEAN_OUTPUT}")

print("\nDataset construction complete.")
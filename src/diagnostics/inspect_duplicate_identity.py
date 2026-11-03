from pathlib import Path
import warnings

import databento as db
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r".*inplace keyword in DataFrame\.set_index is deprecated.*"
)

RAW_DIR = Path("data/raw")
files = sorted(RAW_DIR.glob("*.trades.dbn.zst"))
files = [f for f in files if "20261101" not in f.name]

frames = []

for path in files:
    print("Loading:", path.name)

    store = db.DBNStore.from_file(path)
    df = store.to_df().reset_index()

    if path == files[0]:
        print("\nDBN metadata:")
        print(store.metadata)

        print("\nDataFrame columns:")
        print(df.columns.tolist())

    df = df[df["symbol"] == "NQZ6"].copy()
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# All available columns define an exact repeated record.
identity_columns = list(data.columns)

group_sizes = (
    data.groupby(identity_columns, dropna=False)
    .size()
    .rename("copies")
)

duplicate_groups = group_sizes[group_sizes > 1]

print("\n========== EXACT RECORD GROUPS ==========")
print("Total records:", f"{len(data):,}")
print("Unique exact-record patterns:", f"{len(group_sizes):,}")
print("Patterns repeated more than once:", f"{len(duplicate_groups):,}")
print("Excess copies:", f"{(duplicate_groups - 1).sum():,}")

print("\nCopy-count distribution:")
print(duplicate_groups.value_counts().sort_index().to_string())

# Examine duplicate records with their timestamps.
duplicate_rows = data.loc[
    data.duplicated(subset=identity_columns, keep=False)
].copy()

print("\n========== DUPLICATE TIMESTAMP CHECK ==========")
print(
    "Duplicate-group records with exact ts_event values:",
    duplicate_rows["ts_event"].notna().sum()
)
print(
    "Unique ts_event values among duplicate rows:",
    duplicate_rows["ts_event"].nunique()
)
print(
    "Unique ts_recv values among duplicate rows:",
    duplicate_rows["ts_recv"].nunique()
)

# Check whether multiple distinct records share the same event timestamp.
event_time_groups = data.groupby("ts_event").agg(
    records=("price", "size"),
    unique_prices=("price", "nunique"),
    unique_sizes=("size", "nunique"),
    unique_sides=("side", "nunique"),
    unique_sequences=("sequence", "nunique")
)

event_collisions = event_time_groups[event_time_groups["records"] > 1]

print("\n========== EVENT TIMESTAMP COLLISIONS ==========")
print("Timestamps with multiple records:", len(event_collisions))
print(
    "Timestamps with multiple prices:",
    (event_collisions["unique_prices"] > 1).sum()
)
print(
    "Timestamps with multiple sizes:",
    (event_collisions["unique_sizes"] > 1).sum()
)
print(
    "Timestamps with multiple sides:",
    (event_collisions["unique_sides"] > 1).sum()
)
print(
    "Timestamps with multiple sequence numbers:",
    (event_collisions["unique_sequences"] > 1).sum()
)

# Examine the combined event-time and receive-time pair.
timestamp_groups = data.groupby(
    ["ts_event", "ts_recv"],
    dropna=False
).agg(
    records=("price", "size"),
    unique_prices=("price", "nunique"),
    unique_sizes=("size", "nunique"),
    unique_sides=("side", "nunique"),
    unique_sequences=("sequence", "nunique")
)

timestamp_collisions = timestamp_groups[
    timestamp_groups["records"] > 1
]

print("\n========== EVENT + RECEIVE TIMESTAMP PAIRS ==========")
print("Timestamp pairs with multiple records:", len(timestamp_collisions))
print(
    "Pairs with multiple prices:",
    (timestamp_collisions["unique_prices"] > 1).sum()
)
print(
    "Pairs with multiple sequence numbers:",
    (timestamp_collisions["unique_sequences"] > 1).sum()
)

print("\n========== SAMPLE MULTI-PRICE TIMESTAMP PAIRS ==========")

examples = timestamp_collisions[
    timestamp_collisions["unique_prices"] > 1
].head(5)

if len(examples):
    example_keys = examples.index

    sample = data.set_index(
        ["ts_event", "ts_recv"]
    ).loc[example_keys]

    print(sample[
        ["sequence", "price", "size", "side", "flags"]
    ].to_string())
else:
    print("No multi-price timestamp pairs found.")

print("\nAnalysis complete. No records were removed.")
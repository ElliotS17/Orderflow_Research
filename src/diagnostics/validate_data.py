from pathlib import Path
import databento as db
import pandas as pd
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*inplace keyword in DataFrame\.set_index is deprecated.*"
)

path = Path("data/raw/glbx-mdp3-20261012.trades.dbn.zst")

store = db.DBNStore.from_file(path)
df = store.to_df().reset_index()

nq = df[df["symbol"] == "NQZ6"].copy()

# 1. Inspect exact duplicate records
duplicate_rows = nq[nq.duplicated(keep=False)].sort_values(
    ["sequence", "price"]
)

print("===== EXACT DUPLICATE RECORDS =====")
print("Rows belonging to duplicate groups:", len(duplicate_rows))
print("Unique duplicate groups:",
      duplicate_rows.drop_duplicates().shape[0])

print(duplicate_rows[
    ["ts_event", "ts_recv", "sequence", "price", "size", "side"]
].head(20).to_string(index=False))

# 2. Check whether sequence numbers map to multiple prices or sizes
sequence_stats = nq.groupby("sequence").agg(
    record_count=("sequence", "size"),
    unique_prices=("price", "nunique"),
    unique_sizes=("size", "nunique"),
    unique_sides=("side", "nunique"),
    unique_event_times=("ts_event", "nunique"),
)

print("\n===== SEQUENCE NUMBER REUSE =====")
print("Sequences with multiple records:",
      (sequence_stats["record_count"] > 1).sum())
print("Sequences with multiple prices:",
      (sequence_stats["unique_prices"] > 1).sum())
print("Sequences with multiple sizes:",
      (sequence_stats["unique_sizes"] > 1).sum())
print("Sequences with multiple sides:",
      (sequence_stats["unique_sides"] > 1).sum())

print("\nLargest sequence groups:")
print(
    sequence_stats.sort_values("record_count", ascending=False)
    .head(10)
    .to_string()
)

# 3. Show all records for a sequence with multiple prices
multiple_price_sequences = sequence_stats[
    sequence_stats["unique_prices"] > 1
].index

if len(multiple_price_sequences):
    example_sequence = multiple_price_sequences[0]

    print("\n===== EXAMPLE: SAME SEQUENCE, MULTIPLE PRICES =====")
    print(
        nq[nq["sequence"] == example_sequence][
            ["ts_event", "ts_recv", "sequence",
             "price", "size", "side", "flags"]
        ].to_string(index=False)
    )
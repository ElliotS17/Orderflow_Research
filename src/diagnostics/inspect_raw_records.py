from pathlib import Path
from collections import Counter
import databento as db

RAW_FILE = Path("data/raw/glbx-mdp3-20261012.trades.dbn.zst")
NQZ6_INSTRUMENT_ID = 261401

# Actual record values only; no bound methods.
RAW_FIELDS = [
    "rtype",
    "publisher_id",
    "instrument_id",
    "ts_event",
    "price",
    "size",
    "action",
    "side",
    "depth",
    "ts_recv",
    "flags",
    "ts_in_delta",
    "sequence",
    "ts_out",
    "size_hint",
    "ts_index",
]

store = db.DBNStore.from_file(RAW_FILE)

records = []
for record in store:
    if record.instrument_id != NQZ6_INSTRUMENT_ID:
        continue

    values = tuple(getattr(record, field) for field in RAW_FIELDS)
    records.append(values)

counts = Counter(records)
duplicate_groups = {
    record: copies
    for record, copies in counts.items()
    if copies > 1
}

print("Raw NQZ6 records:", len(records))
print("Unique raw patterns:", len(counts))
print("Repeated raw patterns:", len(duplicate_groups))
print("Excess copies:", sum(copies - 1 for copies in duplicate_groups.values()))

print("\n========== RECORD SIZE ==========")
first_record = next(
    record for record in db.DBNStore.from_file(RAW_FILE)
    if record.instrument_id == NQZ6_INSTRUMENT_ID
)
print("record_size is callable:", callable(first_record.record_size))
if callable(first_record.record_size):
    print("record_size():", first_record.record_size())

print("\n========== RAW-ONLY FIELD VARIATION ==========")
for field in ("ts_index", "ts_out", "size_hint"):
    i = RAW_FIELDS.index(field)
    values = {record[i] for record in records}
    print(f"{field}: {len(values)} distinct values")
    print("Examples:", list(values)[:5])

print("\n========== SAMPLE REPEATED RECORDS ==========")
for record, copies in list(duplicate_groups.items())[:5]:
    print(f"\nCopies: {copies}")
    for field, value in zip(RAW_FIELDS, record):
        print(f"  {field}: {value}")

print("\n========== DATAFRAME COMPARISON ==========")
df = store.to_df()
df = df[df["symbol"] == "NQZ6"]

print("DataFrame records:", len(df))
print("DataFrame exact duplicate excess:",
      int(df.duplicated(keep="first").sum()))

print("\nInspection complete. No records were modified.")
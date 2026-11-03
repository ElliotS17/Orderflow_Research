from pathlib import Path
from collections import defaultdict
import databento as db

RAW_FILE = Path("data/raw/glbx-mdp3-20261012.trades.dbn.zst")
NQZ6_INSTRUMENT_ID = 261401

FIELDS = [
    "rtype", "publisher_id", "instrument_id", "ts_event",
    "price", "size", "action", "side", "depth",
    "ts_recv", "flags", "ts_in_delta", "sequence",
    "ts_out", "size_hint", "ts_index",
]

store = db.DBNStore.from_file(RAW_FILE)

positions = defaultdict(list)
total_records = 0

for record in store:
    if record.instrument_id != NQZ6_INSTRUMENT_ID:
        continue

    key = tuple(getattr(record, field) for field in FIELDS)
    positions[key].append(total_records)
    total_records += 1

duplicate_groups = {
    key: locations
    for key, locations in positions.items()
    if len(locations) > 1
}

gaps = []
adjacent_groups = 0
same_sequence_groups = 0

for key, locations in duplicate_groups.items():
    group_gaps = [
        locations[i + 1] - locations[i]
        for i in range(len(locations) - 1)
    ]
    gaps.extend(group_gaps)

    if any(gap == 1 for gap in group_gaps):
        adjacent_groups += 1

    same_sequence_groups += 1

print("Total NQZ6 records:", total_records)
print("Repeated groups:", len(duplicate_groups))
print("Excess copies:", sum(len(x) - 1 for x in duplicate_groups.values()))

print("\n========== POSITION ANALYSIS ==========")
print("Groups with at least one adjacent repeat:", adjacent_groups)
print("Total gaps between repeated copies:", len(gaps))

if gaps:
    print("Minimum gap:", min(gaps))
    print("Median gap:", sorted(gaps)[len(gaps) // 2])
    print("Maximum gap:", max(gaps))

print("\n========== EXAMPLES ==========")
for key, locations in list(duplicate_groups.items())[:10]:
    print(
        f"Sequence={key[FIELDS.index('sequence')]}, "
        f"price={key[FIELDS.index('price')]}, "
        f"size={key[FIELDS.index('size')]}, "
        f"side={key[FIELDS.index('side')]}, "
        f"positions={locations}"
    )

print("\nNo records were modified.")
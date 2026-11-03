from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_features_deduplicated.parquet"
)

df = pd.read_parquet(DATA_PATH)

# Recover the actual minute timestamps. The previous diagnostic showed
# that the DataFrame has a row-number index, so check for a minute column.
if "minute" in df.columns:
    timestamps = pd.to_datetime(df["minute"])
elif isinstance(df.index, pd.DatetimeIndex):
    timestamps = pd.Series(df.index, index=df.index)
else:
    raise ValueError(
        "Could not find minute timestamps. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.copy()
df["_minute_timestamp"] = timestamps.to_numpy()

required = [
    "absolute_net_volume",
    "absolute_imbalance",
    "signed_price_response",
    "response_per_100_net_contracts",
]

missing = [column for column in required if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df["_minute_timestamp"].isna().any():
    raise ValueError("Minute timestamps contain missing values.")

if df["_minute_timestamp"].duplicated().any():
    raise ValueError("Duplicate minute timestamps detected.")

df = df.sort_values("_minute_timestamp").reset_index(drop=True)

print("=" * 70)
print("TIMESTAMP CHECK")
print("=" * 70)
print(f"Actual date range: {df['_minute_timestamp'].min()}")
print(f"                  {df['_minute_timestamp'].max()}")

gaps = df["_minute_timestamp"].diff().dropna().dt.total_seconds() / 60
print("\nObserved gaps:")
print(gaps.value_counts().sort_index().to_string())

print("\nLargest gaps:")
largest_gap_rows = gaps.nlargest(5)
for row_index, gap in largest_gap_rows.items():
    print(
        f"  {df.loc[row_index - 1, '_minute_timestamp']} -> "
        f"{df.loc[row_index, '_minute_timestamp']}: "
        f"{gap:.0f} minutes"
    )

# Calculate exploratory thresholds on the full sample.
# These are for understanding event counts, not final out-of-sample use.
net_volume_80 = df["absolute_net_volume"].quantile(0.80)
imbalance_80 = df["absolute_imbalance"].quantile(0.80)

net_volume_75 = df["absolute_net_volume"].quantile(0.75)
imbalance_75 = df["absolute_imbalance"].quantile(0.75)

print("\n" + "=" * 70)
print("JOINT STRONG-ORDERFLOW CANDIDATES")
print("=" * 70)

candidate_masks = {
    "Top 20% in BOTH measures": (
        (df["absolute_net_volume"] >= net_volume_80)
        & (df["absolute_imbalance"] >= imbalance_80)
    ),
    "Top 25% in BOTH measures": (
        (df["absolute_net_volume"] >= net_volume_75)
        & (df["absolute_imbalance"] >= imbalance_75)
    ),
}

for label, mask in candidate_masks.items():
    count = int(mask.sum())
    print(f"\n{label}")
    print(f"  Net-volume cutoff: {net_volume_80 if '20%' in label else net_volume_75:.4f}")
    print(f"  Imbalance cutoff:  {imbalance_80 if '20%' in label else imbalance_75:.4f}")
    print(f"  Matching minutes:  {count:,} ({count / len(df):.2%})")

# Examine price response within the stricter candidate group.
strong_flow = candidate_masks["Top 20% in BOTH measures"]
subset = df.loc[strong_flow].copy()

subset["absolute_price_response"] = subset["signed_price_response"].abs()
subset["absolute_response_per_100"] = (
    subset["response_per_100_net_contracts"].abs()
)

print("\n" + "=" * 70)
print("PRICE RESPONSE WITHIN TOP-20%-IN-BOTH GROUP")
print("=" * 70)
print(f"Strong-flow minutes: {len(subset):,}")
print(
    "Minutes with valid price-response ratio: "
    f"{subset['absolute_response_per_100'].notna().sum():,}"
)

for column, label in [
    ("absolute_price_response", "Absolute price response (points)"),
    ("absolute_response_per_100", "Absolute response per 100 net contracts"),
]:
    values = subset[column].dropna()
    print(f"\n{label}")

    for percentile in [10, 25, 50, 75, 90]:
        print(
            f"  {percentile:>2}th percentile: "
            f"{values.quantile(percentile / 100):.4f}"
        )

print("\nDiagnostic complete. No event labels or targets were created.")
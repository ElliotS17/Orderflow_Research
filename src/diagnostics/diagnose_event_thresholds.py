from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "research_features_deduplicated.parquet"

df = pd.read_parquet(DATA_PATH)

required_columns = [
    "absolute_net_volume",
    "absolute_imbalance",
    "signed_price_response",
    "response_per_100_net_contracts",
    "gap_minutes",
]

missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"Duplicate timestamps: {df.index.duplicated().sum():,}")

# Summarize the variables relevant to event definition.
variables = {
    "Absolute net volume": df["absolute_net_volume"],
    "Absolute volume imbalance": df["absolute_imbalance"],
    "Absolute signed price response": df["signed_price_response"].abs(),
    "Absolute response per 100 net contracts":
        df["response_per_100_net_contracts"].abs(),
}

print("\n" + "=" * 70)
print("KEY VARIABLE DISTRIBUTIONS")
print("=" * 70)

for name, series in variables.items():
    values = series.dropna()

    print(f"\n{name}")
    print(f"  Non-missing observations: {len(values):,}")
    print(f"  Mean:   {values.mean():.4f}")
    print(f"  Median: {values.median():.4f}")
    print(f"  Std:    {values.std():.4f}")
    print("  Percentiles:")

    for percentile in [50, 60, 70, 75, 80, 90, 95, 99]:
        value = values.quantile(percentile / 100)
        print(f"    {percentile:>2}th: {value:.4f}")

# Inspect the time gaps between active minute bars.
gaps = df["gap_minutes"].dropna()

print("\n" + "=" * 70)
print("TIME GAP DISTRIBUTION")
print("=" * 70)
print(f"Non-missing gaps: {len(gaps):,}")
print(f"One-minute gaps: {(gaps == 1).sum():,}")
print(f"Gaps greater than one minute: {(gaps > 1).sum():,}")

if not gaps.empty:
    print(f"Largest gap: {gaps.max():.0f} minutes")
    print("\nMost common gap lengths:")
    print(gaps.value_counts().head(10).sort_index().to_string())

# Show how many observations would meet different individual cutoffs.
print("\n" + "=" * 70)
print("CANDIDATE ORDERFLOW CUTOFFS")
print("=" * 70)

for column, label in [
    ("absolute_net_volume", "Absolute net volume"),
    ("absolute_imbalance", "Absolute imbalance"),
]:
    series = df[column].dropna()
    print(f"\n{label}")

    for percentile in [70, 75, 80, 90]:
        cutoff = series.quantile(percentile / 100)
        count = (series >= cutoff).sum()
        print(
            f"  Top {100 - percentile:>2}% cutoff: "
            f"{cutoff:.4f}  |  observations: {count:,}"
        )

print("\nDiagnostic complete. No observations were removed or modified.")
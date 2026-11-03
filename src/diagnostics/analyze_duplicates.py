from pathlib import Path
import warnings

import databento as db
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    message=r".*inplace keyword in DataFrame\.set_index is deprecated.*"
)

RAW_DIR = Path("data/raw")
files = sorted(RAW_DIR.glob("*.trades.dbn.zst"))
files = [f for f in files if "20261101" not in f.name]

frames = []

print("Loading primary-sample trade files...")

for path in files:
    print(f"  {path.name}")
    df = db.DBNStore.from_file(path).to_df().reset_index()
    df = df[df["symbol"] == "NQZ6"].copy()
    frames.append(df)

original = pd.concat(frames, ignore_index=True)
deduplicated = original.drop_duplicates().copy()

def minute_features(df, prefix):
    df = df.copy()

    # Explicit signed integer conversion prevents unsigned underflow.
    size = df["size"].astype("int64")

    df["buy_volume"] = np.where(
        df["side"] == "B", size, 0
    ).astype("int64")

    df["sell_volume"] = np.where(
        df["side"] == "A", size, 0
    ).astype("int64")

    df["minute"] = df["ts_event"].dt.floor("min")

    grouped = df.groupby("minute").agg(
        buy_volume=("buy_volume", "sum"),
        sell_volume=("sell_volume", "sum"),
        records=("size", "size")
    )

    # Both operands are now signed integers.
    net_volume = grouped["buy_volume"] - grouped["sell_volume"]
    total_directional_volume = (
        grouped["buy_volume"] + grouped["sell_volume"]
    )

    # Convert to float only for the division.
    imbalance = (
        net_volume.astype("float64")
        / total_directional_volume.replace(0, np.nan).astype("float64")
    )

    grouped[f"{prefix}_imbalance"] = imbalance
    grouped = grouped.rename(
        columns={"records": f"{prefix}_records"}
    )

    return grouped[
        [f"{prefix}_imbalance", f"{prefix}_records"]
    ]

print("\nCalculating original and deduplicated minute features...")

raw_minutes = minute_features(original, "raw")
clean_minutes = minute_features(deduplicated, "clean")

comparison = raw_minutes.join(clean_minutes, how="outer")

valid = comparison.dropna(
    subset=["raw_imbalance", "clean_imbalance"]
).copy()

differences = (
    valid["raw_imbalance"] - valid["clean_imbalance"]
)

print("\n========== CORRECTED ONE-MINUTE SENSITIVITY ==========")
print("Minutes compared:", len(valid))

if len(valid):
    print(
        "Original imbalance range:",
        valid["raw_imbalance"].min(),
        "to",
        valid["raw_imbalance"].max()
    )
    print(
        "Deduplicated imbalance range:",
        valid["clean_imbalance"].min(),
        "to",
        valid["clean_imbalance"].max()
    )

    print(
        "Correlation, original vs deduplicated:",
        valid["raw_imbalance"].corr(valid["clean_imbalance"])
    )
    print(
        "Mean absolute imbalance difference:",
        differences.abs().mean()
    )
    print(
        "Median absolute imbalance difference:",
        differences.abs().median()
    )
    print(
        "95th percentile absolute difference:",
        differences.abs().quantile(0.95)
    )
    print(
        "Maximum absolute difference:",
        differences.abs().max()
    )
    print(
        "Minutes with any imbalance change:",
        (differences.abs() > 1e-12).sum()
    )

    comparison["imbalance_difference"] = (
        comparison["raw_imbalance"] - comparison["clean_imbalance"]
    )

    print("\nLargest absolute imbalance changes:")
    largest = comparison.loc[
        comparison["imbalance_difference"].abs()
        .nlargest(10).index
    ].sort_values(
        "imbalance_difference",
        key=lambda x: x.abs(),
        ascending=False
    )

    print(largest.to_string())

    # Sanity checks: normalized imbalance must be between -1 and +1.
    assert valid["raw_imbalance"].between(-1, 1).all()
    assert valid["clean_imbalance"].between(-1, 1).all()

    print("\nSanity checks passed: all imbalance values are within [-1, 1].")

print("\nAnalysis complete. Raw files were not modified.")
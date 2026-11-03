from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed")

clean = pd.read_parquet(
    DATA_DIR / "features_1min_deduplicated.parquet"
)

original = pd.read_parquet(
    DATA_DIR / "features_1min_original.parquet"
)

# --------------------------------------------------
# Basic integrity checks
# --------------------------------------------------

clean = clean.sort_values("minute").reset_index(drop=True)
original = original.sort_values("minute").reset_index(drop=True)

assert clean["minute"].equals(original["minute"]), (
    "The two datasets do not contain identical minute timestamps."
)

assert not clean["minute"].duplicated().any()
assert not original["minute"].duplicated().any()

print("Both feature datasets have matching minute timestamps.")
print(f"Number of minute bars: {len(clean):,}")


# --------------------------------------------------
# Volume imbalance distributions
# --------------------------------------------------

def imbalance_summary(data, label):
    imbalance = data["volume_imbalance"].dropna()

    print(f"\n========== {label}: IMBALANCE ==========")
    print(f"Mean: {imbalance.mean():.6f}")
    print(f"Median: {imbalance.median():.6f}")
    print(f"Standard deviation: {imbalance.std():.6f}")
    print(f"Minimum: {imbalance.min():.6f}")
    print(f"Maximum: {imbalance.max():.6f}")

    print("\nPercentiles:")
    print(
        imbalance.quantile(
            [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        ).to_string()
    )

    print("\nDirection:")
    print(f"Positive: {(imbalance > 0).sum():,}")
    print(f"Negative: {(imbalance < 0).sum():,}")
    print(f"Zero: {(imbalance == 0).sum():,}")


imbalance_summary(clean, "DEDUPLICATED")
imbalance_summary(original, "ORIGINAL")


# --------------------------------------------------
# Aggregate vs. average minute imbalance
# --------------------------------------------------

for label, data in [
    ("DEDUPLICATED", clean),
    ("ORIGINAL", original),
]:
    total_buy = data["buy_volume"].sum()
    total_sell = data["sell_volume"].sum()

    aggregate_imbalance = (
        (total_buy - total_sell) / (total_buy + total_sell)
    )

    print(f"\n{label}")
    print(f"Aggregate volume imbalance: {aggregate_imbalance:.6f}")
    print(
        "Unweighted mean minute imbalance: "
        f"{data['volume_imbalance'].mean():.6f}"
    )


# --------------------------------------------------
# Price movement distribution
# --------------------------------------------------

returns = clean["close_change_points"].dropna()

print("\n========== ONE-MINUTE PRICE CHANGES ==========")
print(returns.describe().to_string())

print("\nPositive changes:", (returns > 0).sum())
print("Negative changes:", (returns < 0).sum())
print("Zero changes:", (returns == 0).sum())


# --------------------------------------------------
# Duplicate-removal sensitivity
# --------------------------------------------------

comparison = pd.DataFrame({
    "minute": clean["minute"],
    "clean_imbalance": clean["volume_imbalance"],
    "original_imbalance": original["volume_imbalance"],
    "clean_net_volume": clean["net_volume"],
    "original_net_volume": original["net_volume"],
})

comparison["imbalance_difference"] = (
    comparison["original_imbalance"]
    - comparison["clean_imbalance"]
)

comparison["net_volume_difference"] = (
    comparison["original_net_volume"]
    - comparison["clean_net_volume"]
)

difference = comparison["imbalance_difference"]

print("\n========== DUPLICATE SENSITIVITY ==========")
print(f"Correlation of minute imbalance: "
      f"{comparison['clean_imbalance'].corr(comparison['original_imbalance']):.6f}")
print(f"Mean absolute imbalance difference: "
      f"{difference.abs().mean():.6f}")
print(f"Median absolute imbalance difference: "
      f"{difference.abs().median():.6f}")
print(f"95th percentile absolute difference: "
      f"{difference.abs().quantile(0.95):.6f}")
print(f"Maximum absolute difference: "
      f"{difference.abs().max():.6f}")
print(f"Minutes with changed imbalance: "
      f"{(difference != 0).sum():,}")


# --------------------------------------------------
# Largest changes caused by duplicate removal
# --------------------------------------------------

print("\n========== 10 MOST AFFECTED MINUTES ==========")

largest = comparison.reindex(
    difference.abs().sort_values(ascending=False).index
).head(10)

print(largest.to_string(index=False))

print("\nFeature inspection complete.")
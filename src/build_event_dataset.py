from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_features_deduplicated.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "event_dataset_exploratory.parquet"
)

# Working definition selected from our earlier diagnostics.
NET_VOLUME_CUTOFF = 73.0
IMBALANCE_CUTOFF = 0.2344

WEAK_RESPONSE_PERCENTILE = 0.25
FORWARD_HORIZONS = [1, 5, 15]


# ------------------------------------------------------------
# Load and validate data
# ------------------------------------------------------------

df = pd.read_parquet(INPUT_PATH)

required_columns = [
    "minute",
    "close",
    "net_volume",
    "orderflow_direction",
    "absolute_net_volume",
    "absolute_imbalance",
    "response_per_100_net_contracts",
]

missing_columns = [
    column for column in required_columns if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df.copy()
df["minute"] = pd.to_datetime(df["minute"], utc=True)
df = df.sort_values("minute").reset_index(drop=True)

if df["minute"].isna().any():
    raise ValueError("Minute timestamps contain missing values.")

if df["minute"].duplicated().any():
    raise ValueError("Duplicate minute timestamps detected.")

if df["close"].isna().any():
    raise ValueError("Close prices contain missing values.")

if not df["minute"].is_monotonic_increasing:
    raise ValueError("Minute timestamps are not sorted.")

# Confirm that the stored direction agrees with net volume.
expected_direction = np.sign(df["net_volume"]).astype(int)
actual_direction = df["orderflow_direction"].astype(int)

if not np.array_equal(expected_direction, actual_direction):
    raise ValueError("orderflow_direction does not match sign(net_volume).")


# ------------------------------------------------------------
# Define strong-orderflow and weak-response events
# ------------------------------------------------------------

df["strong_orderflow"] = (
    (df["absolute_net_volume"] >= NET_VOLUME_CUTOFF)
    & (df["absolute_imbalance"] >= IMBALANCE_CUTOFF)
)

# Absolute value measures the magnitude of price response per
# 100 net contracts, regardless of whether price moved up or down.
df["absolute_response_per_100"] = (
    df["response_per_100_net_contracts"].abs()
)

strong_flow_valid_ratio = df.loc[
    df["strong_orderflow"],
    "absolute_response_per_100",
].dropna()

if strong_flow_valid_ratio.empty:
    raise ValueError("No strong-orderflow observations have a valid response ratio.")

weak_response_cutoff = strong_flow_valid_ratio.quantile(
    WEAK_RESPONSE_PERCENTILE
)

df["weak_response"] = False

valid_ratio_mask = df["absolute_response_per_100"].notna()

df.loc[valid_ratio_mask, "weak_response"] = (
    df.loc[valid_ratio_mask, "strong_orderflow"]
    & (
        df.loc[valid_ratio_mask, "absolute_response_per_100"]
        <= weak_response_cutoff
    )
)

df["target_event"] = df["strong_orderflow"] & df["weak_response"]


# ------------------------------------------------------------
# Calculate forward returns without crossing missing minutes
# ------------------------------------------------------------

# Use the timestamp as the index so a future price is only matched
# to its exact expected minute.
close_by_minute = df.set_index("minute")["close"]

for horizon in FORWARD_HORIZONS:
    future_minutes = df["minute"] + pd.Timedelta(minutes=horizon)

    # Exact timestamp lookup: no forward filling or nearest-minute match.
    future_close = future_minutes.map(close_by_minute)

    # Require every intermediate minute to exist as well.
    complete_path = pd.Series(True, index=df.index)

    for offset in range(1, horizon + 1):
        expected_minutes = (
            df["minute"] + pd.Timedelta(minutes=offset)
        )
        complete_path &= expected_minutes.isin(close_by_minute.index)

    valid_target = complete_path & future_close.notna()

    forward_return = pd.Series(np.nan, index=df.index, dtype=float)
    forward_return.loc[valid_target] = (
        future_close.loc[valid_target].to_numpy()
        - df.loc[valid_target, "close"].to_numpy()
    )

    df[f"forward_return_{horizon}m"] = forward_return

    # Positive = future price moved with initial orderflow.
    # Negative = future price moved against initial orderflow.
    df[f"flow_aligned_return_{horizon}m"] = (
        df["orderflow_direction"] * df[f"forward_return_{horizon}m"]
    )

    # These labels are descriptive outcomes, not model inputs.
    df[f"outcome_{horizon}m"] = np.select(
        [
            df[f"flow_aligned_return_{horizon}m"] > 0,
            df[f"flow_aligned_return_{horizon}m"] < 0,
        ],
        [
            "continuation",
            "reversal",
        ],
        default="flat_or_unavailable",
    )

    df.loc[
        df[f"flow_aligned_return_{horizon}m"].isna(),
        f"outcome_{horizon}m",
    ] = "unavailable"


# ------------------------------------------------------------
# Validate and save
# ------------------------------------------------------------

if (df["weak_response"] & ~df["strong_orderflow"]).any():
    raise ValueError("Weak-response observations must also be strong-flow observations.")

if (df["target_event"] != (df["strong_orderflow"] & df["weak_response"])).any():
    raise ValueError("Target-event definition is inconsistent.")

for horizon in FORWARD_HORIZONS:
    aligned = df[f"flow_aligned_return_{horizon}m"].dropna()

    if not np.isfinite(aligned).all():
        raise ValueError(f"Non-finite forward returns found at {horizon} minutes.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUTPUT_PATH, index=False)


# ------------------------------------------------------------
# Report results
# ------------------------------------------------------------

strong_count = int(df["strong_orderflow"].sum())
event_count = int(df["target_event"].sum())

print("=" * 70)
print("EVENT DATASET BUILD COMPLETE")
print("=" * 70)

print(f"Total minute observations: {len(df):,}")
print(f"Strong-orderflow minutes: {strong_count:,}")
print(f"Strong-orderflow percentage: {strong_count / len(df):.2%}")

print(f"\nWeak-response cutoff: {weak_response_cutoff:.4f}")
print(f"Weak-response percentile: {WEAK_RESPONSE_PERCENTILE:.0%}")

print(f"\nStrong-flow + weak-response events: {event_count:,}")
if strong_count:
    print(f"Percentage of strong-flow minutes: {event_count / strong_count:.2%}")
if len(df):
    print(f"Percentage of all minutes: {event_count / len(df):.2%}")

print("\nEvent direction:")
print(
    df.loc[df["target_event"], "orderflow_direction"]
    .map({1: "Buy-initiated", -1: "Sell-initiated", 0: "Zero"})
    .value_counts()
    .to_string()
)

print("\nForward-return availability:")
for horizon in FORWARD_HORIZONS:
    column = f"forward_return_{horizon}m"
    available_all = int(df[column].notna().sum())
    available_events = int(
        df.loc[df["target_event"], column].notna().sum()
    )

    print(
        f"  {horizon:>2}-minute horizon: "
        f"{available_all:,} total; "
        f"{available_events:,} target events"
    )

print("\nSaved dataset:")
print(OUTPUT_PATH)
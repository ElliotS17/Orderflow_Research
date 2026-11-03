from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "event_dataset_exploratory.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "event_dataset_with_context.parquet"

df = pd.read_parquet(INPUT_PATH)
df["minute"] = pd.to_datetime(df["minute"], utc=True)
df = df.sort_values("minute").reset_index(drop=True)

if df["minute"].duplicated().any():
    raise ValueError("Duplicate minute timestamps found.")

required_columns = ["close", "orderflow_direction"]
missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

minutes = pd.DatetimeIndex(df["minute"])
close = pd.Series(df["close"].to_numpy(), index=minutes)

# Recent momentum: price change over exact, uninterrupted windows.
for horizon in [5, 15]:
    past_minutes = minutes - pd.Timedelta(minutes=horizon)
    complete_path = np.ones(len(df), dtype=bool)

    for offset in range(1, horizon + 1):
        prior_minutes = minutes - pd.Timedelta(minutes=offset)
        complete_path &= prior_minutes.isin(minutes)

    past_close = close.reindex(past_minutes).to_numpy()
    momentum = df["close"].to_numpy() - past_close
    momentum[~complete_path] = np.nan

    df[f"momentum_{horizon}m"] = momentum

    # Positive means recent price movement aligned with current orderflow.
    df[f"flow_aligned_momentum_{horizon}m"] = (
        df["orderflow_direction"] * momentum
    )

# One-minute log returns, valid only across consecutive minutes.
log_returns = np.log(close / close.shift(1))
consecutive_minutes = minutes.to_series().diff() == pd.Timedelta(minutes=1)
log_returns.loc[~consecutive_minutes.to_numpy()] = np.nan

# Realized volatility over the previous 15 completed minutes.
# Require all 15 one-minute returns; do not fill missing returns with zero.
realized_variance = (
    log_returns.rolling("15min", closed="left", min_periods=15)
    .apply(lambda values: np.sum(values ** 2), raw=True)
)

df["realized_volatility_15m_bps"] = (
    np.sqrt(realized_variance.to_numpy()) * 10_000
)

# New York regular US equity cash-market hours.
new_york_time = df["minute"].dt.tz_convert("America/New_York")
ny_minutes = new_york_time.dt.hour * 60 + new_york_time.dt.minute

df["is_us_cash_hours"] = (
    (ny_minutes >= 9 * 60 + 30) & (ny_minutes < 16 * 60)
)

# Basic checks.
feature_columns = [
    "momentum_5m",
    "momentum_15m",
    "flow_aligned_momentum_5m",
    "flow_aligned_momentum_15m",
    "realized_volatility_15m_bps",
]

for column in feature_columns:
    if np.isinf(df[column].dropna()).any():
        raise ValueError(f"Infinite values found in {column}.")

if (df["realized_volatility_15m_bps"].dropna() < 0).any():
    raise ValueError("Realized volatility cannot be negative.")

df.to_parquet(OUTPUT_PATH, index=False)

# Report feature coverage and time-of-day distribution.
print("=" * 60)
print("CONTEXT FEATURE DATASET")
print("=" * 60)
print(f"Rows: {len(df):,}")
print(f"Date range: {df['minute'].min()} to {df['minute'].max()}")

print("\nFeature availability:")
for column in feature_columns:
    print(f"  {column}: {df[column].notna().sum():,} valid")

print("\nUS cash hours:")
print(f"  {df['is_us_cash_hours'].sum():,} minutes")

print("\nSaved:")
print(OUTPUT_PATH)
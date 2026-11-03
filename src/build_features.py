from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROCESSED_DIR = Path("data/processed")

INPUTS = {
    "deduplicated": PROCESSED_DIR / "trades_deduplicated.parquet",
    "original": PROCESSED_DIR / "trades_original.parquet",
}

OUTPUTS = {
    "deduplicated": PROCESSED_DIR / "features_1min_deduplicated.parquet",
    "original": PROCESSED_DIR / "features_1min_original.parquet",
}


# --------------------------------------------------
# Feature construction
# --------------------------------------------------

def build_features(input_path):
    print(f"\nReading {input_path}...")
    trades = pd.read_parquet(input_path)

    if trades.empty:
        raise RuntimeError(f"No trade records found in {input_path}")

    required = {"ts_event", "price", "size", "side"}
    missing = required - set(trades.columns)

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Ensure event timestamps are timezone-aware UTC.
    trades["ts_event"] = pd.to_datetime(
        trades["ts_event"], utc=True
    )

    # Sort trades so OHLC prices follow chronological trade order.
    trades = trades.sort_values(
        ["ts_event", "ts_recv"],
        kind="stable"
    ).reset_index(drop=True)

    trades["size"] = trades["size"].astype("int64")
    trades["price"] = trades["price"].astype("float64")

    if (trades["size"] < 0).any():
        raise RuntimeError("Negative trade sizes detected.")

    if not np.isfinite(trades["price"]).all():
        raise RuntimeError("Non-finite trade prices detected.")

    if not trades["side"].isin(["B", "A", "N"]).all():
        raise RuntimeError("Unexpected trade-side values detected.")

    # Assign every trade to its UTC calendar minute.
    trades["minute"] = trades["ts_event"].dt.floor("min")

    # Signed and directional trade volume.
    trades["_buy_volume"] = trades["size"].where(
        trades["side"] == "B", 0
    )

    trades["_sell_volume"] = trades["size"].where(
        trades["side"] == "A", 0
    )

    trades["_unknown_volume"] = trades["size"].where(
        trades["side"] == "N", 0
    )

    # Side-specific trade sizes for composition features.
    trades["_buy_size"] = trades["size"].where(
        trades["side"] == "B"
    )

    trades["_sell_size"] = trades["size"].where(
        trades["side"] == "A"
    )

    # Aggregate trades into one-minute bars.
    features = (
        trades.groupby("minute", sort=True)
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),

            buy_volume=("_buy_volume", "sum"),
            sell_volume=("_sell_volume", "sum"),
            unknown_volume=("_unknown_volume", "sum"),

            trade_count=("size", "size"),
            average_trade_size=("size", "mean"),
            median_trade_size=("size", "median"),
            maximum_trade_size=("size", "max"),

            buy_trade_count=("_buy_size", "count"),
            sell_trade_count=("_sell_size", "count"),
            unknown_trade_count=("side", lambda x: (x == "N").sum()),

            average_buy_trade_size=("_buy_size", "mean"),
            average_sell_trade_size=("_sell_size", "mean"),
        )
        .reset_index()
    )

    # --------------------------------------------------
    # Directional orderflow
    # --------------------------------------------------

    features["total_volume"] = (
        features["buy_volume"]
        + features["sell_volume"]
        + features["unknown_volume"]
    )

    features["directional_volume"] = (
        features["buy_volume"] + features["sell_volume"]
    )

    features["net_volume"] = (
        features["buy_volume"] - features["sell_volume"]
    )

    # Unknown-side volume is excluded from the imbalance denominator.
    features["volume_imbalance"] = (
        features["net_volume"]
        / features["directional_volume"].replace(0, np.nan)
    )

    # --------------------------------------------------
    # Price behavior and response
    # --------------------------------------------------

    features["price_range_points"] = (
        features["high"] - features["low"]
    )

    # Minutes with no trades are absent from this table.
    # Only calculate close-to-close returns across adjacent minutes.
    features["gap_minutes"] = (
        features["minute"].diff().dt.total_seconds() / 60
    )

    consecutive_minutes = features["gap_minutes"].eq(1)

    features["close_change_points"] = (
        features["close"].diff().where(consecutive_minutes)
    )

    previous_close = features["close"].shift(1)

    features["close_return_pct"] = (
        (features["close"] / previous_close - 1) * 100
    ).where(consecutive_minutes)

    features["absolute_close_change_points"] = (
        features["close_change_points"].abs()
    )

    # --------------------------------------------------
    # Time context
    # --------------------------------------------------

    chicago_time = features["minute"].dt.tz_convert(
        "America/Chicago"
    )

    features["weekday"] = chicago_time.dt.dayofweek
    features["hour_chicago"] = chicago_time.dt.hour

    features["minute_of_day_chicago"] = (
        chicago_time.dt.hour * 60
        + chicago_time.dt.minute
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    if features["minute"].duplicated().any():
        raise RuntimeError("Duplicate minute bars detected.")

    valid_imbalance = features["volume_imbalance"].dropna()

    if not valid_imbalance.between(-1, 1).all():
        raise RuntimeError("Volume imbalance outside [-1, 1].")

    if not (
        features["low"] <= features["open"]
    ).all():
        raise RuntimeError("Invalid OHLC open/low relationship.")

    if not (
        features["low"] <= features["close"]
    ).all():
        raise RuntimeError("Invalid OHLC close/low relationship.")

    if not (
        features["high"] >= features["open"]
    ).all():
        raise RuntimeError("Invalid OHLC high/open relationship.")

    if not (
        features["high"] >= features["close"]
    ).all():
        raise RuntimeError("Invalid OHLC high/close relationship.")

    # Confirm volume aggregation reconciles with trade-level data.
    source_buy = int(trades["_buy_volume"].sum())
    source_sell = int(trades["_sell_volume"].sum())
    source_unknown = int(trades["_unknown_volume"].sum())

    assert int(features["buy_volume"].sum()) == source_buy
    assert int(features["sell_volume"].sum()) == source_sell
    assert int(features["unknown_volume"].sum()) == source_unknown

    return features


# --------------------------------------------------
# Build both versions
# --------------------------------------------------

for version, input_path in INPUTS.items():
    features = build_features(input_path)

    output_path = OUTPUTS[version]
    features.to_parquet(output_path, index=False)

    print(f"\n========== {version.upper()} FEATURES ==========")
    print(f"Minute bars: {len(features):,}")
    print(f"Buy volume: {int(features['buy_volume'].sum()):,}")
    print(f"Sell volume: {int(features['sell_volume'].sum()):,}")
    print(f"Unknown volume: {int(features['unknown_volume'].sum()):,}")
    print(f"Minutes with valid imbalance: "
          f"{features['volume_imbalance'].notna().sum():,}")
    print(f"Minutes with consecutive-minute returns: "
          f"{features['close_change_points'].notna().sum():,}")
    print(f"Mean volume imbalance: "
          f"{features['volume_imbalance'].mean():.6f}")
    print(f"Saved: {output_path}")

print("\nFeature construction complete.")
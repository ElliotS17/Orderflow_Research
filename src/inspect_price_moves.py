from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

features = pd.read_parquet(
    DATA_DIR / "features_1min_deduplicated.parquet"
)

trades = pd.read_parquet(
    DATA_DIR / "trades_deduplicated.parquet"
)

features["minute"] = pd.to_datetime(features["minute"], utc=True)
trades["ts_event"] = pd.to_datetime(trades["ts_event"], utc=True)

features = features.sort_values("minute").reset_index(drop=True)
features["previous_close"] = features["close"].shift(1)
features["previous_minute"] = features["minute"].shift(1)

# Keep only consecutive active calendar minutes.
extremes = features[
    features["close_change_points"].notna()
].copy()

extremes["absolute_change"] = (
    extremes["close_change_points"].abs()
)

largest = extremes.nlargest(15, "absolute_change")

print("========== 15 LARGEST ONE-MINUTE PRICE CHANGES ==========")

columns = [
    "previous_minute",
    "minute",
    "previous_close",
    "close",
    "close_change_points",
    "open",
    "high",
    "low",
    "buy_volume",
    "sell_volume",
    "volume_imbalance",
]

print(largest[columns].to_string(index=False))

# Inspect actual trades for the five largest moves.
print("\n========== UNDERLYING TRADES FOR FIVE LARGEST MOVES ==========")

for _, row in largest.head(5).iterrows():
    start = row["minute"] - pd.Timedelta(minutes=1)
    end = row["minute"] + pd.Timedelta(minutes=1)

    interval_trades = trades[
        (trades["ts_event"] >= start)
        & (trades["ts_event"] < end)
    ].copy()

    interval_trades = interval_trades.sort_values(
        ["ts_event", "ts_recv"],
        kind="stable"
    )

    print("\n----------------------------------------")
    print("Previous minute:", start)
    print("Current minute:", row["minute"])
    print("Close change:", row["close_change_points"], "points")
    print("Trades inspected:", len(interval_trades))

    print("\nFirst five trades:")
    print(
        interval_trades[
            ["ts_event", "price", "size", "side", "sequence"]
        ].head(5).to_string(index=False)
    )

    print("\nLast five trades:")
    print(
        interval_trades[
            ["ts_event", "price", "size", "side", "sequence"]
        ].tail(5).to_string(index=False)
    )

print("\nPrice-move inspection complete.")
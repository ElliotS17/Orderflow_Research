from pathlib import Path

import pandas as pd
import numpy as np


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRADES_PATH = PROJECT_ROOT / "data" / "processed" / "trades_deduplicated.parquet"
BASE_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features_1min_deduplicated.parquet"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_1min_universe.parquet"


def load_data():
    """Load the deduplicated trade data and existing minute-level features."""

    trade_columns = [
        "ts_event",
        "ts_recv",
        "price",
        "size",
        "side",
    ]

    trades = pd.read_parquet(TRADES_PATH, columns=trade_columns)
    base_features = pd.read_parquet(BASE_FEATURES_PATH)

    return trades, base_features


def validate_data(trades, base_features):
    """Check that the input datasets contain the expected data."""

    required_trade_columns = {
        "ts_event",
        "ts_recv",
        "price",
        "size",
        "side",
    }

    required_base_columns = {
        "minute",
        "total_volume",
        "buy_volume",
        "sell_volume",
        "net_volume",
        "volume_imbalance",
    }

    assert required_trade_columns.issubset(trades.columns)
    assert required_base_columns.issubset(base_features.columns)

    assert trades["ts_event"].notna().all()
    assert trades["price"].notna().all()
    assert trades["size"].notna().all()
    assert trades["side"].isin(["B", "A", "N"]).all()

    assert base_features["minute"].is_unique

    print("Input validation passed.")

def prepare_trades(trades):
    """Prepare trade records for minute-level feature calculations."""

    trades = trades.copy()

    trades = trades.sort_values(
        ["ts_event", "ts_recv"],
        kind="stable"
    )

    trades["minute"] = trades["ts_event"].dt.floor("min")

    trades["size"] = trades["size"].astype("int64")

    return trades

def calculate_trade_size_features(trades):
    """Calculate minute-level trade-size distribution and concentration features."""

    grouped = trades.groupby("minute", sort=False)["size"]

    total_volume = grouped.sum()

    features = pd.DataFrame(index=total_volume.index)

    # Trade-size distribution
    features["trade_size_std"] = grouped.std()
    features["trade_size_p90"] = grouped.quantile(0.90)
    features["trade_size_p95"] = grouped.quantile(0.95)

    # Large-trade activity
    for threshold in [3, 5, 10]:
        large_trades = trades["size"] >= threshold

        large_count = (
            large_trades.groupby(trades["minute"], sort=False).sum()
        )

        large_volume = (
            trades["size"]
            .where(large_trades, 0)
            .groupby(trades["minute"], sort=False)
            .sum()
        )

        features[f"large_trade_count_{threshold}"] = large_count
        features[f"large_trade_volume_share_{threshold}"] = (
            large_volume / total_volume
        )

    # Trade-size concentration
    features["trade_size_hhi"] = (
        (trades["size"] ** 2)
        .groupby(trades["minute"], sort=False)
        .sum()
        / total_volume.pow(2)
    )

    features["largest_trade_volume_share"] = grouped.max() / total_volume

    top_five_volume = (
        grouped.nlargest(5)
        .groupby(level=0)
        .sum()
    )

    features["top_5_trade_volume_share"] = (
        top_five_volume / total_volume
    )

    return features

def calculate_timing_features(trades):
    """Calculate minute-level timing and activity concentration features."""

    trades = trades.copy()

    # Seconds elapsed since the beginning of each minute
    trades["seconds_into_minute"] = (
        trades["ts_event"] - trades["minute"]
    ).dt.total_seconds()

    # Directional trade volume
    trades["_net_volume"] = np.where(
        trades["side"] == "B",
        trades["size"],
        np.where(trades["side"] == "A", -trades["size"], 0)
    )

    trades["_directional_volume"] = np.where(
        trades["side"].isin(["B", "A"]),
        trades["size"],
        0
    )

    total_volume = trades.groupby("minute")["size"].sum()

    features = pd.DataFrame(index=total_volume.index)

    # Early and late activity
    for period, mask in [
        ("early", trades["seconds_into_minute"] < 30),
        ("late", trades["seconds_into_minute"] >= 30),
    ]:
        period_trades = trades.loc[mask]

        grouped = period_trades.groupby("minute")

        features[f"{period}_trade_count"] = grouped.size()
        features[f"{period}_volume"] = grouped["size"].sum()
        features[f"{period}_net_volume"] = grouped["_net_volume"].sum()
        features[f"{period}_directional_volume"] = (
            grouped["_directional_volume"].sum()
        )

    # Convert missing activity in a period to zero
    features = features.fillna(0)

    features["early_volume_share"] = (
        features["early_volume"] / total_volume
    )

    features["late_volume_share"] = (
        features["late_volume"] / total_volume
    )

    early_pressure = (
        features["early_net_volume"]
        / features["early_directional_volume"].replace(0, np.nan)
    )

    late_pressure = (
        features["late_net_volume"]
        / features["late_directional_volume"].replace(0, np.nan)
    )

    features["late_early_pressure_diff"] = (
        late_pressure - early_pressure
    )

    # Volume-weighted average timing of activity
    weighted_time = (
        trades["size"] * trades["seconds_into_minute"]
    ).groupby(trades["minute"]).sum()

    features["activity_time_centroid"] = weighted_time / total_volume

    # Fixed-window activity concentration
    for window_size in [5, 10]:
        trades["_window"] = (
            trades["seconds_into_minute"] // window_size
        ).astype(int)

        window_data = (
            trades.groupby(["minute", "_window"])["size"]
            .agg(volume="sum", trade_count="size")
        )

        window_volume = window_data["volume"].unstack(
            level="_window"
        )

        window_count = window_data["trade_count"].unstack(
            level="_window"
        )

        number_of_windows = 60 // window_size

        # Include windows with zero activity
        window_volume = window_volume.reindex(
            index=total_volume.index,
            columns=range(number_of_windows),
            fill_value=0
        ).fillna(0)

        window_count = window_count.reindex(
            index=total_volume.index,
            columns=range(number_of_windows),
            fill_value=0
        ).fillna(0)

        features[f"max_{window_size}s_volume_share"] = (
            window_volume.max(axis=1) / total_volume
        )

        features[f"max_{window_size}s_trade_count"] = (
            window_count.max(axis=1)
        )

        if window_size == 5:
            features["trade_intensity_cv"] = (
                window_count.std(axis=1, ddof=0)
                / window_count.mean(axis=1)
            )

    return features

def calculate_sequencing_features(trades):
    """Calculate minute-level directional sequencing and persistence features."""

    trades = trades.copy()

    all_minutes = pd.Index(
        trades["minute"].unique(),
        name="minute"
    )

    # Identify the immediately preceding trade's side within each minute
    previous_side = trades.groupby("minute")["side"].shift(1)

    is_directional = trades["side"].isin(["B", "A"])
    previous_is_directional = previous_side.isin(["B", "A"])

    # Count transitions only when both adjacent trades have known sides
    eligible_transition = is_directional & previous_is_directional

    side_switch = (
        eligible_transition
        & trades["side"].ne(previous_side)
    )

    transition_count = (
        eligible_transition.groupby(trades["minute"])
        .sum()
        .reindex(all_minutes, fill_value=0)
    )

    switch_count = (
        side_switch.groupby(trades["minute"])
        .sum()
        .reindex(all_minutes, fill_value=0)
    )

    features = pd.DataFrame(index=all_minutes)

    features["directional_transition_count"] = transition_count
    features["side_switch_count"] = switch_count

    features["side_switch_rate"] = (
        switch_count / transition_count.replace(0, np.nan)
    )

    features["directional_persistence"] = (
        1 - features["side_switch_rate"]
    )

    # Identify the beginning of each uninterrupted directional run.
    # Unknown-side trades break runs.
    run_start = (
        is_directional
        & (
            trades["side"].ne(previous_side)
            | previous_side.isna()
        )
    )

    trades["run_id"] = run_start.cumsum()

    # Calculate the length of each directional run
    run_data = (
        trades.loc[is_directional]
        .groupby(["minute", "run_id", "side"])
        .size()
        .rename("run_length")
        .reset_index()
    )

    # Summarize buy and sell runs separately
    for side, label in [("B", "buy"), ("A", "sell")]:
        side_runs = run_data.loc[run_data["side"] == side]

        run_summary = (
            side_runs.groupby("minute")["run_length"]
            .agg(["count", "mean", "max"])
            .reindex(all_minutes)
        )

        features[f"{label}_run_count"] = (
            run_summary["count"].fillna(0)
        )

        features[f"avg_{label}_run_length"] = run_summary["mean"]

        features[f"max_{label}_run_length"] = (
            run_summary["max"].fillna(0)
        )

    features["max_directional_run_length"] = features[
        ["max_buy_run_length", "max_sell_run_length"]
    ].max(axis=1)

    return features

def calculate_price_path_features(trades):
    """Calculate intraminute price-path features without duplicating OHLC."""

    trades = trades.copy()

    grouped = trades.groupby("minute", sort=False)["price"]

    features = pd.DataFrame(index=grouped.size().index)

    # First and last trade prices are used internally, not returned
    first_price = grouped.first()
    last_price = grouped.last()
    high_price = grouped.max()
    low_price = grouped.min()

    # Intraminute price displacement
    displacement = last_price - first_price

    # Maximum upward and downward excursions from the first trade
    features["max_upward_excursion"] = high_price - first_price
    features["max_downward_excursion"] = first_price - low_price

    # How much price recovered from its intraminute low
    features["recovery_from_low"] = last_price - low_price

    # How much price pulled back from its intraminute high
    features["pullback_from_high"] = high_price - last_price

    # Total absolute price movement between consecutive trades
    price_changes = grouped.diff()

    path_length = (
        price_changes.abs()
        .groupby(trades["minute"])
        .sum()
    )

    features["price_path_length"] = path_length

    # Net displacement relative to total accumulated price movement
    features["price_path_efficiency"] = (
        displacement.abs() / path_length.replace(0, np.nan)
    )

    # Count directional reversals, ignoring unchanged trade prices
    price_direction = np.sign(price_changes)

    nonzero_direction = price_direction.where(
        price_direction != 0
    )

    previous_nonzero_direction = (
        nonzero_direction
        .groupby(trades["minute"])
        .ffill()
        .groupby(trades["minute"])
        .shift()
    )

    reversal = (
        nonzero_direction.notna()
        & previous_nonzero_direction.notna()
        & nonzero_direction.ne(previous_nonzero_direction)
    )

    features["price_reversal_count"] = (
        reversal.groupby(trades["minute"])
        .sum()
    )

    return features

def calculate_price_level_features(trades):
    """Calculate minute-level price distribution and VWAP features."""

    price_levels = (
        trades.groupby(["minute", "price"])["size"]
        .sum()
    )

    total_volume = trades.groupby("minute")["size"].sum()

    features = pd.DataFrame(index=total_volume.index)

    # Price-level count
    features["distinct_price_levels"] = (
        price_levels.groupby(level="minute").size()
    )

    # Volume share at each price level
    price_shares = (
        price_levels
        / total_volume.reindex(price_levels.index, level="minute")
    )

    # Price-level concentration
    features["price_level_hhi"] = (
        price_shares.pow(2)
        .groupby(level="minute")
        .sum()
    )

    features["largest_price_level_share"] = (
        price_shares.groupby(level="minute").max()
    )

    # Normalized Shannon entropy
    entropy = (
        -(price_shares * np.log(price_shares))
        .groupby(level="minute")
        .sum()
    )

    features["price_level_entropy"] = (
        entropy / np.log(features["distinct_price_levels"])
    )

    # Identify the Point of Control (highest-volume price)
    profile = price_levels.rename("volume").reset_index()

    poc_rows = profile.loc[
        profile.groupby("minute")["volume"].idxmax()
    ]

    poc_price = poc_rows.set_index("minute")["price"]

    features["point_of_control_price"] = poc_price

    # VWAP
    weighted_price = (
        trades["price"] * trades["size"]
    ).groupby(trades["minute"]).sum()

    features["vwap"] = weighted_price / total_volume

    last_price = trades.groupby("minute")["price"].last()

    features["close_vwap_displacement"] = (
        last_price - features["vwap"]
    )

    features["close_poc_displacement"] = (
        last_price - poc_price
    )

    # Volume concentration in upper and lower thirds of price range
    bounds = trades.groupby("minute")["price"].agg(
        low="min",
        high="max"
    )

    profile = profile.merge(
        bounds,
        left_on="minute",
        right_index=True
    )

    price_range = profile["high"] - profile["low"]

    lower_boundary = profile["low"] + price_range / 3
    upper_boundary = profile["low"] + 2 * price_range / 3

    lower_volume = (
        profile.loc[profile["price"] <= lower_boundary]
        .groupby("minute")["volume"]
        .sum()
    )

    upper_volume = (
        profile.loc[profile["price"] >= upper_boundary]
        .groupby("minute")["volume"]
        .sum()
    )

    features["lower_range_volume_share"] = (
        lower_volume.reindex(total_volume.index, fill_value=0)
        / total_volume
    )

    features["upper_range_volume_share"] = (
        upper_volume.reindex(total_volume.index, fill_value=0)
        / total_volume
    )

        # Aggressor-specific volume-weighted execution prices.
    directional_trades = trades.loc[
        trades["side"].isin(["B", "A"])
    ].copy()

    directional_trades["weighted_price"] = (
        directional_trades["price"] * directional_trades["size"]
    )

    side_aggregates = (
        directional_trades
        .groupby(["minute", "side"])
        .agg(
            volume=("size", "sum"),
            weighted_price=("weighted_price", "sum")
        )
    )

    side_volume = (
        side_aggregates["volume"]
        .unstack("side")
        .reindex(index=total_volume.index, columns=["B", "A"])
    )

    side_weighted_price = (
        side_aggregates["weighted_price"]
        .unstack("side")
        .reindex(index=total_volume.index, columns=["B", "A"])
    )

    buy_vwap = side_weighted_price["B"] / side_volume["B"]
    sell_vwap = side_weighted_price["A"] / side_volume["A"]

    features["buy_vwap"] = buy_vwap
    features["sell_vwap"] = sell_vwap
    features["buy_sell_vwap_diff"] = buy_vwap - sell_vwap

    # Aggressor-specific volume distribution within the minute's price range.
    lower_boundary_by_minute = (
        bounds["low"] + (bounds["high"] - bounds["low"]) / 3
    )

    upper_boundary_by_minute = (
        bounds["low"] + 2 * (bounds["high"] - bounds["low"]) / 3
    )

    trade_lower_boundary = directional_trades["minute"].map(
        lower_boundary_by_minute
    )

    trade_upper_boundary = directional_trades["minute"].map(
        upper_boundary_by_minute
    )

    buy_upper_volume = (
        directional_trades.loc[
            (directional_trades["side"] == "B")
            & (directional_trades["price"] >= trade_upper_boundary)
        ]
        .groupby("minute")["size"]
        .sum()
        .reindex(total_volume.index, fill_value=0)
    )

    sell_lower_volume = (
        directional_trades.loc[
            (directional_trades["side"] == "A")
            & (directional_trades["price"] <= trade_lower_boundary)
        ]
        .groupby("minute")["size"]
        .sum()
        .reindex(total_volume.index, fill_value=0)
    )

    features["buy_upper_range_volume_share"] = (
        buy_upper_volume / side_volume["B"]
    )

    features["sell_lower_range_volume_share"] = (
        sell_lower_volume / side_volume["A"]
    )

    return features

def calculate_context_features(base_features):
    """Calculate rolling market-context features using prior minutes."""

    context = base_features.sort_values("minute").copy()

    # Identify new trading sessions after gaps greater than one minute.
    session_break = context["minute"].diff() > pd.Timedelta(minutes=1)
    context["session_id"] = session_break.cumsum()

    context = context.set_index("minute")

    grouped = context.groupby("session_id", sort=False)

    # Calculate minute-to-minute price movement within each session.
    abs_price_change = (
        grouped["close"].diff().abs()
    )

    minute_returns = grouped["close"].transform(
    lambda x: x.pct_change()
)

    features = pd.DataFrame(index=context.index)

    for window in [5, 15, 30]:

        # Previous-window orderflow.
        prior_net = grouped["net_volume"].transform(
            lambda x: x.shift(1).rolling(
                window=window, min_periods=window
            ).sum()
        )

        prior_directional = grouped["directional_volume"].transform(
            lambda x: x.shift(1).rolling(
                window=window, min_periods=window
            ).sum()
        )

        features[f"prior_{window}m_net_volume"] = prior_net

        features[f"prior_{window}m_directional_volume"] = (
            prior_directional
        )

        features[f"prior_{window}m_volume_imbalance"] = (
            prior_net / prior_directional
        ).where(prior_directional > 0)

        # Current activity relative to the previous window.
        prior_avg_volume = grouped["total_volume"].transform(
            lambda x: x.shift(1).rolling(
                window=window, min_periods=window
            ).mean()
        )

        prior_avg_trades = grouped["trade_count"].transform(
            lambda x: x.shift(1).rolling(
                window=window, min_periods=window
            ).mean()
        )

        features[f"relative_volume_{window}m"] = (
            context["total_volume"] / prior_avg_volume
        )

        features[f"relative_trade_count_{window}m"] = (
            context["trade_count"] / prior_avg_trades
        )

        # Recent price movement, excluding the current minute.
        prior_abs_change = abs_price_change.groupby(
            context["session_id"], sort=False
        ).transform(
            lambda x: x.shift(1).rolling(
                window=window, min_periods=window
            ).sum()
        )

            # Net price displacement over the previous window.
        prior_close = grouped["close"].transform(
            lambda x: x.shift(1)
        )

        reference_close = grouped["close"].transform(
            lambda x: x.shift(window + 1)
        )

        prior_price_change = prior_close - reference_close

        features[f"prior_{window}m_price_change"] = (
            prior_price_change
        )

        # Directional efficiency of the previous price movement.
        features[f"prior_{window}m_price_efficiency"] = (
            prior_price_change.abs() / prior_abs_change
        ).where(prior_abs_change > 0)

        features[f"prior_{window}m_abs_price_change"] = (
            prior_abs_change
        )
                # Previous-window high-low range.
        prior_high = grouped["high"].transform(
            lambda x: x.shift(1).rolling(
                window=window, min_periods=window
            ).max()
        )

        prior_low = grouped["low"].transform(
            lambda x: x.shift(1).rolling(
                window=window, min_periods=window
            ).min()
        )

        features[f"prior_{window}m_price_range"] = (
            prior_high - prior_low
        )

        # Variability of previous minute returns.
        prior_return_std = minute_returns.groupby(
            context["session_id"], sort=False
        ).transform(
            lambda x: x.shift(1).rolling(
                window=window, min_periods=window
            ).std()
        )

        features[f"prior_{window}m_return_std"] = (
            prior_return_std
        )

        # Percentage price change over the previous window.
        features[f"prior_{window}m_return"] = (
            prior_close / reference_close - 1
        )

    return features

def main():
    trades, base_features = load_data()

    validate_data(trades, base_features)

    trades = prepare_trades(trades)

# Construct size-based features
    size_features = calculate_trade_size_features(trades)

    # Validate trade-size features
    share_columns = [
        col for col in size_features.columns
        if "share" in col or col == "trade_size_hhi"
    ]

    assert size_features[share_columns].ge(0).all().all()
    assert size_features[share_columns].le(1).all().all()

    assert (
        size_features["large_trade_count_3"]
        >= size_features["large_trade_count_5"]
    ).all()

    assert (
        size_features["large_trade_count_5"]
        >= size_features["large_trade_count_10"]
    ).all()

    assert (
        size_features["large_trade_volume_share_3"]
        >= size_features["large_trade_volume_share_5"]
    ).all()

    assert (
        size_features["large_trade_volume_share_5"]
        >= size_features["large_trade_volume_share_10"]
    ).all()

    print("Trade-size feature validation passed.")

    print("\nTrade-size feature dataset:")
    print(size_features.shape)

    print("\nFeature summary:")
    print(size_features.describe().round(4))

    print("\nMissing values:")
    print(size_features.isna().sum())

    assert size_features.index.is_unique
    assert size_features.index.isin(base_features["minute"]).all()
    assert len(size_features) == len(base_features)

    print(f"\nTrades loaded: {len(trades):,}")
    print(f"Base feature rows: {len(base_features):,}")
    print(f"Base feature columns: {len(base_features.columns)}")

    print("\nTrade data types:")
    print(trades.dtypes)

    print("\nBaseline minute column type:")
    print(base_features["minute"].dtype)

# Construct timing-based features
    timing_features = calculate_timing_features(trades)

    print("\nTiming feature dataset:")
    print(timing_features.shape)

    print("\nFeature summary:")
    print(timing_features.describe().round(4))

    print("\nMissing values:")
    print(timing_features.isna().sum())

    assert timing_features.index.is_unique
    assert timing_features.index.isin(base_features["minute"]).all()
    assert len(timing_features) == len(base_features)

# Construct sequence-based features
    sequencing_features = calculate_sequencing_features(trades)

    print("\nSequencing feature dataset:")
    print(sequencing_features.shape)

    print("\nFeature summary:")
    print(sequencing_features.describe().round(4))

    print("\nMissing values:")
    print(sequencing_features.isna().sum())

    assert sequencing_features.index.is_unique
    assert sequencing_features.index.isin(base_features["minute"]).all()
    assert len(sequencing_features) == len(base_features)

    assert sequencing_features["side_switch_rate"].dropna().between(
        0, 1
    ).all()

    assert sequencing_features["directional_persistence"].dropna().between(
        0, 1
    ).all()

    assert (
        sequencing_features["side_switch_count"]
        <= sequencing_features["directional_transition_count"]
    ).all()

    print("Sequencing feature validation passed.")

# Construct path-based features
    price_path_features = calculate_price_path_features(trades)

    print("\nPrice-path feature dataset:")
    print(price_path_features.shape)

    print("\nFeature summary:")
    print(price_path_features.describe().round(4))

    print("\nMissing values:")
    print(price_path_features.isna().sum())

    assert price_path_features.index.is_unique
    assert price_path_features.index.isin(base_features["minute"]).all()
    assert len(price_path_features) == len(base_features)

    assert price_path_features["price_path_length"].ge(0).all()

    assert price_path_features[
        "price_path_efficiency"
    ].dropna().between(0, 1).all()

    assert price_path_features["max_upward_excursion"].ge(0).all()
    assert price_path_features["max_downward_excursion"].ge(0).all()

    print("Price-path feature validation passed.")

# Construct price-level-based features
    price_level_features = calculate_price_level_features(trades)

    aggressor_share_features = price_level_features[
        [
            "buy_upper_range_volume_share",
            "sell_lower_range_volume_share"
        ]
    ].dropna()

    assert aggressor_share_features.ge(0).all().all()
    assert aggressor_share_features.le(1).all().all()

    assert price_level_features["buy_vwap"].dropna().notna().all()
    assert price_level_features["sell_vwap"].dropna().notna().all()

    print("Aggressor-specific price feature validation passed.")

    print("\nPrice-level feature dataset:")
    print(price_level_features.shape)

    print("\nFeature summary:")
    print(price_level_features.describe().round(4))

    print("\nMissing values:")
    print(price_level_features.isna().sum())

    assert price_level_features.index.is_unique
    assert price_level_features.index.isin(base_features["minute"]).all()
    assert len(price_level_features) == len(base_features)

    assert price_level_features["distinct_price_levels"].ge(1).all()

    assert price_level_features["price_level_hhi"].between(0, 1).all()

    assert price_level_features[
        "largest_price_level_share"
    ].between(0, 1).all()

    assert price_level_features["price_level_entropy"].between(0, 1).all()

    assert price_level_features["vwap"].notna().all()

    assert price_level_features[
        ["lower_range_volume_share", "upper_range_volume_share"]
    ].ge(0).all().all()

    assert price_level_features[
        ["lower_range_volume_share", "upper_range_volume_share"]
    ].le(1).all().all()

    assert price_level_features["point_of_control_price"].notna().all()

    assert price_level_features["vwap"].notna().all()

    print("Price-level feature validation passed.")

# Construct contextual features
    context_features = calculate_context_features(base_features)

    print("\nContext feature dataset:")
    print(context_features.shape)

    print("\nFeature summary:")
    print(context_features.describe().T.round(4))

    print("\nMissing values:")
    print(context_features.isna().sum())

    # Validate nonnegative price ranges, ignoring expected NaN values.
    assert (
        context_features.filter(like="_price_range")
        .dropna()
        .ge(0)
        .all()
        .all()
    )

    # Validate nonnegative return standard deviations.
    assert (
        context_features.filter(like="_return_std")
        .dropna()
        .ge(0)
        .all()
        .all()
    )
    print("Context feature validation passed.")

        # Inventory all candidate feature families.
    feature_families = {
        "Trade Size": size_features,
        "Timing": timing_features,
        "Sequencing": sequencing_features,
        "Price Path": price_path_features,
        "Price Level": price_level_features,
        "Market Context": context_features,
    }

        # Mathematical redundancy audit.
    print("\n" + "=" * 60)
    print("MATHEMATICAL REDUNDANCY AUDIT")
    print("=" * 60)

    base_indexed = base_features.set_index("minute")

    def check_identity(name, left, right):
        comparison = pd.concat(
            [left.rename("left"), right.rename("right")],
            axis=1
        ).dropna()

        difference = (
            comparison["left"] - comparison["right"]
        ).abs()

        print(f"\n{name}")
        print("  Observations compared:", len(comparison))
        print("  Maximum absolute difference:", difference.max())
        print(
            "  Exact within tolerance:",
            np.allclose(
                comparison["left"],
                comparison["right"],
                rtol=1e-10,
                atol=1e-10
            )
        )

    # 1. Directional sequencing.
    check_identity(
        "Side switch rate + directional persistence = 1",
        sequencing_features["side_switch_rate"]
        + sequencing_features["directional_persistence"],
        pd.Series(1.0, index=sequencing_features.index)
    )

    # 2. Early/late activity partitions.
    check_identity(
        "Early + late trade count = total trade count",
        timing_features["early_trade_count"]
        + timing_features["late_trade_count"],
        base_indexed["trade_count"]
    )

    check_identity(
        "Maximum directional run = maximum buy/sell run",
        sequencing_features["max_directional_run_length"],
        sequencing_features[
            ["max_buy_run_length", "max_sell_run_length"]
        ].max(axis=1)
    )

    check_identity(
        "Early + late volume = total volume",
        timing_features["early_volume"]
        + timing_features["late_volume"],
        base_indexed["total_volume"]
    )

    check_identity(
        "Early + late net volume = total net volume",
        timing_features["early_net_volume"]
        + timing_features["late_net_volume"],
        base_indexed["net_volume"]
    )

    check_identity(
        "Early + late directional volume = total directional volume",
        timing_features["early_directional_volume"]
        + timing_features["late_directional_volume"],
        base_indexed["directional_volume"]
    )

    check_identity(
        "Early volume share + late volume share = 1",
        timing_features["early_volume_share"]
        + timing_features["late_volume_share"],
        pd.Series(1.0, index=timing_features.index)
    )

    # 3. Price-path relationships.
    check_identity(
        "Price path efficiency = absolute intraminute change / path length",
        price_path_features["price_path_efficiency"],
        (base_indexed["close"] - base_indexed["open"]).abs()
        / price_path_features["price_path_length"]
    )

    check_identity(
        "Recovery from low = close - low",
        price_path_features["recovery_from_low"],
        base_indexed["close"] - base_indexed["low"]
    )

    check_identity(
        "Pullback from high = high - close",
        price_path_features["pullback_from_high"],
        base_indexed["high"] - base_indexed["close"]
    )

    check_identity(
        "Maximum upward excursion = high - open",
        price_path_features["max_upward_excursion"],
        base_indexed["high"] - base_indexed["open"]
    )

    check_identity(
        "Maximum downward excursion = open - low",
        price_path_features["max_downward_excursion"],
        base_indexed["open"] - base_indexed["low"]
    )

    # 4. Price-level relationships.
    check_identity(
        "Buy-sell VWAP difference = buy VWAP - sell VWAP",
        price_level_features["buy_sell_vwap_diff"],
        price_level_features["buy_vwap"]
        - price_level_features["sell_vwap"]
    )

    check_identity(
        "Close-VWAP displacement = close - VWAP",
        price_level_features["close_vwap_displacement"],
        base_indexed["close"] - price_level_features["vwap"]
    )

    check_identity(
        "Close-POC displacement = close - POC",
        price_level_features["close_poc_displacement"],
        base_indexed["close"]
        - price_level_features["point_of_control_price"]
    )

        # Statistical redundancy audit.
    print("\n" + "=" * 60)
    print("STATISTICAL REDUNDANCY AUDIT")
    print("=" * 60)

    baseline_audit_columns = [
        "open",
        "high",
        "low",
        "close",
        "buy_volume",
        "sell_volume",
        "unknown_volume",
        "trade_count",
        "average_trade_size",
        "median_trade_size",
        "maximum_trade_size",
        "buy_trade_count",
        "sell_trade_count",
        "average_buy_trade_size",
        "average_sell_trade_size",
        "total_volume",
        "directional_volume",
        "net_volume",
        "volume_imbalance",
        "price_range_points",
        "close_change_points",
        "close_return_pct",
        "absolute_close_change_points",
    ]

    # Combine baseline measurements and all new feature families.
    audit_data = pd.concat(
        [
            base_indexed[baseline_audit_columns],
            *feature_families.values()
        ],
        axis=1
    )

    # Keep numeric columns only.
    audit_data = audit_data.select_dtypes(include="number")

    # Calculate pairwise Pearson correlations.
    correlation_matrix = audit_data.corr(
        method="pearson",
        min_periods=1000
    )

    # Extract each unique feature pair once.
    upper_triangle = np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)

    correlation_pairs = (
        correlation_matrix
        .where(upper_triangle)
        .stack()
        .reset_index()
    )

    correlation_pairs.columns = [
        "feature_1",
        "feature_2",
        "correlation"
    ]

    correlation_pairs = correlation_pairs.dropna(
        subset=["correlation"]
    )

    correlation_pairs["absolute_correlation"] = (
        correlation_pairs["correlation"].abs()
    )

    correlation_pairs = correlation_pairs.sort_values(
        "absolute_correlation",
        ascending=False
    )

    print("\nPairs with absolute correlation >= 0.95:")
    print(
        correlation_pairs[
            correlation_pairs["absolute_correlation"] >= 0.95
        ].to_string(index=False)
    )

    print("\nPairs with absolute correlation between 0.90 and 0.95:")
    print(
        correlation_pairs[
            (correlation_pairs["absolute_correlation"] >= 0.90)
            & (correlation_pairs["absolute_correlation"] < 0.95)
        ].to_string(index=False)
    )

    print("\nTotal numeric columns audited:", len(audit_data.columns))
    print("Total feature pairs evaluated:", len(correlation_pairs))

    print("\n" + "=" * 60)
    print("TARGETED REDUNDANCY INVESTIGATION")
    print("=" * 60)

    # Sequencing count relationships.
    print("\n--- Sequencing counts ---")

    sequencing_comparison = pd.DataFrame({
        "trade_count": base_indexed["trade_count"],
        "transitions": sequencing_features["directional_transition_count"],
        "switches": sequencing_features["side_switch_count"],
        "buy_runs": sequencing_features["buy_run_count"],
        "sell_runs": sequencing_features["sell_run_count"],
    })

    print(sequencing_comparison.describe().round(3))

    print("\nBuy-run minus sell-run count:")
    print(
        (sequencing_comparison["buy_runs"]
         - sequencing_comparison["sell_runs"])
        .value_counts()
        .sort_index()
    )

    # Unknown-side volume contribution.
    print("\n--- Unknown-side volume contribution ---")

    unknown_volume = pd.DataFrame({
        "total_unknown_volume": base_indexed["unknown_volume"],
        "early_unknown_volume": (
            timing_features["early_volume"]
            - timing_features["early_directional_volume"]
        ),
        "late_unknown_volume": (
            timing_features["late_volume"]
            - timing_features["late_directional_volume"]
        ),
    })

    print(unknown_volume.describe().round(4))

    # Aggressor VWAP displacements.
    print("\n--- Aggressor-specific price displacements ---")

    aggressor_displacements = pd.DataFrame({
        "buy_sell_vwap_diff": price_level_features["buy_sell_vwap_diff"],
        "close_vwap_displacement": price_level_features["close_vwap_displacement"],
        "close_poc_displacement": price_level_features["close_poc_displacement"],
    })

    print(
        aggressor_displacements.corr().round(4)
    )

    # Market-context volatility relationships.
    print("\n--- Market-context volatility ---")

    volatility_columns = [
        "prior_5m_abs_price_change",
        "prior_5m_return_std",
        "prior_15m_abs_price_change",
        "prior_15m_return_std",
        "prior_30m_abs_price_change",
        "prior_30m_return_std",
        "prior_5m_price_range",
        "prior_15m_price_range",
        "prior_30m_price_range",
    ]

    print(
        context_features[volatility_columns]
        .corr()
        .round(3)
    )

    print("\n" + "=" * 60)
    print("FINAL SEQUENCING AND VOLUME CHECKS")
    print("=" * 60)

    # 1. Difference between total trades and directional transitions.
    transition_difference = (
        base_indexed["trade_count"]
        - sequencing_features["directional_transition_count"]
    )

    print("\n--- Trade count vs directional transitions ---")
    print("Difference distribution:")
    print(transition_difference.value_counts().sort_index())

    # 2. Relationship between run counts and side switches.
    total_runs = (
        sequencing_features["buy_run_count"]
        + sequencing_features["sell_run_count"]
    )

    run_switch_difference = (
        total_runs - sequencing_features["side_switch_count"]
    )

    print("\n--- Total runs vs side switches ---")
    print(run_switch_difference.describe())

    print("\nDifference distribution:")
    print(run_switch_difference.value_counts().sort_index())

    # 3. Unknown-side volume as a proportion of total volume.
    unknown_volume_share = (
        base_indexed["unknown_volume"]
        / base_indexed["total_volume"]
    )

    print("\n--- Unknown-side volume share ---")
    print(unknown_volume_share.describe(
        percentiles=[0.50, 0.90, 0.95, 0.99, 0.999]
    ))

    print(
        "Minutes with unknown volume share > 1%:",
        (unknown_volume_share > 0.01).sum()
    )

    print(
        "Minutes with unknown volume share > 5%:",
        (unknown_volume_share > 0.05).sum()
    )

    print(
        "Minutes with unknown volume share > 10%:",
        (unknown_volume_share > 0.10).sum()
    )

    print("\n" + "=" * 60)
    print("COMPLETE FEATURE UNIVERSE")
    print("=" * 60)

    all_feature_names = []

    for family_name, feature_df in feature_families.items():
        names = feature_df.columns.tolist()
        all_feature_names.extend(names)

        print(f"\n{family_name}: {len(names)} features")

        for name in names:
            print(f"  - {name}")

    print("\nTotal new features:", len(all_feature_names))
    print("Unique feature names:", len(set(all_feature_names)))

    # Check for duplicate feature names across families.
    duplicates = (
        pd.Series(all_feature_names)
        .value_counts()
    )

    duplicates = duplicates[duplicates > 1]

    print("\nDuplicate feature names:")
    print(duplicates if not duplicates.empty else "None")

    # Check for overlap with baseline columns.
    baseline_columns = set(base_features.columns) - {"minute"}
    overlap = set(all_feature_names) & baseline_columns

    print("\nOverlap with baseline features:")
    print(sorted(overlap) if overlap else "None")


    # Build the complete minute-level feature universe.
    print("\n" + "=" * 60)
    print("BUILDING FINAL FEATURE UNIVERSE DATASET")
    print("=" * 60)

    base_indexed = base_features.set_index("minute").sort_index()

    # Validate each feature family's minute index.
    for family_name, feature_df in feature_families.items():
        assert feature_df.index.is_unique
        assert len(feature_df) == len(base_indexed)
        assert feature_df.index.isin(base_indexed.index).all()

        print(f"{family_name}: minute index validated.")

    # Validate feature names and baseline overlap.
    all_feature_names = [
        name
        for feature_df in feature_families.values()
        for name in feature_df.columns
    ]

    assert len(all_feature_names) == 95
    assert len(all_feature_names) == len(set(all_feature_names))

    baseline_columns = set(base_indexed.columns)
    assert not (set(all_feature_names) & baseline_columns)

    # Combine baseline and all new feature families.
    combined_features = pd.concat(
        [
            base_indexed,
            *[
                feature_df.reindex(base_indexed.index)
                for feature_df in feature_families.values()
            ],
        ],
        axis=1
    )

    combined_features = combined_features.sort_index()

    combined_features = combined_features.copy()

    # Validate the combined dataset before saving.
    assert combined_features.index.is_unique
    assert len(combined_features) == len(base_features)
    assert combined_features.columns.is_unique
    assert len(combined_features.columns) == 123

    combined_features.index.name = "minute"
    output_features = combined_features.reset_index()

    assert output_features.shape == (20580, 124)
    assert output_features["minute"].is_unique
    assert output_features["minute"].is_monotonic_increasing

    print("\nFinal dataset shape:", output_features.shape)
    print("Baseline columns:", len(base_features.columns))
    print("New features:", len(all_feature_names))
    print("Total columns:", len(output_features.columns))

    # Save the completed feature universe.
    output_features.to_parquet(OUTPUT_PATH, index=False)

    print("\nSaved dataset to:", OUTPUT_PATH)

    # Read the saved file back and validate the output.
    saved_features = pd.read_parquet(OUTPUT_PATH)

    assert saved_features.shape == output_features.shape
    assert saved_features.columns.equals(output_features.columns)
    assert saved_features["minute"].is_unique
    assert saved_features["minute"].equals(output_features["minute"])

    print("Saved dataset read-back validation passed.")
    print("Final feature universe construction complete.")

if __name__ == "__main__":
    main()

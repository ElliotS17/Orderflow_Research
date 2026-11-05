"""Build the focused order-flow confirmation research dataset.

Run from the repository root:
    python3 src/research/build_research_dataset.py

The script reads the existing feature universe and writes a research-specific
Parquet dataset. It does not modify the source feature universe.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features_1min_universe.parquet"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_orderflow_confirmation.parquet"
)

# Features are selected to represent the research question, not to maximize
# the number of columns. Keep this list fixed for the initial analysis.
FEATURE_COLUMNS = [
    # Baseline directional pressure
    "volume_imbalance",
    "directional_volume",
    # Trade concentration
    "trade_size_hhi",
    # Activity timing
    "max_10s_volume_share",
    "late_early_pressure_diff",
    # Directional sequencing
    "directional_persistence",
    # Intraminute price confirmation
    "price_path_efficiency",
    "close_vwap_displacement",
    "buy_sell_vwap_diff",
    # Prior market context
    "prior_5m_volume_imbalance",
    "prior_5m_return",
    "relative_volume_5m",
    "prior_5m_return_std",
]

HORIZONS_MINUTES = (1, 5, 15)


def load_and_validate() -> pd.DataFrame:
    """Load the feature universe and validate required fields and timestamps."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Feature universe not found: {INPUT_PATH}\n"
            "Build the feature universe before running this script."
        )

    df = pd.read_parquet(INPUT_PATH)

    required_columns = {
        "minute",
        "close",
        "gap_minutes",
        *FEATURE_COLUMNS,
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Feature universe is missing columns: {sorted(missing)}")

    df = df.copy()
    df["minute"] = pd.to_datetime(df["minute"], utc=True)
    df = df.sort_values("minute", kind="stable").reset_index(drop=True)

    if df["minute"].isna().any():
        raise ValueError("The minute column contains missing timestamps.")
    if df["minute"].duplicated().any():
        raise ValueError("The feature universe contains duplicate minute timestamps.")
    if not df["minute"].is_monotonic_increasing:
        raise ValueError("Minute timestamps are not chronological after sorting.")
    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise ValueError("Close prices must be present and strictly positive.")

    # Recompute gaps from timestamps rather than relying blindly on a stored
    # field. A new segment begins whenever observations are not one minute apart.
    df["_actual_gap_minutes"] = df["minute"].diff().dt.total_seconds() / 60
    stored_gaps = pd.to_numeric(df["gap_minutes"], errors="coerce")
    comparable = df["_actual_gap_minutes"].notna() & stored_gaps.notna()
    if not np.allclose(
        df.loc[comparable, "_actual_gap_minutes"],
        stored_gaps.loc[comparable],
        rtol=0,
        atol=1e-9,
    ):
        raise ValueError("Stored gap_minutes does not match timestamp differences.")

    return df


def add_forward_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add close-to-close forward returns within uninterrupted minute segments."""
    df = df.copy()

    # First row starts a segment. Any gap other than exactly one minute starts
    # another segment, including session breaks and missing-minute intervals.
    df["session_segment_id"] = (
        df["_actual_gap_minutes"].ne(1).cumsum().astype("int64")
    )

    grouped = df.groupby("session_segment_id", sort=False)
    for horizon in HORIZONS_MINUTES:
        future_minute = grouped["minute"].shift(-horizon)
        future_close = grouped["close"].shift(-horizon)

        exact_horizon = (
            future_minute - df["minute"]
        ).eq(pd.Timedelta(minutes=horizon))

        valid_target = exact_horizon & future_close.notna()
        forward_log_return = pd.Series(np.nan, index=df.index, dtype="float64")
        forward_points = pd.Series(np.nan, index=df.index, dtype="float64")

        forward_log_return.loc[valid_target] = np.log(
            future_close.loc[valid_target] / df.loc[valid_target, "close"]
        )
        forward_points.loc[valid_target] = (
            future_close.loc[valid_target] - df.loc[valid_target, "close"]
        )

        df[f"forward_close_{horizon}m_timestamp"] = future_minute.where(valid_target)
        df[f"forward_log_return_{horizon}m"] = forward_log_return
        df[f"forward_price_change_{horizon}m_points"] = forward_points

    return df


def build_dataset() -> pd.DataFrame:
    """Construct and save the eligible-minute research dataset."""
    df = load_and_validate()
    df = add_forward_targets(df)

    imbalance = pd.to_numeric(df["volume_imbalance"], errors="coerce")
    directional_volume = pd.to_numeric(df["directional_volume"], errors="coerce")

    eligible = (
        imbalance.notna()
        & np.isfinite(imbalance)
        & imbalance.ne(0)
        & directional_volume.notna()
        & directional_volume.gt(0)
    )

    research = df.loc[eligible, [
        "minute",
        "session_segment_id",
        *FEATURE_COLUMNS,
        *[
            column
            for horizon in HORIZONS_MINUTES
            for column in (
                f"forward_close_{horizon}m_timestamp",
                f"forward_log_return_{horizon}m",
                f"forward_price_change_{horizon}m_points",
            )
        ],
    ]].copy()

    research.insert(
        2,
        "orderflow_direction",
        np.sign(research["volume_imbalance"]).astype("int8"),
    )

    # Directionally aligned log return: positive means continuation in the
    # direction of current-minute order flow; negative means reversal.
    for horizon in HORIZONS_MINUTES:
        research[f"aligned_forward_log_return_{horizon}m"] = (
            research["orderflow_direction"]
            * research[f"forward_log_return_{horizon}m"]
        )

    if research["minute"].duplicated().any():
        raise RuntimeError("Duplicate minutes found in the research dataset.")
    if not research["orderflow_direction"].isin([-1, 1]).all():
        raise RuntimeError("Eligible observations must have direction -1 or +1.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    research.to_parquet(OUTPUT_PATH, index=False)

    print("\n========== ORDER-FLOW CONFIRMATION DATASET ==========")
    print(f"Input observations: {len(df):,}")
    print(f"Eligible observations: {len(research):,}")
    print(f"Excluded observations: {len(df) - len(research):,}")
    print(f"Distinct contiguous segments: {research['session_segment_id'].nunique():,}")
    print(f"Output shape: {research.shape}")

    print("\nForward-target availability:")
    for horizon in HORIZONS_MINUTES:
        target = f"forward_log_return_{horizon}m"
        available = int(research[target].notna().sum())
        print(
            f"  {horizon:>2}-minute: {available:,} / {len(research):,} "
            f"({available / len(research) * 100:.2f}%)"
        )

    print("\nDirection counts:")
    print(research["orderflow_direction"].value_counts().sort_index().to_string())
    print(f"\nSaved: {OUTPUT_PATH}")

    return research


if __name__ == "__main__":
    build_dataset()

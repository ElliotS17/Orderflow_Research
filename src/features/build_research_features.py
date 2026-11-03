from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/processed")

INPUTS = {
    "deduplicated": DATA_DIR / "features_1min_deduplicated.parquet",
    "original": DATA_DIR / "features_1min_original.parquet",
}

OUTPUTS = {
    "deduplicated": DATA_DIR / "research_features_deduplicated.parquet",
    "original": DATA_DIR / "research_features_original.parquet",
}


def build_research_features(path):
    df = pd.read_parquet(path)
    df = df.sort_values("minute").copy()
    df = df.set_index("minute")

    # --------------------------------------------------
    # Current-minute orderflow strength
    # --------------------------------------------------

    df["absolute_net_volume"] = df["net_volume"].abs()
    df["absolute_imbalance"] = df["volume_imbalance"].abs()

    df["orderflow_direction"] = np.sign(df["net_volume"])

    # --------------------------------------------------
    # Contemporaneous price response
    # --------------------------------------------------

    # Positive = price moved with directional orderflow.
    # Negative = price moved against directional orderflow.
    # Zero = no price change.
    # Missing = no valid consecutive-minute return or no
    # directional orderflow.
    df["signed_price_response"] = (
        df["orderflow_direction"] * df["close_change_points"]
    ).where(
        df["close_change_points"].notna()
        & df["net_volume"].ne(0)
    )

    # Descriptive normalization. Interpret cautiously when
    # absolute net volume is small.
    df["response_per_100_net_contracts"] = (
        df["signed_price_response"]
        / df["absolute_net_volume"].replace(0, np.nan)
        * 100
    )

    # --------------------------------------------------
    # Past-only rolling orderflow context
    # --------------------------------------------------

    for window in ["5min", "15min", "30min"]:
        rolling_net = df["net_volume"].rolling(
            window=window,
            closed="left",
            min_periods=1,
        ).sum()

        rolling_directional = df["directional_volume"].rolling(
            window=window,
            closed="left",
            min_periods=1,
        ).sum()

        df[f"rolling_{window}_net_volume"] = rolling_net

        df[f"rolling_{window}_imbalance"] = (
            rolling_net
            / rolling_directional.replace(0, np.nan)
        )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    for column in [
        "volume_imbalance",
        "rolling_5min_imbalance",
        "rolling_15min_imbalance",
        "rolling_30min_imbalance",
    ]:
        valid = df[column].dropna()

        if not valid.between(-1, 1).all():
            raise RuntimeError(
                f"{column} contains values outside [-1, 1]."
            )

    # A positive signed response must represent movement
    # in the direction of net orderflow.
    valid_response = df["signed_price_response"].notna()

    expected = (
        df.loc[valid_response, "orderflow_direction"]
        * df.loc[valid_response, "close_change_points"]
    )

    if not np.allclose(
        df.loc[valid_response, "signed_price_response"],
        expected,
    ):
        raise RuntimeError("Signed price response validation failed.")

    df = df.reset_index()

    return df


for version, input_path in INPUTS.items():
    print(f"\nBuilding {version} research features...")

    result = build_research_features(input_path)

    output_path = OUTPUTS[version]
    result.to_parquet(output_path, index=False)

    print(f"Minute bars: {len(result):,}")
    print(
        "Valid signed price responses:",
        f"{result['signed_price_response'].notna().sum():,}"
    )

    print("\nFeature summary:")
    print(
        result[
            [
                "absolute_net_volume",
                "absolute_imbalance",
                "signed_price_response",
                "rolling_5min_imbalance",
                "rolling_15min_imbalance",
                "rolling_30min_imbalance",
            ]
        ].describe().round(4).to_string()
    )

    print(f"\nSaved: {output_path}")

print("\nResearch feature construction complete.")
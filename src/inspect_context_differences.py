from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "event_dataset_with_context.parquet"

df = pd.read_parquet(DATA_PATH)
df = df.loc[df["strong_orderflow"]].copy()

df["group"] = df["target_event"].map({
    True: "event",
    False: "control",
})

print("=" * 65)
print("EVENT VS CONTROL: CONTEXT FEATURES")
print("=" * 65)
print(f"Events: {(df['group'] == 'event').sum():,}")
print(f"Controls: {(df['group'] == 'control').sum():,}")

# Compare continuous features.
features = [
    "momentum_5m",
    "momentum_15m",
    "flow_aligned_momentum_5m",
    "flow_aligned_momentum_15m",
    "realized_volatility_15m_bps",
]

for feature in features:
    print(f"\n{feature}")

    group_means = {}

    for group in ["event", "control"]:
        values = df.loc[df["group"] == group, feature].dropna()
        group_means[group] = values.mean()

        print(
            f"  {group:>7}: n={len(values):>4}, "
            f"mean={values.mean():>10.4f}, "
            f"median={values.median():>10.4f}"
        )

    print(
        f"  Mean difference (event - control): "
        f"{group_means['event'] - group_means['control']:+.4f}"
    )

# Compare US cash hours.
print("\n" + "=" * 65)
print("US CASH HOURS")
print("=" * 65)

for group in ["event", "control"]:
    group_data = df.loc[df["group"] == group, "is_us_cash_hours"]

    print(
        f"  {group:>7}: {group_data.mean():.2%} "
        f"({group_data.sum():,} / {len(group_data):,})"
    )

event_rate = df.loc[df["is_us_cash_hours"], "target_event"].mean()
outside_rate = df.loc[~df["is_us_cash_hours"], "target_event"].mean()

print(f"\nEvent rate during US cash hours: {event_rate:.2%}")
print(f"Event rate outside US cash hours: {outside_rate:.2%}")

# Check correlation between momentum horizons.
print("\n" + "=" * 65)
print("MOMENTUM CORRELATION")
print("=" * 65)

correlation_features = [
    "momentum_5m",
    "momentum_15m",
    "flow_aligned_momentum_5m",
    "flow_aligned_momentum_15m",
]

print(df[correlation_features].corr().round(3).to_string())

print("\nContext comparison complete.")
print("These are descriptive comparisons, not significance tests.")
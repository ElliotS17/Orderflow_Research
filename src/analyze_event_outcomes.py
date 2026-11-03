from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "event_dataset_exploratory.parquet"
)

HORIZONS = [1, 5, 15]

df = pd.read_parquet(INPUT_PATH)

required_columns = [
    "minute",
    "strong_orderflow",
    "target_event",
    "orderflow_direction",
]

for horizon in HORIZONS:
    required_columns.append(f"flow_aligned_return_{horizon}m")

missing = [column for column in required_columns if column not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["minute"] = pd.to_datetime(df["minute"], utc=True)
df = df.sort_values("minute").reset_index(drop=True)

# The control group consists only of strong-flow minutes that are
# not classified as weak-response events.
df["research_group"] = "other_minutes"
df.loc[df["strong_orderflow"], "research_group"] = "strong_flow_control"
df.loc[df["target_event"], "research_group"] = "weak_response_event"

event_count = int(df["target_event"].sum())
control_count = int(
    (df["strong_orderflow"] & ~df["target_event"]).sum()
)

print("=" * 72)
print("RESEARCH GROUPS")
print("=" * 72)
print(f"Weak-response events: {event_count:,}")
print(f"Strong-flow controls: {control_count:,}")
print(
    "Other minutes (not used in primary comparison): "
    f"{(~df['strong_orderflow']).sum():,}"
)

# Summarize directional outcomes for each group and horizon.
for horizon in HORIZONS:
    column = f"flow_aligned_return_{horizon}m"

    print("\n" + "=" * 72)
    print(f"{horizon}-MINUTE FORWARD OUTCOMES")
    print("=" * 72)

    for group in ["weak_response_event", "strong_flow_control"]:
        values = df.loc[df["research_group"] == group, column].dropna()

        if values.empty:
            print(f"\n{group}: no valid observations")
            continue

        reversal_rate = (values < 0).mean()
        continuation_rate = (values > 0).mean()
        flat_rate = (values == 0).mean()

        print(f"\n{group.replace('_', ' ').title()}")
        print(f"  Valid observations: {len(values):,}")
        print(f"  Reversal rate:      {reversal_rate:.2%}")
        print(f"  Continuation rate:  {continuation_rate:.2%}")
        print(f"  Flat rate:          {flat_rate:.2%}")
        print(f"  Mean aligned return:   {values.mean():.4f} points")
        print(f"  Median aligned return: {values.median():.4f} points")
        print(f"  Std. dev.:             {values.std():.4f} points")

    event_values = df.loc[
        df["research_group"] == "weak_response_event", column
    ].dropna()

    control_values = df.loc[
        df["research_group"] == "strong_flow_control", column
    ].dropna()

    if not event_values.empty and not control_values.empty:
        event_reversal = (event_values < 0).mean()
        control_reversal = (control_values < 0).mean()

        print("\nEvent minus control:")
        print(
            "  Difference in reversal rates: "
            f"{event_reversal - control_reversal:+.2%}"
        )
        print(
            "  Difference in mean aligned return: "
            f"{event_values.mean() - control_values.mean():+.4f} points"
        )

# Check whether events are concentrated on a small number of dates.
df["trading_date_utc"] = df["minute"].dt.date

daily = (
    df.loc[df["target_event"]]
    .groupby("trading_date_utc")
    .size()
    .rename("event_count")
)

print("\n" + "=" * 72)
print("EVENTS BY UTC DATE")
print("=" * 72)
print(daily.to_string())
print(f"\nDates with at least one event: {len(daily)}")
print(f"Maximum events on one date: {daily.max() if not daily.empty else 0}")

print("\nDescriptive analysis complete.")
print("No statistical significance tests have been performed.")
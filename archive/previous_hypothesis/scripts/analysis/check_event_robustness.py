from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "event_dataset_exploratory.parquet"

HORIZONS = [1, 5, 15]
MIN_EVENT_SPACING = 15

df = pd.read_parquet(DATA_PATH)
df["minute"] = pd.to_datetime(df["minute"], utc=True)
df = df.sort_values("minute").reset_index(drop=True)

# Label each observation by its CME equity-index-futures session.
local_time = df["minute"].dt.tz_convert("America/Chicago")
df["session"] = (
    local_time.dt.normalize()
    + pd.to_timedelta((local_time.dt.hour >= 17).astype(int), unit="D")
).dt.date

df["group"] = np.where(
    df["target_event"],
    "event",
    np.where(df["strong_orderflow"], "control", "other"),
)


def calculate_differences(data, horizon):
    column = f"flow_aligned_return_{horizon}m"
    event = data.loc[data["group"] == "event", column].dropna()
    control = data.loc[data["group"] == "control", column].dropna()

    if event.empty or control.empty:
        return None

    return {
        "event_n": len(event),
        "control_n": len(control),
        "mean_difference": event.mean() - control.mean(),
        "reversal_difference": (event < 0).mean() - (control < 0).mean(),
    }


# ------------------------------------------------------------
# 1. Leave-one-session-out analysis
# ------------------------------------------------------------

print("=" * 70)
print("LEAVE-ONE-SESSION-OUT ANALYSIS")
print("=" * 70)

sessions = sorted(df.loc[df["group"].isin(["event", "control"]), "session"].unique())
print(f"Sessions included: {len(sessions)}")

for horizon in HORIZONS:
    full_result = calculate_differences(df, horizon)
    results = []

    for session in sessions:
        reduced = df.loc[df["session"] != session]
        result = calculate_differences(reduced, horizon)

        if result is not None:
            result["omitted_session"] = session
            results.append(result)

    results_df = pd.DataFrame(results)

    print(f"\n{horizon}-minute horizon")
    print(
        f"Full-sample mean difference: "
        f"{full_result['mean_difference']:+.4f} points"
    )
    print(
        f"Leave-one-out range: "
        f"{results_df['mean_difference'].min():+.4f} to "
        f"{results_df['mean_difference'].max():+.4f} points"
    )
    print(
        f"Sessions where mean difference stayed positive: "
        f"{(results_df['mean_difference'] > 0).sum()} / {len(results_df)}"
    )

    print(
        f"Full-sample reversal-rate difference: "
        f"{full_result['reversal_difference']:+.2%}"
    )
    print(
        f"Leave-one-out reversal-rate range: "
        f"{results_df['reversal_difference'].min():+.2%} to "
        f"{results_df['reversal_difference'].max():+.2%}"
    )

    # Show which omitted session changes the mean difference most.
    results_df["change_from_full"] = (
        results_df["mean_difference"] - full_result["mean_difference"]
    )
    most_influential = results_df.iloc[
        results_df["change_from_full"].abs().argmax()
    ]

    print(
        f"Most influential omission: {most_influential['omitted_session']} "
        f"(mean difference {most_influential['mean_difference']:+.4f})"
    )


# ------------------------------------------------------------
# 2. Non-overlapping strong-orderflow observations
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("NON-OVERLAPPING OBSERVATION CHECK")
print("=" * 70)

# Use only strong-flow observations with all three forward outcomes.
# This ensures the same selected sample can be compared at every horizon.
outcome_columns = [
    f"flow_aligned_return_{horizon}m" for horizon in HORIZONS
]

candidates = df.loc[
    df["group"].isin(["event", "control"])
    & df[outcome_columns].notna().all(axis=1)
].copy()


def select_spaced_observations(data, reverse=False):
    data = data.sort_values("minute", ascending=not reverse)

    selected = []
    last_selected_time = None
    spacing = pd.Timedelta(minutes=MIN_EVENT_SPACING)

    for row in data.itertuples():
        current_time = row.minute

        if last_selected_time is None:
            selected.append(row.Index)
            last_selected_time = current_time
        elif not reverse and current_time - last_selected_time >= spacing:
            selected.append(row.Index)
            last_selected_time = current_time
        elif reverse and last_selected_time - current_time >= spacing:
            selected.append(row.Index)
            last_selected_time = current_time

    return data.loc[selected].copy()


for label, reverse in [
    ("Chronological selection", False),
    ("Reverse-chronological selection", True),
]:
    selected = select_spaced_observations(candidates, reverse=reverse)

    print(f"\n{label}")
    print(f"Selected observations: {len(selected):,}")
    print(f"  Events:   {(selected['group'] == 'event').sum():,}")
    print(f"  Controls: {(selected['group'] == 'control').sum():,}")

    # Verify the spacing rule.
    gaps = selected["minute"].sort_values().diff().dropna()
    if not gaps.empty and (gaps < pd.Timedelta(minutes=MIN_EVENT_SPACING)).any():
        raise ValueError("Non-overlapping selection contains observations too close together.")

    for horizon in HORIZONS:
        result = calculate_differences(selected, horizon)

        if result is None:
            print(f"  {horizon}m: insufficient observations in one group")
            continue

        print(
            f"  {horizon:>2}m: "
            f"mean difference {result['mean_difference']:+.4f} points; "
            f"reversal-rate difference {result['reversal_difference']:+.2%}"
        )

print("\nRobustness checks complete.")
print("These are sensitivity analyses, not new significance tests.")
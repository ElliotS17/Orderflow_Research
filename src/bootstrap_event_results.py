from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "event_dataset_exploratory.parquet"

HORIZONS = [1, 5, 15]
BOOTSTRAP_REPS = 10_000
RNG = np.random.default_rng(42)

df = pd.read_parquet(DATA_PATH)
df["minute"] = pd.to_datetime(df["minute"], utc=True)

# CME equity index futures sessions begin at 17:00 Chicago time.
# Assign evening observations to the following session date.
local_time = df["minute"].dt.tz_convert("America/Chicago")
session_start_date = (
    local_time.dt.normalize()
    + pd.to_timedelta((local_time.dt.hour >= 17).astype(int), unit="D")
)
df["session"] = session_start_date.dt.date

df["research_group"] = np.where(
    df["target_event"],
    "event",
    np.where(df["strong_orderflow"], "control", "other"),
)

sessions = sorted(df.loc[df["research_group"].isin(["event", "control"]), "session"].unique())

print(f"Trading sessions represented: {len(sessions)}")
print(f"Bootstrap repetitions: {BOOTSTRAP_REPS:,}")

for horizon in HORIZONS:
    column = f"flow_aligned_return_{horizon}m"

    sample = df.loc[
        df["research_group"].isin(["event", "control"])
        & df[column].notna(),
        ["session", "research_group", column],
    ]

    # Store event and control outcomes separately for each session.
    session_values = {}
    for session in sessions:
        day = sample[sample["session"] == session]

        session_values[session] = {
            "event": day.loc[
                day["research_group"] == "event", column
            ].to_numpy(),
            "control": day.loc[
                day["research_group"] == "control", column
            ].to_numpy(),
        }

    event_values = sample.loc[
        sample["research_group"] == "event", column
    ].to_numpy()
    control_values = sample.loc[
        sample["research_group"] == "control", column
    ].to_numpy()

    observed_mean_difference = event_values.mean() - control_values.mean()
    observed_reversal_difference = (
        (event_values < 0).mean() - (control_values < 0).mean()
    )

    mean_differences = []
    reversal_differences = []

    for _ in range(BOOTSTRAP_REPS):
        selected_sessions = RNG.choice(
            sessions, size=len(sessions), replace=True
        )

        boot_events = np.concatenate([
            session_values[session]["event"]
            for session in selected_sessions
        ])
        boot_controls = np.concatenate([
            session_values[session]["control"]
            for session in selected_sessions
        ])

        if len(boot_events) == 0 or len(boot_controls) == 0:
            continue

        mean_differences.append(
            boot_events.mean() - boot_controls.mean()
        )
        reversal_differences.append(
            (boot_events < 0).mean() - (boot_controls < 0).mean()
        )

    mean_ci = np.quantile(mean_differences, [0.025, 0.975])
    reversal_ci = np.quantile(reversal_differences, [0.025, 0.975])

    print("\n" + "=" * 65)
    print(f"{horizon}-MINUTE RESULTS")
    print("=" * 65)
    print(f"Event observations: {len(event_values):,}")
    print(f"Control observations: {len(control_values):,}")

    print("\nEvent minus control mean aligned return:")
    print(f"  Observed difference: {observed_mean_difference:+.4f} points")
    print(f"  95% session-bootstrap CI: [{mean_ci[0]:+.4f}, {mean_ci[1]:+.4f}]")

    print("\nEvent minus control reversal rate:")
    print(f"  Observed difference: {observed_reversal_difference:+.2%}")
    print(f"  95% session-bootstrap CI: [{reversal_ci[0]:+.2%}, {reversal_ci[1]:+.2%}]")

print("\nBootstrap analysis complete.")
print("Intervals describe session-resampling uncertainty, not causality.")
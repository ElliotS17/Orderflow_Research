from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "research_orderflow_confirmation.parquet"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "orderflow_confirmation"
)

TABLE_DIR = REPORT_DIR / "tables"
PREDICTION_DIR = REPORT_DIR / "model_comparison"

TARGET = "aligned_forward_log_return_5m"

INITIAL_TRAIN_SESSIONS = 10
RIDGE_ALPHA = 1.0

# Model 1: Basic directional order-flow information.
BASELINE_FEATURES = [
    "orderflow_direction",
    "abs_volume_imbalance",
    "directional_volume",
]

# Model 2: Baseline plus recent market conditions.
CONTEXT_FEATURES = [
    "aligned_prior_5m_volume_imbalance",
    "aligned_prior_5m_return",
    "relative_volume_5m",
    "prior_5m_return_std",
]

# Model 3: Context model plus the seven selected
# order-flow confirmation features.
CONFIRMATION_FEATURES = [
    "trade_size_hhi",
    "max_10s_volume_share",
    "aligned_late_early_pressure_diff",
    "directional_persistence",
    "price_path_efficiency",
    "aligned_close_vwap_displacement",
    "aligned_buy_sell_vwap_diff",
]

MODELS = {
    "baseline": BASELINE_FEATURES,
    "context": BASELINE_FEATURES + CONTEXT_FEATURES,
    "confirmation": (
        BASELINE_FEATURES
        + CONTEXT_FEATURES
        + CONFIRMATION_FEATURES
    ),
}


# ============================================================
# Data preparation
# ============================================================

def load_data():
    df = pd.read_parquet(DATA_PATH)

    required = [
        "minute",
        "session_segment_id",
        "orderflow_direction",
        "volume_imbalance",
        "directional_volume",
        "prior_5m_volume_imbalance",
        "prior_5m_return",
        "relative_volume_5m",
        "prior_5m_return_std",
        "late_early_pressure_diff",
        "close_vwap_displacement",
        "buy_sell_vwap_diff",
        TARGET,
    ] + [
        "trade_size_hhi",
        "max_10s_volume_share",
        "directional_persistence",
        "price_path_efficiency",
    ]

    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["minute"] = pd.to_datetime(df["minute"])

    # Align signed historical context with current order-flow direction.
    df["aligned_prior_5m_volume_imbalance"] = (
        df["orderflow_direction"]
        * df["prior_5m_volume_imbalance"]
    )

    df["aligned_prior_5m_return"] = (
        df["orderflow_direction"]
        * df["prior_5m_return"]
    )

    # Align signed intraminute confirmation features.
    df["aligned_late_early_pressure_diff"] = (
        df["orderflow_direction"]
        * df["late_early_pressure_diff"]
    )

    df["aligned_close_vwap_displacement"] = (
        df["orderflow_direction"]
        * df["close_vwap_displacement"]
    )

    df["aligned_buy_sell_vwap_diff"] = (
        df["orderflow_direction"]
        * df["buy_sell_vwap_diff"]
    )

    df["abs_volume_imbalance"] = df["volume_imbalance"].abs()

    # Only rows with an observed primary outcome are evaluated.
    df = df.dropna(subset=[TARGET]).copy()

    # Replace infinities with missing values. Imputation is fitted
    # separately inside each training fold.
    numeric_features = sorted({
        feature
        for features in MODELS.values()
        for feature in features
    })

    df[numeric_features] = df[numeric_features].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df.sort_values("minute").reset_index(drop=True)


# ============================================================
# Model
# ============================================================

def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=RIDGE_ALPHA)),
    ])


# ============================================================
# Evaluation
# ============================================================

def calculate_metrics(actual, predicted, baseline_mean):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    residual_sum_squares = np.sum((actual - predicted) ** 2)

    reference_sum_squares = np.sum(
        (actual - baseline_mean) ** 2
    )

    oos_r2 = (
        1 - residual_sum_squares / reference_sum_squares
        if reference_sum_squares > 0
        else np.nan
    )

    # Exclude exactly zero realized returns from directional accuracy.
    nonzero = actual != 0

    directional_accuracy = (
        np.mean(np.sign(actual[nonzero]) == np.sign(predicted[nonzero]))
        if nonzero.any()
        else np.nan
    )

    return {
        "n": len(actual),
        "mae_bps": np.mean(np.abs(actual - predicted)) * 10000,
        "oos_r2": oos_r2,
        "directional_accuracy_pct": directional_accuracy * 100,
        "actual_mean_bps": np.mean(actual) * 10000,
        "predicted_mean_bps": np.mean(predicted) * 10000,
    }


def main():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading research dataset...")
    df = load_data()

    sessions = sorted(df["session_segment_id"].unique())

    print(f"Observations with valid target: {len(df):,}")
    print(f"Available sessions: {len(sessions)}")

    if len(sessions) <= INITIAL_TRAIN_SESSIONS:
        raise ValueError("Not enough sessions for the planned validation.")

    print(
        f"Initial training sessions: {INITIAL_TRAIN_SESSIONS}"
    )
    print(
        f"Out-of-sample test sessions: "
        f"{len(sessions) - INITIAL_TRAIN_SESSIONS}"
    )

    session_results = []
    prediction_results = []

    # Expanding-window validation:
    # Train on earlier sessions, test on the next complete session.
    for test_index in range(
        INITIAL_TRAIN_SESSIONS,
        len(sessions),
    ):
        train_sessions = sessions[:test_index]
        test_session = sessions[test_index]

        train = df[
            df["session_segment_id"].isin(train_sessions)
        ].copy()

        test = df[
            df["session_segment_id"] == test_session
        ].copy()

        print(
            f"\nTesting session {test_session}: "
            f"train n={len(train):,}, test n={len(test):,}"
        )

        actual = test[TARGET].to_numpy()
        training_mean = train[TARGET].mean()

        for model_name, features in MODELS.items():
            model = create_model()

            model.fit(
                train[features],
                train[TARGET],
            )

            predicted = model.predict(test[features])

            metrics = calculate_metrics(
                actual,
                predicted,
                training_mean,
            )

            metrics.update({
                "model": model_name,
                "test_session": test_session,
                "train_sessions": len(train_sessions),
            })

            session_results.append(metrics)

            prediction_frame = pd.DataFrame({
                "minute": test["minute"].to_numpy(),
                "session_segment_id": test_session,
                "actual": actual,
                "predicted": predicted,
                "model": model_name,
            })

            prediction_results.append(prediction_frame)

            print(
                f"  {model_name:15s} "
                f"MAE={metrics['mae_bps']:.4f} bps | "
                f"OOS R2={metrics['oos_r2']:.4f} | "
                f"Accuracy={metrics['directional_accuracy_pct']:.2f}%"
            )

    # --------------------------------------------------------
    # Save individual-session results.
    # --------------------------------------------------------

    session_results = pd.DataFrame(session_results)

    session_results.to_csv(
        TABLE_DIR / "session_metrics.csv",
        index=False,
    )

    predictions = pd.concat(
        prediction_results,
        ignore_index=True,
    )

    predictions.to_parquet(
        PREDICTION_DIR / "out_of_sample_predictions.parquet",
        index=False,
    )

    # --------------------------------------------------------
    # Calculate pooled out-of-sample metrics.
    # Each test session is evaluated against its own
    # training-period mean benchmark.
    # --------------------------------------------------------

    pooled_results = []

    for model_name in MODELS:
        model_predictions = predictions[
            predictions["model"] == model_name
        ].copy()

        model_sessions = session_results[
            session_results["model"] == model_name
        ]

        actual = model_predictions["actual"].to_numpy()
        predicted = model_predictions["predicted"].to_numpy()

        # Reconstruct the expanding-window training mean for each
        # session from the same historical observations.
        reference_sum_squares = 0.0

        for _, row in model_sessions.iterrows():
            test_session = row["test_session"]
            test_mask = (
                model_predictions["session_segment_id"]
                == test_session
            )

            session_actual = model_predictions.loc[
                test_mask, "actual"
            ].to_numpy()

            training_sessions = [
                s for s in sessions
                if s < test_session
            ]

            training_mean = df[
                df["session_segment_id"].isin(training_sessions)
            ][TARGET].mean()

            reference_sum_squares += np.sum(
                (session_actual - training_mean) ** 2
            )

        residual_sum_squares = np.sum(
            (actual - predicted) ** 2
        )

        pooled_r2 = (
            1 - residual_sum_squares / reference_sum_squares
            if reference_sum_squares > 0
            else np.nan
        )

        nonzero = actual != 0

        accuracy = (
            np.mean(
                np.sign(actual[nonzero])
                == np.sign(predicted[nonzero])
            ) * 100
            if nonzero.any()
            else np.nan
        )

        pooled_results.append({
            "model": model_name,
            "n": len(actual),
            "mae_bps": np.mean(np.abs(actual - predicted)) * 10000,
            "oos_r2": pooled_r2,
            "directional_accuracy_pct": accuracy,
            "actual_mean_bps": np.mean(actual) * 10000,
            "predicted_mean_bps": np.mean(predicted) * 10000,
        })

    pooled_results = pd.DataFrame(pooled_results)

    pooled_results.to_csv(
        TABLE_DIR / "pooled_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save concise model comparison.
    # --------------------------------------------------------

    comparison = pooled_results.copy()

    context_mae = comparison.loc[
        comparison["model"] == "context",
        "mae_bps",
    ].iloc[0]

    confirmation_mae = comparison.loc[
        comparison["model"] == "confirmation",
        "mae_bps",
    ].iloc[0]

    comparison["mae_improvement_vs_context_pct"] = (
        (context_mae - comparison["mae_bps"])
        / context_mae
        * 100
    )

    comparison["oos_r2_improvement_vs_context"] = (
        comparison["oos_r2"]
        - comparison.loc[
            comparison["model"] == "context",
            "oos_r2",
        ].iloc[0]
    )

    comparison.to_csv(
        TABLE_DIR / "model_comparison.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print("\n========== POOLED OUT-OF-SAMPLE RESULTS ==========")
    print(pooled_results.round(4).to_string(index=False))

    print("\n========== INCREMENTAL CONFIRMATION VALUE ==========")
    print(
        comparison[
            [
                "model",
                "mae_improvement_vs_context_pct",
                "oos_r2_improvement_vs_context",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n========== SESSION-LEVEL RESULTS ==========")
    print(
        session_results[
            [
                "test_session",
                "model",
                "mae_bps",
                "oos_r2",
                "directional_accuracy_pct",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved model metric tables to:")
    print(TABLE_DIR.relative_to(PROJECT_ROOT))
    print("Saved out-of-sample predictions to:")
    print(PREDICTION_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

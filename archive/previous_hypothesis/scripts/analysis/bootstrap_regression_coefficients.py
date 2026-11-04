from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "event_dataset_with_context.parquet"
OUTPUT_PATH = ROOT / "results" / "session_bootstrap_regression_coefficients.csv"

N_BOOTSTRAPS = 10_000
RANDOM_SEED = 42

PREDICTORS = [
    "event",
    "flow_aligned_momentum_5m",
    "flow_aligned_momentum_15m",
    "realized_volatility_15m_bps",
    "is_us_cash_hours",
]

OUTCOMES = [
    "flow_aligned_return_1m",
    "flow_aligned_return_5m",
    "flow_aligned_return_15m",
]


def main():
    df = pd.read_parquet(DATA_PATH)
    df["minute"] = pd.to_datetime(df["minute"], utc=True)

    df = df.loc[df["strong_orderflow"]].copy()
    df["event"] = df["target_event"].astype(int)
    df["is_us_cash_hours"] = df["is_us_cash_hours"].astype(int)

    chicago_time = df["minute"].dt.tz_convert("America/Chicago")
    df["session_id"] = (
        chicago_time - pd.Timedelta(hours=17)
    ).dt.strftime("%Y-%m-%d")

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    for outcome in OUTCOMES:
        columns = PREDICTORS + [outcome, "session_id"]
        model_data = df[columns].dropna().copy()

        X = sm.add_constant(
            model_data[PREDICTORS],
            has_constant="add",
        )
        y = model_data[outcome]

        # Recalculate the original session-clustered regression.
        clustered_model = sm.OLS(y, X)
        clustered_result = clustered_model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": model_data["session_id"],
                "use_correction": True,
                "df_correction": True,
            },
            use_t=True,
        )

        observed_coefficient = clustered_result.params["event"]
        clustered_ci = clustered_result.conf_int().loc["event"]

        # Store each session's design matrix and outcome separately.
        session_data = {}

        for session_id, group in model_data.groupby("session_id"):
            session_X = sm.add_constant(
                group[PREDICTORS],
                has_constant="add",
            ).to_numpy()

            session_y = group[outcome].to_numpy()

            session_data[session_id] = (session_X, session_y)

        session_ids = list(session_data.keys())
        n_sessions = len(session_ids)

        bootstrap_coefficients = []
        rank_deficient_samples = 0

        for _ in range(N_BOOTSTRAPS):
            # Sample entire sessions, retaining duplicate draws.
            sampled_sessions = rng.choice(
                session_ids,
                size=n_sessions,
                replace=True,
            )

            X_parts = []
            y_parts = []

            for session_id in sampled_sessions:
                session_X, session_y = session_data[session_id]
                X_parts.append(session_X)
                y_parts.append(session_y)

            X_boot = np.vstack(X_parts)
            y_boot = np.concatenate(y_parts)

            # Refit OLS on the resampled sessions.
            coefficients, _, rank, _ = np.linalg.lstsq(
                X_boot,
                y_boot,
                rcond=None,
            )

            if rank < X_boot.shape[1]:
                rank_deficient_samples += 1

            # Event is the first predictor after the intercept.
            bootstrap_coefficients.append(coefficients[1])

        bootstrap_coefficients = np.asarray(bootstrap_coefficients)

        # Marginal 95% percentile confidence interval.
        ci_95 = np.quantile(
            bootstrap_coefficients,
            [0.025, 0.975],
        )

        # Bonferroni-adjusted interval across three horizons.
        adjusted_alpha = 0.05 / len(OUTCOMES)
        ci_adjusted = np.quantile(
            bootstrap_coefficients,
            [adjusted_alpha / 2, 1 - adjusted_alpha / 2],
        )

        results.append({
            "outcome": outcome,
            "observations": len(model_data),
            "trading_sessions": n_sessions,
            "observed_event_coefficient": observed_coefficient,
            "clustered_standard_error": clustered_result.bse["event"],
            "clustered_p_value": clustered_result.pvalues["event"],
            "clustered_ci_lower": clustered_ci.iloc[0],
            "clustered_ci_upper": clustered_ci.iloc[1],
            "bootstrap_standard_error": bootstrap_coefficients.std(ddof=1),
            "bootstrap_ci_95_lower": ci_95[0],
            "bootstrap_ci_95_upper": ci_95[1],
            "bootstrap_ci_adjusted_lower": ci_adjusted[0],
            "bootstrap_ci_adjusted_upper": ci_adjusted[1],
            "bootstrap_share_positive": np.mean(
                bootstrap_coefficients > 0
            ),
            "bootstrap_repetitions": N_BOOTSTRAPS,
            "rank_deficient_samples": rank_deficient_samples,
        })

        print("\n" + "=" * 65)
        print(outcome)
        print("=" * 65)
        print(f"Observations: {len(model_data)}")
        print(f"Trading sessions: {n_sessions}")
        print(f"Observed event coefficient: {observed_coefficient:+.4f}")
        print(
            f"Clustered 95% CI: "
            f"[{clustered_ci.iloc[0]:+.4f}, {clustered_ci.iloc[1]:+.4f}]"
        )
        print(
            f"Bootstrap 95% CI: "
            f"[{ci_95[0]:+.4f}, {ci_95[1]:+.4f}]"
        )
        print(
            f"Bonferroni-adjusted CI: "
            f"[{ci_adjusted[0]:+.4f}, {ci_adjusted[1]:+.4f}]"
        )
        print(
            f"Bootstrap standard error: "
            f"{bootstrap_coefficients.std(ddof=1):.4f}"
        )
        print(
            f"Bootstrap estimates above zero: "
            f"{np.mean(bootstrap_coefficients > 0):.2%}"
        )
        print(
            f"Rank-deficient samples: {rank_deficient_samples}"
        )

    results_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 65)
    print("BOOTSTRAP REGRESSION SUMMARY")
    print("=" * 65)
    print(results_df.round(4).to_string(index=False))
    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

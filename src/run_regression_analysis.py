from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "event_dataset_with_context.parquet"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA_PATH)
df["minute"] = pd.to_datetime(df["minute"], utc=True)

# Keep strong-orderflow observations: weak-response events and controls.
df = df.loc[df["strong_orderflow"]].copy()

df["event"] = df["target_event"].astype(int)
df["is_us_cash_hours"] = df["is_us_cash_hours"].astype(int)

# Assign each observation to the CME session beginning at 17:00 Chicago time.
chicago_time = df["minute"].dt.tz_convert("America/Chicago")
df["session_id"] = (
    chicago_time - pd.Timedelta(hours=17)
).dt.strftime("%Y-%m-%d")

predictors = [
    "event",
    "flow_aligned_momentum_5m",
    "flow_aligned_momentum_15m",
    "realized_volatility_15m_bps",
    "is_us_cash_hours",
]

outcomes = [
    "flow_aligned_return_1m",
    "flow_aligned_return_5m",
    "flow_aligned_return_15m",
]

coefficient_results = []
vif_results = []
model_results = []

print("=" * 65)
print("OLS REGRESSION: EVENT AND CONTEXTUAL CONTROLS")
print("=" * 65)

for outcome in outcomes:
    required_columns = predictors + [outcome, "session_id"]
    model_data = df[required_columns].dropna().copy()

    X = sm.add_constant(model_data[predictors], has_constant="add")
    y = model_data[outcome]
    groups = model_data["session_id"]

    # OLS with standard errors clustered by trading session.
    model = sm.OLS(y, X)
    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": groups,
            "use_correction": True,
            "df_correction": True,
        },
        use_t=True,
    )

    n_sessions = groups.nunique()

    print("\n" + "-" * 65)
    print(f"OUTCOME: {outcome}")
    print("-" * 65)
    print(f"Observations: {len(model_data):,}")
    print(f"Trading sessions: {n_sessions}")
    print(f"R-squared: {result.rsquared:.4f}")
    print(f"Adjusted R-squared: {result.rsquared_adj:.4f}")
    print(f"Event coefficient: {result.params['event']:+.4f}")
    print(f"Event standard error: {result.bse['event']:.4f}")
    print(f"Event p-value: {result.pvalues['event']:.4f}")

    # Save coefficient estimates and confidence intervals.
    confidence_intervals = result.conf_int()

    for variable in X.columns:
        lower, upper = confidence_intervals.loc[variable]

        coefficient_results.append({
            "outcome": outcome,
            "variable": variable,
            "coefficient": result.params[variable],
            "standard_error": result.bse[variable],
            "p_value": result.pvalues[variable],
            "ci_lower": lower,
            "ci_upper": upper,
        })

    # Calculate VIF for each explanatory variable.
    for i, variable in enumerate(X.columns):
        if variable == "const":
            continue

        vif = variance_inflation_factor(X.to_numpy(), i)

        vif_results.append({
            "outcome": outcome,
            "variable": variable,
            "vif": vif,
        })

    model_results.append({
        "outcome": outcome,
        "observations": len(model_data),
        "trading_sessions": n_sessions,
        "r_squared": result.rsquared,
        "adjusted_r_squared": result.rsquared_adj,
        "event_coefficient": result.params["event"],
        "event_standard_error": result.bse["event"],
        "event_p_value": result.pvalues["event"],
    })

# Save results separately so they can be inspected later.
coefficients_df = pd.DataFrame(coefficient_results)
vif_df = pd.DataFrame(vif_results)
models_df = pd.DataFrame(model_results)

coefficients_df.to_csv(
    RESULTS_DIR / "ols_session_clustered_coefficients.csv",
    index=False,
)
vif_df.to_csv(
    RESULTS_DIR / "ols_vif_results.csv",
    index=False,
)
models_df.to_csv(
    RESULTS_DIR / "ols_model_summary.csv",
    index=False,
)

print("\n" + "=" * 65)
print("VIF RESULTS")
print("=" * 65)
print(vif_df.round(3).to_string(index=False))

print("\n" + "=" * 65)
print("MODEL SUMMARY")
print("=" * 65)
print(models_df.round(4).to_string(index=False))

print("\nSaved regression results to:")
print(RESULTS_DIR)
print("\nNote: Standard errors are clustered by trading session.")
print("There are relatively few sessions, so inference remains exploratory.")
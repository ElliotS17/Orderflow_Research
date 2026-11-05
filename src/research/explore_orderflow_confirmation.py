from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "orderflow_confirmation"
)
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TARGET = "aligned_forward_log_return_5m"
DIRECTION = "orderflow_direction"

# Original selected confirmation features.
BASE_CONFIRMATION_FEATURES = [
    "trade_size_hhi",
    "max_10s_volume_share",
    "late_early_pressure_diff",
    "directional_persistence",
    "price_path_efficiency",
    "close_vwap_displacement",
    "buy_sell_vwap_diff",
]

# Context and baseline variables used for descriptive correlations.
CONTEXT_FEATURES = [
    "abs_volume_imbalance",
    "directional_volume",
    "prior_5m_volume_imbalance",
    "prior_5m_return",
    "relative_volume_5m",
    "prior_5m_return_std",
]

PRIMARY_COMPARISON_FEATURES = [
    "directional_persistence",
    "price_path_efficiency",
    "aligned_close_vwap_displacement",
]

N_QUANTILES = 5
TAIL_CLIP_LOWER = 0.01
TAIL_CLIP_UPPER = 0.99


# ============================================================
# Helpers
# ============================================================

def save_csv(dataframe, filename):
    path = TABLE_DIR / filename
    dataframe.to_csv(path, index=False)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


def save_figure(filename):
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")

def require_columns(dataframe, columns):
    missing = sorted(set(columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def quantile_labels(series, q=N_QUANTILES):
    """Create quantile bins; tied values may result in fewer than q bins."""
    valid = series.dropna()
    if valid.nunique() < 2:
        return pd.Series(index=series.index, dtype="float64")
    try:
        binned = pd.qcut(
            series,
            q=q,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return pd.Series(index=series.index, dtype="float64")
    return binned.astype("float64") + 1


def descriptive_feature_summary(dataframe, features):
    rows = []
    for feature in features:
        values = dataframe[feature].dropna()
        rows.append({
            "feature": feature,
            "count": int(values.count()),
            "missing": int(dataframe[feature].isna().sum()),
            "missing_pct": dataframe[feature].isna().mean() * 100,
            "unique_values": int(values.nunique()),
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "p01": values.quantile(0.01),
            "p25": values.quantile(0.25),
            "median": values.median(),
            "p75": values.quantile(0.75),
            "p99": values.quantile(0.99),
            "max": values.max(),
            "skew": values.skew(),
        })
    return pd.DataFrame(rows)

def build_direction_diagnostics(dataframe):
    working = dataframe.dropna(
        subset=[
            "volume_imbalance",
            "forward_log_return_5m",
            TARGET,
            DIRECTION,
            "session_segment_id",
        ]
    ).copy()

    # --------------------------------------------------
    # Validate direction and target construction
    # --------------------------------------------------

    expected_direction = np.sign(working["volume_imbalance"])
    direction_mismatches = (
        expected_direction != working[DIRECTION]
    ).sum()

    expected_target = (
        working[DIRECTION] * working["forward_log_return_5m"]
    )

    target_errors = (
        working[TARGET] - expected_target
    ).abs()

    print("\n========== DIRECTION VALIDATION ==========")
    print(f"Direction mismatches: {direction_mismatches:,}")
    print(f"Maximum target construction error: {target_errors.max():.12f}")

    if direction_mismatches > 0:
        raise ValueError("Order-flow direction does not match volume imbalance.")

    if not np.allclose(working[TARGET], expected_target):
        raise ValueError("Aligned target construction validation failed.")

    # --------------------------------------------------
    # Overall buyer vs seller comparison
    # --------------------------------------------------

    working["direction_group"] = working[DIRECTION].map({
        1: "buyer_dominant",
        -1: "seller_dominant",
    })

    overall = (
        working.groupby("direction_group")
        .agg(
            n=(TARGET, "count"),
            raw_return_mean=("forward_log_return_5m", "mean"),
            raw_return_median=("forward_log_return_5m", "median"),
            aligned_return_mean=(TARGET, "mean"),
            aligned_return_median=(TARGET, "median"),
            continuation_rate=(TARGET, lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    for column in [
        "raw_return_mean",
        "raw_return_median",
        "aligned_return_mean",
        "aligned_return_median",
    ]:
        overall[column] *= 10000

    overall["continuation_rate"] *= 100

    print("\n========== OVERALL DIRECTION COMPARISON ==========")
    print(overall.to_string(index=False))

    # --------------------------------------------------
    # Session-by-session comparison
    # --------------------------------------------------

    sessions = (
        working.groupby(
            ["session_segment_id", "direction_group"]
        )
        .agg(
            n=(TARGET, "count"),
            raw_return_mean=("forward_log_return_5m", "mean"),
            aligned_return_mean=(TARGET, "mean"),
            continuation_rate=(TARGET, lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    sessions["raw_return_mean_bps"] = (
        sessions["raw_return_mean"] * 10000
    )
    sessions["aligned_return_mean_bps"] = (
        sessions["aligned_return_mean"] * 10000
    )
    sessions["continuation_rate_pct"] = (
        sessions["continuation_rate"] * 100
    )

    sessions = sessions.drop(columns=[
        "raw_return_mean",
        "aligned_return_mean",
        "continuation_rate",
    ])

    print("\n========== SESSION-LEVEL DIRECTION COMPARISON ==========")
    print(sessions.to_string(index=False))

    return overall, sessions


def build_quantile_analysis(dataframe, features):
    rows = []

    for feature in features:
        working = dataframe[
            [feature, TARGET, DIRECTION]
        ].dropna(subset=[feature, TARGET]).copy()

        working["feature_quantile"] = quantile_labels(working[feature])

        for quantile, group in working.dropna(
            subset=["feature_quantile"]
        ).groupby("feature_quantile", sort=True):
            rows.append({
                "feature": feature,
                "direction_group": "all",
                "quantile": int(quantile),
                "n": len(group),
                "feature_mean": group[feature].mean(),
                "aligned_return_mean": group[TARGET].mean(),
                "aligned_return_median": group[TARGET].median(),
                "continuation_rate_pct": (group[TARGET] > 0).mean() * 100,
                "aligned_return_mean_bps": group[TARGET].mean() * 10000,
            })

            for direction, direction_group in group.groupby(DIRECTION):
                direction_label = (
                    "buyer_dominant" if direction == 1
                    else "seller_dominant"
                )
                rows.append({
                    "feature": feature,
                    "direction_group": direction_label,
                    "quantile": int(quantile),
                    "n": len(direction_group),
                    "feature_mean": direction_group[feature].mean(),
                    "aligned_return_mean": direction_group[TARGET].mean(),
                    "aligned_return_median": direction_group[TARGET].median(),
                    "continuation_rate_pct": (
                        direction_group[TARGET] > 0
                    ).mean() * 100,
                    "aligned_return_mean_bps": (
                        direction_group[TARGET].mean() * 10000
                    ),
                })

    return pd.DataFrame(rows)

def build_interaction_analysis(dataframe):
    """
    Explore whether confirmation features behave differently
    at different levels of absolute order-flow pressure.

    Pressure and confirmation features are divided into terciles.
    Results are descriptive, not out-of-sample predictions.
    """

    interactions = [
        ("directional_persistence", "pressure_x_persistence"),
        ("price_path_efficiency", "pressure_x_price_efficiency"),
    ]

    rows = []

    for feature, interaction_name in interactions:

        working = dataframe[
            [
                "abs_volume_imbalance",
                feature,
                TARGET,
            ]
        ].dropna().copy()

        if len(working) == 0:
            continue

        # Divide pressure strength and confirmation feature
        # into low, medium, and high groups.
        working["pressure_group"] = pd.qcut(
            working["abs_volume_imbalance"],
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )

        working["confirmation_group"] = pd.qcut(
            working[feature],
            q=3,
            labels=["low", "medium", "high"],
            duplicates="drop",
        )

        # Summarize each combination.
        for (pressure, confirmation), group in working.groupby(
            ["pressure_group", "confirmation_group"],
            observed=True,
        ):

            rows.append({
                "interaction": interaction_name,
                "confirmation_feature": feature,
                "pressure_group": pressure,
                "confirmation_group": confirmation,
                "n": len(group),
                "mean_aligned_return_bps": (
                    group[TARGET].mean() * 10000
                ),
                "median_aligned_return_bps": (
                    group[TARGET].median() * 10000
                ),
                "continuation_rate_pct": (
                    group[TARGET].gt(0).mean() * 100
                ),
            })

    return pd.DataFrame(rows)


def build_tail_sensitivity(dataframe, features):
    """
    Compare raw mean and median with a 1%-99% clipped-target mean.
    No observations are removed; clipping is a sensitivity diagnostic only.
    """
    working = dataframe.dropna(subset=[TARGET]).copy()
    lower = working[TARGET].quantile(TAIL_CLIP_LOWER)
    upper = working[TARGET].quantile(TAIL_CLIP_UPPER)
    working["target_clipped_1_99"] = working[TARGET].clip(lower, upper)

    rows = []
    for feature in features:
        subset = working.dropna(subset=[feature]).copy()
        subset["feature_quantile"] = quantile_labels(subset[feature])

        for quantile, group in subset.dropna(
            subset=["feature_quantile"]
        ).groupby("feature_quantile", sort=True):
            rows.append({
                "feature": feature,
                "quantile": int(quantile),
                "n": len(group),
                "raw_mean_bps": group[TARGET].mean() * 10000,
                "raw_median_bps": group[TARGET].median() * 10000,
                "clipped_mean_bps": (
                    group["target_clipped_1_99"].mean() * 10000
                ),
            })

    return pd.DataFrame(rows), lower, upper


def make_distribution_plots(dataframe, features):
    plot_features = features + [TARGET]

    for feature in plot_features:
        values = dataframe[feature].dropna()
        if values.empty:
            continue

        plt.figure(figsize=(8, 4.5))
        plt.hist(values, bins=60)
        plt.title(f"Distribution: {feature}")
        plt.xlabel(
            "Five-minute aligned return (basis points)"
            if feature == TARGET else feature
        )
        plt.ylabel("Observations")
        save_figure(f"distribution_{feature}.png")


def make_quantile_plots(quantile_data, features):
    all_data = quantile_data[
        quantile_data["direction_group"] == "all"
    ]

    for feature in features:
        subset = all_data[all_data["feature"] == feature].sort_values(
            "quantile"
        )
        if subset.empty:
            continue

        plt.figure(figsize=(7, 4.5))
        plt.plot(
            subset["quantile"],
            subset["aligned_return_mean_bps"],
            marker="o",
            label="Mean aligned return",
        )
        plt.axhline(0, linewidth=1)
        plt.xticks(sorted(subset["quantile"].unique()))
        plt.xlabel("Feature quantile (1 = lowest, 5 = highest)")
        plt.ylabel("Mean five-minute aligned return (bps)")
        plt.title(f"Five-minute continuation by {feature} quantile")
        plt.grid(True, alpha=0.25)
        save_figure(f"quantile_{feature}.png")


def make_comparison_plot(quantile_data):
    all_data = quantile_data[
        quantile_data["direction_group"] == "all"
    ]

    plt.figure(figsize=(8, 5))
    for feature in PRIMARY_COMPARISON_FEATURES:
        subset = all_data[
            all_data["feature"] == feature
        ].sort_values("quantile")
        if not subset.empty:
            plt.plot(
                subset["quantile"],
                subset["aligned_return_mean_bps"],
                marker="o",
                label=feature,
            )

    plt.axhline(0, linewidth=1)
    plt.xticks(range(1, N_QUANTILES + 1))
    plt.xlabel("Feature quantile (1 = lowest, 5 = highest)")
    plt.ylabel("Mean five-minute aligned return (bps)")
    plt.title("Primary pre-analysis feature comparison")
    plt.legend()
    plt.grid(True, alpha=0.25)
    save_figure("primary_feature_comparison.png")


def write_summary_report(
    dataframe,
    feature_summary,
    correlations,
    quantile_data,
    tail_lower,
    tail_upper,
):
    valid_target = dataframe[TARGET].dropna()
    eligible_count = len(dataframe)

    persistence_corr = correlations[
        correlations["feature"] == "directional_persistence"
    ]
    efficiency_corr = correlations[
        correlations["feature"] == "price_path_efficiency"
    ]
    aligned_vwap_corr = correlations[
        correlations["feature"] == "aligned_close_vwap_displacement"
    ]

    def corr_text(frame):
        if frame.empty:
            return "Not available"
        row = frame.iloc[0]
        return (
            f"Pearson = {row['pearson']:.4f}; "
            f"Spearman = {row['spearman']:.4f}; "
            f"n = {int(row['n']):,}"
        )

    report = f"""# Order-Flow Confirmation — Exploratory Analysis

## Scope

This report is descriptive and exploratory. It does not establish predictive
performance, statistical significance, or trading profitability.

- Eligible observations: {eligible_count:,}
- Five-minute target observations: {valid_target.count():,}
- Mean aligned five-minute return: {valid_target.mean() * 10000:.4f} bps
- Median aligned five-minute return: {valid_target.median() * 10000:.4f} bps
- Positive aligned-return frequency: {(valid_target > 0).mean() * 100:.2f}%
- Target 1st percentile: {tail_lower:.8f}
- Target 99th percentile: {tail_upper:.8f}

## Pre-analysis prediction

The researcher's prediction is that directional persistence will be more
informative for five-minute continuation than price-path efficiency, because
aggressor sequencing may capture sustained directional pressure while
price-path efficiency describes a completed intraminute price movement.

## Initial correlations with the five-minute aligned return

- Directional persistence: {corr_text(persistence_corr)}
- Price-path efficiency: {corr_text(efficiency_corr)}
- Aligned close-VWAP displacement: {corr_text(aligned_vwap_corr)}

These are in-sample descriptive correlations. They are not causal estimates
and do not account for overlapping forward-return windows or session-level
dependence.

## Interpretation safeguards

- Extreme observations are retained in the primary analysis.
- A 1st-to-99th percentile clipped-target mean is reported only as a sensitivity
  diagnostic; it does not replace the raw outcome.
- Quantile groups are descriptive and are not a final model specification.
- No p-values or claims of statistical significance are produced.
- The 15-session sample is limited; chronological out-of-sample validation
  remains necessary.
- Trade-size concentration describes trade records, not unique participants.

## Generated outputs

See the CSV tables and figures in this report directory:
- `tables/feature_summary.csv`
- `tables/correlations_pearson_spearman.csv`
- `tables/quantile_analysis.csv`
- `tables/feature_interactions.csv`
- `tables/tail_sensitivity.csv`
- `figures/`
"""
    path = OUTPUT_DIR / "exploratory_report.md"
    path.write_text(report)
    print(f"Saved: {path.relative_to(PROJECT_ROOT)}")


# ============================================================
# Main analysis
# ============================================================

def main():
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in cast",
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Research dataset not found: {DATA_PATH}\n"
            "Run build_research_dataset.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading research dataset...")
    df = pd.read_parquet(DATA_PATH)

    required_context_columns = [
        "volume_imbalance" if column == "abs_volume_imbalance" else column
        for column in CONTEXT_FEATURES
    ]

    require_columns(
        df,
        BASE_CONFIRMATION_FEATURES
        + required_context_columns
        + [TARGET, DIRECTION, "minute"],
    )

    # Directionally align signed intraminute confirmation variables.
    df = df.copy()

    # Absolute imbalance measures pressure strength, independent of direction.
    df["abs_volume_imbalance"] = df["volume_imbalance"].abs()

    df["aligned_close_vwap_displacement"] = (
        df[DIRECTION] * df["close_vwap_displacement"]
    )
    df["aligned_late_early_pressure_diff"] = (
        df[DIRECTION] * df["late_early_pressure_diff"]
    )
    df["aligned_buy_sell_vwap_diff"] = (
        df[DIRECTION] * df["buy_sell_vwap_diff"]
    )

    confirmation_features = (
        BASE_CONFIRMATION_FEATURES
        + [
            "aligned_close_vwap_displacement",
            "aligned_late_early_pressure_diff",
            "aligned_buy_sell_vwap_diff",
        ]
    )

    analysis_features = confirmation_features + CONTEXT_FEATURES

    print("\n========== DATASET CHECK ==========")
    print(f"Rows: {len(df):,}")
    print(f"Five-minute target observations: {df[TARGET].notna().sum():,}")
    print(
        "Five-minute aligned-return mean: "
        f"{df[TARGET].mean() * 10000:.4f} bps"
    )
    print(
        "Five-minute continuation frequency: "
        f"{(df[TARGET].dropna() > 0).mean() * 100:.2f}%"
    )

    overall_direction, session_direction = (build_direction_diagnostics(df))

    save_csv(
        overall_direction,
        "overall_direction_diagnostics.csv",
    )

    save_csv(
        session_direction,
        "session_direction_diagnostics.csv",
    )

    print("\n========== FEATURE SUMMARY ==========")
    summary = descriptive_feature_summary(df, analysis_features)
    save_csv(summary, "feature_summary.csv")
    print(summary.to_string(index=False))

    print("\n========== CORRELATIONS ==========")
    corr_features = analysis_features + [TARGET]
    corr_data = df[corr_features].copy()

    pearson = corr_data.corr(method="pearson")[TARGET].drop(TARGET)
    spearman = corr_data.corr(method="spearman")[TARGET].drop(TARGET)

    correlation_table = pd.DataFrame({
        "feature": pearson.index,
        "pearson": pearson.values,
        "spearman": spearman.reindex(pearson.index).values,
        "n": [
            int(df[[feature, TARGET]].dropna().shape[0])
            for feature in pearson.index
        ],
    }).sort_values(
        "spearman",
        key=lambda series: series.abs(),
        ascending=False,
    )

    save_csv(correlation_table, "correlations_pearson_spearman.csv")
    print(correlation_table.to_string(index=False))

    print("\n========== QUANTILE ANALYSIS ==========")
    quantile_data = build_quantile_analysis(
        df,
        confirmation_features,
    )
    save_csv(quantile_data, "quantile_analysis.csv")

    print("\n--- All observations ---")
    print(
        quantile_data[
            quantile_data["direction_group"] == "all"
        ].to_string(index=False)
    )

    print("\n--- Buyer-dominant observations ---")
    print(
        quantile_data[
            quantile_data["direction_group"] == "buyer_dominant"
        ].to_string(index=False)
    )

    print("\n--- Seller-dominant observations ---")
    print(
        quantile_data[
            quantile_data["direction_group"] == "seller_dominant"
        ].to_string(index=False)
    )

    print("\n========== FEATURE INTERACTIONS ==========")

    interaction_data = build_interaction_analysis(df)

    save_csv(
        interaction_data,
        "feature_interactions.csv",
    )

    print(interaction_data.to_string(index=False))

    print("\n========== EXTREME-RETURN SENSITIVITY ==========")
    tail_data, tail_lower, tail_upper = build_tail_sensitivity(
        df,
        confirmation_features,
    )
    save_csv(tail_data, "tail_sensitivity.csv")
    print(
        f"Target clipping thresholds for sensitivity only: "
        f"{tail_lower:.8f}, {tail_upper:.8f}"
    )

    print("\n========== GENERATING FIGURES ==========")
    make_distribution_plots(df, analysis_features)
    make_quantile_plots(quantile_data, confirmation_features)
    make_comparison_plot(quantile_data)

    write_summary_report(
        df,
        summary,
        correlation_table,
        quantile_data,
        tail_lower,
        tail_upper,
    )

    print("\n========== COMPLETE ==========")
    print(f"Outputs: {OUTPUT_DIR}")
    print(
        "Reminder: treat all correlations and quantile patterns as "
        "exploratory, not confirmatory."
    )


if __name__ == "__main__":
    main()

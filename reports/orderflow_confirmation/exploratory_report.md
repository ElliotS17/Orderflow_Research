# Order-Flow Confirmation — Exploratory Analysis

## Scope

This report is descriptive and exploratory. It does not establish predictive
performance, statistical significance, or trading profitability.

- Eligible observations: 20,316
- Five-minute target observations: 20,241
- Mean aligned five-minute return: -0.0294 bps
- Median aligned five-minute return: 0.0000 bps
- Positive aligned-return frequency: 49.43%
- Target 1st percentile: -0.00204919
- Target 99th percentile: 0.00207856

## Pre-analysis prediction

The researcher's prediction is that directional persistence will be more
informative for five-minute continuation than price-path efficiency, because
aggressor sequencing may capture sustained directional pressure while
price-path efficiency describes a completed intraminute price movement.

## Initial correlations with the five-minute aligned return

- Directional persistence: Pearson = -0.0024; Spearman = -0.0000; n = 20,241
- Price-path efficiency: Pearson = 0.0017; Spearman = 0.0047; n = 20,241
- Aligned close-VWAP displacement: Pearson = -0.0020; Spearman = -0.0102; n = 20,241

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
- `feature_summary.csv`
- `correlations_pearson_spearman.csv`
- `quantile_analysis.csv`
- `feature_interactions.csv`
- `tail_sensitivity.csv`
- `figures/`

# MFT Orderflow Research

## 1. Project Objective

Develop a comprehensive universe of orderflow and market-context features using Nasdaq-100 E-mini futures (NQ) trade data.

The central research question is whether observable buyer- and seller-initiated trading activity contains predictive information about subsequent price movements.

While making this project I tried emphasizing creative feature engineering, statistical analysis, and reproducibility. The feature universe is intended to support multiple future research questions rather than being designed around one predetermined hypothesis.

## 2. Dataset

- **Provider:** Databento
- **Dataset:** GLBX.MDP3
- **Schema:** Trades
- **Instrument:** NQZ6
- **Coverage:** October 12–30, 2026
- **Trading sessions:** 15
- **Primary timestamps:** UTC
- **Raw trade records:** 7,648,473
- **Deduplicated trade records:** 7,607,886

The primary research dataset uses deduplicated trade records. Original records are retained for potential additional analysis.

Exact duplicate records were removed based on trade-message fields, excluding source-file identity. I could not find a unique execution identifier, so deduplication should not be interpreted as proof that every removed record was an invalid event.

The trade-side field identifies aggressor direction:
- `B`: buyer-initiated trade
- `A`: seller-initiated trade
- `N`: unknown aggressor direction

Unknown-side records are included in total activity and volume measurements but excluded from directional measurements.

## 3. Project Structure

- `data/`: Raw and processed market data
- `notebooks/`: Research notebooks and documentation
- `src/`: Active research pipeline
  - `data/`: Trade dataset construction
  - `features/`: Base aggregation and feature engineering
  - `diagnostics/`: Data inspection, validation, and feature diagnostics
- `archive/`: Previous exploratory research, including hypothesis-specific datasets, scripts, and results
- `figures/`: Research visualizations
- `results/`: Current research outputs

### Key datasets

| Dataset | Description |
|---|---|
| `trades_deduplicated.parquet` | Primary deduplicated trade-level input |
| `features_1min_deduplicated.parquet` | Existing 29-column baseline minute dataset |
| `features_1min_universe.parquet` | Final baseline plus 95 new features |

The feature-universe builder is `src/features/build_feature_universe.py`.

## 4. Feature Universe: Completed

The feature universe contains **95 features across six families**, in addition to the 29 baseline columns.

The final dataset contains:
- 20,580 minute-level observations
- 124 total columns, including the `minute` timestamp
- 29 baseline columns
- 95 new features

### 4.1 Trade Size — 12 features

Describes the distribution and concentration of individual trade sizes within each minute.

- `trade_size_std`
- `trade_size_p90`
- `trade_size_p95`
- `large_trade_count_3`
- `large_trade_volume_share_3`
- `large_trade_count_5`
- `large_trade_volume_share_5`
- `large_trade_count_10`
- `large_trade_volume_share_10`
- `trade_size_hhi`
- `largest_trade_volume_share`
- `top_5_trade_volume_share`

The large-trade thresholds are measured in contracts. HHI is the sum of squared individual-trade volume shares within a minute. These are measures of concentration across trade records, not identified market participants.

### 4.2 Timing and Activity — 17 features

Describes how trading activity is distributed through each minute.

- `early_trade_count`
- `early_volume`
- `early_net_volume`
- `early_directional_volume`
- `late_trade_count`
- `late_volume`
- `late_net_volume`
- `late_directional_volume`
- `early_volume_share`
- `late_volume_share`
- `late_early_pressure_diff`
- `activity_time_centroid`
- `max_5s_volume_share`
- `max_5s_trade_count`
- `trade_intensity_cv`
- `max_10s_volume_share`
- `max_10s_trade_count`

Early activity is defined as the first 30 seconds of a minute; late activity is the final 30 seconds. Five- and ten-second activity measures use fixed, non-overlapping windows. The activity-time centroid is volume-weighted.

`late_early_pressure_diff` compares normalized directional pressure between the two halves. It is undefined when either half has no directional volume.

### 4.3 Directional Sequencing — 11 features

Describes the ordering and persistence of buyer- and seller-initiated trades.

- `directional_transition_count`
- `side_switch_count`
- `side_switch_rate`
- `directional_persistence`
- `buy_run_count`
- `avg_buy_run_length`
- `max_buy_run_length`
- `sell_run_count`
- `avg_sell_run_length`
- `max_sell_run_length`
- `max_directional_run_length`

Unknown-side trades break directional runs and are not silently removed to connect otherwise non-adjacent observations. Trade ordering uses event timestamps with receive timestamps as a secondary sort key. Tied timestamps mean the precise ordering of some records is approximate.

### 4.4 Intraminute Price Path — 7 features

Describes the path taken by trade prices within a minute, beyond the baseline OHLC summary.

- `max_upward_excursion`
- `max_downward_excursion`
- `recovery_from_low`
- `pullback_from_high`
- `price_path_length`
- `price_path_efficiency`
- `price_reversal_count`

Excursions and path length are expressed in NQ price points. Path efficiency compares net intraminute displacement with total absolute price movement. Reversal count measures direction changes between nonzero consecutive price changes; unchanged prices are ignored.

### 4.5 Price-Level Distribution — 15 features

Describes where trading volume occurs across price levels and where aggressor-specific activity takes place.

- `distinct_price_levels`
- `price_level_hhi`
- `largest_price_level_share`
- `price_level_entropy`
- `point_of_control_price`
- `vwap`
- `close_vwap_displacement`
- `close_poc_displacement`
- `lower_range_volume_share`
- `upper_range_volume_share`
- `buy_vwap`
- `sell_vwap`
- `buy_sell_vwap_diff`
- `buy_upper_range_volume_share`
- `sell_lower_range_volume_share`

Price-level HHI is calculated from the volume shares at each traded price. Entropy is normalized by the logarithm of the number of distinct price levels.

The point of control (POC) is the price level with the greatest traded volume during the minute. It is a description of executed volume, not resting liquidity or a confirmed support/resistance level.

The lower and upper range shares use the lower and upper thirds of the minute's traded price range. Aggressor-specific features use buyer- or seller-initiated volume; unknown-side trades are excluded from these calculations.

### 4.6 Market Context — 33 features

Describes recent market conditions using information from prior minutes. Three rolling horizons are used: 5, 15, and 30 minutes.

For each horizon:

- `prior_{window}m_net_volume`
- `prior_{window}m_directional_volume`
- `prior_{window}m_volume_imbalance`
- `relative_volume_{window}m`
- `relative_trade_count_{window}m`
- `prior_{window}m_price_change`
- `prior_{window}m_price_efficiency`
- `prior_{window}m_abs_price_change`
- `prior_{window}m_price_range`
- `prior_{window}m_return_std`
- `prior_{window}m_return`

The `{window}` placeholder is replaced by `5`, `15`, or `30`.

Rolling calculations are reset across session gaps and shifted to use prior observations. Relative volume and relative trade count describe the current completed minute relative to the preceding window's average activity. These current-minute-normalized features are available only after the minute has completed.

These features provide market-state context and controls; they are not all pure orderflow measurements.

## 5. Validation and Redundancy Audit

### Structural validation

The saved feature dataset was independently checked.

- Shape: 20,580 rows × 124 columns
- Duplicate columns: 0
- Duplicate minute timestamps: 0
- Chronological ordering: passed
- All 29 baseline columns preserved: passed
- Baseline minute sequence preserved: passed
- All six feature families have unique minute indexes matching the baseline.
- No new feature names overlap with baseline columns.

The builder also reads the saved Parquet file back and checks its shape, column names, and minute sequence.

### Mathematical redundancy

Fifteen exact mathematical relationships were verified across the 20,580 observations.

They include:
- Side-switch rate and directional persistence sum to one.
- Early and late trade counts, volume, net volume, and directional volume reconstruct their corresponding minute totals.
- Early and late volume shares sum to one.
- Maximum directional run length equals the maximum of the buy- and sell-run maxima.
- Price-path efficiency is derived from absolute intraminute displacement divided by path length.
- Intraminute excursions and recovery/pullback measures can be reconstructed from OHLC values.
- Aggressor VWAP difference and close-to-VWAP/POC displacements are derived from their component prices.

These relationships identify exact derivations and complements. They do not automatically make the features useless: some remain intuitive summaries or useful standalone descriptors.

### Statistical redundancy

A Pearson correlation audit examined 118 numeric variables, producing 6,903 unique feature pairs. Correlations were calculated with a minimum of 1,000 paired observations.

Important findings:
- `side_switch_rate` and `directional_persistence` are perfectly negatively correlated.
- `early_volume_share` and `late_volume_share` are perfectly negatively correlated.
- Several sequencing counts are strongly related to overall trade activity.
- Prior absolute price movement and return standard deviation are highly correlated at some horizons.
- Relative volume and relative trade count are highly correlated.
- Price-level HHI and largest price-level share are related, as are trade-size HHI and top-five trade volume share.
- Raw price-level variables are highly correlated because they share the same underlying NQ price level.

High correlation does not establish that two features contain identical information. No features were removed solely because of Pearson correlation. The full 95-feature catalog was retained.

## 6. Missing Values

Missing values have been retained rather than replaced with zero.

Known patterns include:
- `gap_minutes`: one missing value at the first observation in the dataset.
- Baseline close-change and return measures: 15 missing values at the first observation of each session, where no prior in-session minute exists.
- `late_early_pressure_diff`: four missing values where one half-minute has no directional trades.
- Rolling context features: missing at the beginning of each session until the required historical window is available. Return-based context features also require an additional prior close.

## 7. Data and Methodological Limitations

The dataset contains individual trade records, not full order-book depth.

Consequently, the research cannot directly observe:
- Resting bid or ask liquidity
- Queue position
- Order cancellations
- Limit-order replenishment
- Actual depth depletion
- Hidden orders or participant identity

This limitation made for a interesting opportunity to attempt capturing characteristics of order flow based on trade activity rather than using level 2 data.

Other limitations:
- The sample covers 15 trading sessions, which limits the reliability of long-run estimates and time-of-day normalization.
- Event timestamps can be tied. Stable sorting by event and receive timestamps provides a consistent ordering, but not necessarily the true ordering of all tied records.
- The `N` aggressor-side category is small overall but can matter in individual minutes. Total and directional volume are therefore kept distinct.
- Price-path measures can be sensitive to trade-level noise and timestamp ordering.
- Feature relationships observed in this sample should not be assumed to generalize without further testing.

## 8. Completed Research: Order-Flow Confirmation

After the feature-universe development phase I conducted a hypothesis-driven study of short-term price continuation to access how the feature families could be used together.

**Research question:** When buying or selling pressure occurs, do additional characteristics of order flow—such as trade concentration, trading activity, directional sequencing, and price behavior, help predict whether prices will continue moving in that direction over the next five minutes?

The study used 15 trading sessions of NQ futures data from October 12–30, 2026.

The research pipeline:
1. Constructed a hypothesis-specific minute-level dataset.
2. Engineered aligned forward-return targets at one-, five-, and fifteen-minute horizons.
3. Conducted exploratory feature and directional analysis.
4. Compared baseline, context-adjusted, and confirmation Ridge regression models.
5. Evaluated performance using expanding chronological out-of-sample validation.

### Main Findings

The confirmation features did not improve out-of-sample predictive performance.

| Model | MAE (bps) | OOS R² | Directional accuracy |
|---|---:|---:|---:|
| Baseline | 5.7222 | -0.0004 | 51.05% |
| Context | 5.7238 | -0.0015 | 50.52% |
| Confirmation | 5.7262 | -0.0024 | 50.43% |

Exploratory analysis identified a possible asymmetry between buyer-dominant and seller-dominant observations. This pattern was inconsistent across sessions and is treated as a potential research direction rather than an established effect.

**Conclusion:** The tested order-flow confirmation features did not demonstrate incremental predictive value for five-minute price continuation in this sample but led to another discovery that could be pursued further in the future.

For the complete methodology, results, limitations, and interpretation, see:

- [Final Research Report](reports/orderflow_confirmation/final_research_report.md)
- [Research Pipeline Documentation](src/research/README.md)

## 9. Research Methodology and Project Status

The project progressed through the following stages:

1. Validate the underlying trade data.
2. Construct baseline minute-level aggregations.
3. Engineer a broad, hypothesis-independent feature universe.
4. Document feature definitions, information availability, and limitations.
5. Diagnose distributions, missingness, outliers, and redundancy.
6. Finalize the feature universe.
7. Develop a research hypothesis using the established feature catalog.
8. Construct a hypothesis-specific dataset and forward outcome variables.
9. Evaluate the hypothesis using chronological out-of-sample validation.
10. Document findings, limitations, and reproducible results.

**Current status:** The feature universe and the initial order-flow confirmation study are complete. The original confirmation hypothesis was not supported by the model comparison. Further research directions are documented as potential extensions.

## 10. Data and Reproducibility

The research scripts and documentation are maintained in the repository. Market-data files are excluded from Git.

The hypothesis-specific research pipeline requires:

```text
data/processed/features_1min_universe.parquet
data/processed/research_orderflow_confirmation.parquet
```

See [Research Pipeline Documentation](src/research/README.md) for reproduction commands and output descriptions.

Raw and processed market data are not included in the repository.

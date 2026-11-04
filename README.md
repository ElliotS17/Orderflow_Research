# MFT Orderflow Research

## 1. Project Objective

Develop a comprehensive, hypothesis-independent universe of orderflow and market-context features using Nasdaq-100 E-mini futures (NQ) trade data.

The central research question is whether observable buyer- and seller-initiated trading activity contains predictive information about subsequent price movements.

The project emphasizes creative feature engineering, statistical analysis, independent research contributions, and reproducibility. The feature universe is intended to support multiple future research questions rather than being designed around one predetermined hypothesis.

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

The primary research dataset uses deduplicated trade records. Original records are retained for potential sensitivity analysis.

Exact duplicate records were removed based on trade-message fields, excluding source-file identity. A unique execution identifier was not established, so deduplication should not be interpreted as proof that every removed record was an invalid event.

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

The feature universe contains **95 newly engineered features across six families**, in addition to the 29 baseline columns.

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

These relationships identify exact derivations and complements. They do not automatically make the features useless: some remain intuitive summaries or useful standalone descriptors. Their derivation should be considered during later feature selection.

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

High correlation does not establish that two features contain identical information. No features were removed solely because of Pearson correlation. The full 95-feature catalog is retained; a smaller modeling shortlist can be selected later based on the research question, interpretability, and out-of-sample evaluation.

### Preliminary modeling priorities

Some features are lower-priority candidates because they are exact derivations, complements, or strongly activity-sensitive.

Examples include:
- Choosing one of `side_switch_rate` and `directional_persistence` for a model.
- Choosing one of `early_volume_share` and `late_volume_share` when both add no independent information.
- Treating `directional_transition_count`, `buy_run_count`, and `sell_run_count` cautiously because they closely track activity and sequencing counts.
- Treating `price_reversal_count` cautiously because it is strongly associated with trade count.

These are modeling considerations, not reasons to delete features from the research catalog.

## 6. Missing Values

Missing values have been retained rather than replaced with zero.

Known patterns include:
- `gap_minutes`: one missing value at the first observation in the dataset.
- Baseline close-change and return measures: 15 missing values at the first observation of each session, where no prior in-session minute exists.
- `late_early_pressure_diff`: four missing values where one half-minute has no directional trades.
- Rolling context features: missing at the beginning of each session until the required historical window is available. Return-based context features also require an additional prior close.

These missing values reflect unavailable history or undefined calculations, not necessarily zero activity or zero price movement.

## 7. Data and Methodological Limitations

The dataset contains individual trade records, not full order-book depth.

Consequently, the research cannot directly observe:
- Resting bid or ask liquidity
- Queue position
- Order cancellations
- Limit-order replenishment
- Actual depth depletion
- Hidden orders or participant identity

Large trades and trade-size concentration must not be interpreted as proof of institutional activity or the number of independent participants. Individual trade records do not identify unique market participants.

Other limitations:
- The sample covers 15 trading sessions, which limits the reliability of long-run estimates and time-of-day normalization.
- Event timestamps can be tied. Stable sorting by event and receive timestamps provides a consistent ordering, but not necessarily the true ordering of all tied records.
- The `N` aggressor-side category is small overall but can matter in individual minutes. Total and directional volume are therefore kept distinct.
- Price-path measures can be sensitive to trade-level noise and timestamp ordering.
- Feature relationships observed in this sample should not be assumed to generalize without further testing.

## 8. Research Directions Raised During Feature Exploration

The following are exploratory research directions, not established hypotheses or tested findings. They preserve ideas raised during the collaborative feature-design process.

- **Late-minute bursts:** whether aggressive activity concentrated near the end of a minute has a different relationship with subsequent price movement than earlier activity.
- **Concentrated versus distributed participation:** whether a similar amount of volume expressed through fewer large trades differs from volume distributed across many small trades. Trade records do not identify participants.
- **Intraminute price path:** whether minutes with similar net buying pressure and closing displacement differ according to their internal price path, including reversals and recoveries.
- **Late concentrated buying:** whether buying concentrated into a short late-minute interval carries different information from the same total buying pressure distributed throughout the minute.
- **POC displacement:** whether the closing price's displacement from the minute's point of control is associated with subsequent movement.
- **Aggressor-specific price distribution:** whether the relative execution prices of buyer- and seller-initiated trades add information beyond aggregate volume imbalance.
- **Directional concentration interaction:** whether trade-size concentration combined with one-sided aggressor imbalance is more informative than concentration considered alone.

These ideas will be revisited after the feature universe is finalized and the next research question is deliberately selected.

## 9. Research Methodology and Next Phase

The project follows this sequence:

1. Validate the underlying trade data.
2. Construct baseline minute-level aggregations.
3. Engineer a broad, hypothesis-independent feature universe.
4. Document feature definitions, information availability, and limitations.
5. Diagnose distributions, missingness, outliers, and redundancy.
6. Finalize the feature universe.
7. Develop a research hypothesis using the established feature catalog.
8. Construct a hypothesis-specific dataset and future outcome variables.
9. Evaluate the hypothesis using appropriate statistical methods and out-of-sample safeguards.
10. Document findings, limitations, and reproducible results.

**Current status:** the 95-feature universe has been engineered, validated, and saved. The next phase is research design and hypothesis development. No new hypothesis has been tested as part of the feature-universe construction.

## 10. Previous Research

Previous exploratory work examined the relationship between strong directional orderflow, price response, and subsequent returns.

That research has been archived under `archive/previous_hypothesis/`. It is not the starting point for the current feature-universe development phase, and its exploratory findings are not treated as established conclusions.

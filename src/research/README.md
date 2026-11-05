# Order-Flow Confirmation and Short-Term Price Continuation

## Research Status

**Status:** Empirical study completed
**Primary outcome:** Original hypothesis not supported by the model comparison
**Primary prediction horizon:** Five minutes
**Validation:** Expanding chronological out-of-sample evaluation

This study investigates whether characteristics of executed order flow provide information about short-term price continuation beyond conventional directional buying and selling pressure.

The study uses the existing minute-level feature universe to construct a hypothesis-specific dataset, conduct exploratory analysis, and compare three progressively expanded predictive models.

For the complete methodology, findings, and interpretation, see [the final research report](../../reports/orderflow_confirmation/final_research_report.md).

## 1. Research Question and Hypothesis

**Research question**

When buying or selling pressure occurs, do additional characteristics of order flow—such as trade concentration, trading activity, directional sequencing, and price behavior—help predict whether prices will continue moving in that direction over the next five minutes?

**Hypothesis**

Buying or selling pressure accompanied by confirming characteristics in trade concentration, trading activity, directional sequencing, and price behavior will be followed by stronger five-minute price continuation than pressure without those characteristics.

The hypothesis was not supported by the final model comparison.

## 2. Data and Sample

The study uses trade-level NQ futures data from the Databento GLBX.MDP3 dataset, covering contract NQZ6 and 15 trading sessions from October 12–30, 2026.

The existing feature universe contains 20,580 minute-level observations and 124 columns, including 29 baseline and 95 engineered features.

The hypothesis-specific research dataset contains:

- 20,316 eligible observations.
- 20,241 valid five-minute forward-return targets.
- 15 trading sessions.

The primary sample includes every minute satisfying all of the following:

- `volume_imbalance` is present and finite.
- `volume_imbalance` is nonzero.
- `directional_volume` is greater than zero.

The sign of `volume_imbalance` defines the current-minute order-flow direction:

- Positive: buyer-initiated volume exceeds seller-initiated volume.
- Negative: seller-initiated volume exceeds buyer-initiated volume.

No minimum imbalance threshold is imposed.

## 3. Feature Specification

The model comparison uses three feature groups.

### 3.1 Baseline directional pressure

| Feature | Role |
|---|---|
| `orderflow_direction` | Sign of net aggressive volume; identifies buyer- or seller-dominant pressure. |
| `abs_volume_imbalance` | Magnitude of net aggressive volume, independent of direction. |
| `directional_volume` | Amount of volume with a known buyer- or seller-initiated classification. |

### 3.2 Prior market context

| Feature | Role |
|---|---|
| `aligned_prior_5m_volume_imbalance` | Prior five-minute order-flow imbalance aligned with current-minute direction. |
| `aligned_prior_5m_return` | Prior five-minute return aligned with current-minute order-flow direction. |
| `relative_volume_5m` | Controls for recent relative trading activity. |
| `prior_5m_return_std` | Controls for recent return variability. |

### 3.3 Order-flow confirmation

| Dimension | Feature | Research rationale |
|---|---|---|
| Trade concentration | `trade_size_hhi` | Measures concentration across individual trade records. |
| Activity timing | `max_10s_volume_share` | Measures volume concentration in the busiest fixed ten-second interval. |
| Activity timing | `late_early_pressure_diff` | Measures changes in normalized directional pressure within a minute. |
| Directional sequencing | `directional_persistence` | Measures persistence of buyer- or seller-initiated trade direction. |
| Price path | `price_path_efficiency` | Measures net intraminute displacement relative to total path length. |
| Price positioning | `close_vwap_displacement` | Measures closing price displacement from minute VWAP. |
| Aggressor-specific pricing | `buy_sell_vwap_diff` | Compares buyer-initiated and seller-initiated VWAP. |

Redundant or closely related features were excluded from the initial modeling shortlist to reduce unnecessary feature duplication.

## 4. Target Construction

The primary horizon is five minutes. One-minute and fifteen-minute horizons are retained for secondary exploration.

For horizon \(h\), the directionally aligned forward log return is:

\[
R^{aligned}_{t,h}
=
\operatorname{sign}(\text{volume\_imbalance}_t)
\log\left(\frac{C_{t+h}}{C_t}\right)
\]

Interpretation:

- Positive aligned return: price continued in the direction of current-minute order flow.
- Negative aligned return: price moved against current-minute order flow.
- Zero aligned return: no net close-to-close price change.

Forward targets are calculated within contiguous one-minute segments. Targets do not cross missing-minute gaps, session breaks, or weekends.

## 5. Exploratory Analysis

The exploratory analysis examined feature distributions, correlations, quantile relationships, feature interactions, tail sensitivity, and buyer-seller directional asymmetry.

The mean aligned five-minute return was approximately -0.0294 basis points, with a continuation frequency of 49.43%.

Individual confirmation features exhibited very weak correlations with the aligned target. Quantile patterns and feature interactions did not reveal consistent, strongly monotonic relationships.

### Buyer-Seller Asymmetry

Exploratory diagnostics showed differences between buyer-dominant and seller-dominant observations:

| Group | Observations | Mean aligned return | Continuation frequency |
|---|---:|---:|---:|
| Buyer-dominant | 10,175 | -0.2565 bps | 47.43% |
| Seller-dominant | 10,066 | +0.2002 bps | 51.45% |

This asymmetry was not consistent across all sessions and is treated as an exploratory observation rather than an established market effect.

One possible interpretation is that buying and selling pressure may translate into price movement differently. The idea that prices may rise more gradually but fall more sharply is a potential explanation for further investigation, not a demonstrated conclusion.

## 6. Model Comparison

Three Ridge regression models were evaluated.

| Model | Features | Purpose |
|---|---|---|
| Baseline | Directional pressure | Establish predictive performance from conventional order-flow measures. |
| Context | Baseline + prior market context | Evaluate whether recent market conditions add information. |
| Confirmation | Context + seven confirmation features | Evaluate whether additional order-flow characteristics improve prediction. |

All models use Ridge regression with `alpha=1.0`.

Missing values are median-imputed and features are standardized within a scikit-learn pipeline. Preprocessing is fitted on training data only.

### Validation Design

An expanding chronological validation design was used:

1. The first ten sessions form the initial training window.
2. Each of the next five sessions is held out individually.
3. The training window expands to include all sessions preceding the current test session.
4. Minute observations are never randomly shuffled.

The primary target is the five-minute aligned forward log return.

### Pooled Out-of-Sample Results

| Model | Observations | MAE (bps) | OOS R² | Directional accuracy |
|---|---:|---:|---:|---:|
| Baseline | 6,782 | 5.7222 | -0.0004 | 51.05% |
| Context | 6,782 | 5.7238 | -0.0015 | 50.52% |
| Confirmation | 6,782 | 5.7262 | -0.0024 | 50.43% |

The baseline model performed marginally better than the expanded models.

All three models produced negative pooled out-of-sample R². Directional accuracy remained close to 50%.

Adding prior market context did not improve overall performance, and adding the confirmation features did not improve performance over the context model.

These results do not support the original hypothesis in the tested sample.

## 7. Limitations

- The sample covers one futures contract and 15 trading sessions, with only five out-of-sample test sessions.
- Five-minute forward returns overlap across adjacent observations, introducing serial dependence.
- No inferential significance testing was performed.
- The data contains executed trades, not full order-book depth or participant identities.
- Trade-size concentration is calculated across trade records, not unique participants.
- Tied event timestamps are ordered using receive timestamps as a secondary key; exact ordering of tied events remains approximate.
- The study uses Ridge regression with a fixed regularization parameter and does not exhaustively test alternative model classes.
- No transaction costs, slippage, execution assumptions, or trading profitability are evaluated.

The findings should not be generalized beyond the sampled contract, dates, and modeling approach without further testing.

## 8. Reproducibility

Run commands from the repository root.

### Required data

The research pipeline requires the following authorized local data files:

```text
data/processed/features_1min_universe.parquet
data/processed/research_orderflow_confirmation.parquet
```

The first is required to rebuild the hypothesis-specific research dataset. The second is used by the exploratory and model-comparison scripts.

These data files are intentionally excluded from Git.

### Build the research dataset

```bash
python3 src/research/build_research_dataset.py
```

Input:

```text
data/processed/features_1min_universe.parquet
```

Output:

```text
data/processed/research_orderflow_confirmation.parquet
```

### Run exploratory analysis

```bash
python3 src/research/explore_orderflow_confirmation.py
```
### Outputs

All research outputs are stored under:

reports/orderflow_confirmation/

- `final_research_report.md`: authoritative research report.
- `exploratory_report.md`: generated exploratory summary.
- `tables/`: CSV analysis tables and model-comparison metrics.
- `figures/`: exploratory visualizations.
- `model_comparison/`: local out-of-sample prediction artifacts, including the predictions Parquet file.

Current local datasets and generated prediction Parquet files are excluded from Git. Some historical archived artifacts were committed earlier and remain tracked.

### Run model comparison

```bash
python3 src/research/run_model_comparison.py
```

Model outputs are split across two directories:

- `reports/orderflow_confirmation/tables/`: session metrics, pooled metrics, and model comparison.
- `reports/orderflow_confirmation/model_comparison/`: out-of-sample prediction Parquet file.

The scripts require the Python dependencies used by the project, including pandas, NumPy, PyArrow, Matplotlib, and scikit-learn.

## 9. Researcher Decision Log

- **Research direction:** Order-flow confirmation and short-term price continuation.
- **Primary sample:** All minutes with finite, nonzero volume imbalance and positive directional volume.
- **Primary horizon:** Five minutes.
- **Secondary horizons:** One minute and fifteen minutes.
- **Primary model:** Ridge regression.
- **Validation:** Expanding chronological evaluation across five held-out sessions.
- **Final finding:** The confirmation features did not improve pooled out-of-sample model performance.

The exploratory buyer-seller asymmetry and the possible difference between gradual upward movement and sharper downward movement remain potential future research directions.

## 10. Final Conclusion

The project developed a broad minute-level order-flow feature universe and evaluated whether additional characteristics of executed trading activity improve short-term price-continuation predictions.

The selected confirmation features did not improve out-of-sample model performance in the tested sample. The study therefore does not establish a predictive or profitable trading signal.

Its contribution is the research process: feature engineering, hypothesis development, target construction, chronological validation, comparative modeling, and transparent interpretation of a hypothesis that was not supported by the results.

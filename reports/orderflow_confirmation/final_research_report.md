# Order-Flow Confirmation and Short-Term Price Continuation

## Abstract

This study investigates whether characteristics of futures order flow provide information about short-term price continuation beyond basic buying and selling pressure. Using trade-level Nasdaq-100 E-mini futures (NQ) data, the study constructs minute-level order-flow features and evaluates whether trade concentration, trading activity, directional sequencing, and price behavior improve predictions of five-minute price continuation.

The analysis uses 15 trading sessions from October 12–30, 2026, and evaluates three progressively expanded Ridge regression models using chronological out-of-sample validation. The confirmation model does not outperform the baseline model. All three models exhibit negative pooled out-of-sample R² and directional accuracy close to 50%. The findings do not support the hypothesis that the selected confirmation features improve short-term continuation predictions in this sample.

## 1. Research Question and Hypothesis

**Research question**

When buying or selling pressure occurs, do additional characteristics of order flow—such as trade concentration, trading activity, directional sequencing, and price behavior—help predict whether prices will continue moving in that direction over the next five minutes?

**Hypothesis**

Buying or selling pressure accompanied by confirming characteristics in trade concentration, trading activity, directional sequencing, and price behavior will be followed by stronger five-minute price continuation than pressure without those characteristics.

## 2. Data and Sample

The study uses trade-level NQ futures data from the Databento GLBX.MDP3 dataset, covering contract NQZ6 and 15 trading sessions between October 12 and October 30, 2026.

Trade records are aggregated into one-minute observations. The resulting research dataset contains 20,316 eligible observations, of which 20,241 have valid five-minute forward-return targets.

The analysis uses classified buyer- and seller-initiated trading volume. Trades with unknown aggressor side are excluded from directional calculations.

The dataset represents executed trades rather than the full order book. Consequently, the analysis does not measure resting liquidity, cancellations, queue position, participant identity, or hidden orders.

## 3. Feature Engineering

The feature universe contains 124 minute-level features, including 29 baseline features and 95 engineered features across six feature families.

The research dataset focuses on three groups:

**Baseline pressure**
- Volume imbalance
- Directional volume

**Order-flow confirmation**
- Trade-size Herfindahl-Hirschman Index (HHI)
- Maximum ten-second volume share
- Late-versus-early pressure difference
- Directional persistence
- Price-path efficiency
- Close-to-VWAP displacement
- Buy-versus-sell VWAP difference

**Prior market context**
- Previous five-minute volume imbalance
- Previous five-minute return
- Relative five-minute volume
- Previous five-minute return volatility

Redundant or closely related features were excluded from the primary model to limit unnecessary feature duplication.

## 4. Target Construction

The primary prediction horizon is five minutes, with one- and fifteen-minute horizons retained for secondary exploration.

Order-flow direction is defined as the sign of minute-level volume imbalance:

- +1 indicates buyer-dominant pressure.
- -1 indicates seller-dominant pressure.

The aligned forward-return target is:

`aligned_forward_return = orderflow_direction × forward_log_return`

Positive values represent continuation in the direction of current order flow; negative values represent movement against it.

Forward returns are calculated using future minute closes within contiguous trading segments. Targets do not cross session gaps.

## 5. Exploratory Findings

The average aligned five-minute return across the eligible sample was approximately -0.0294 basis points. The overall continuation frequency was 49.43%.

Individual confirmation features exhibited very weak correlations with the aligned target. For example, directional persistence had a Spearman correlation of approximately -0.00005, while price-path efficiency had a correlation of approximately 0.0047.

Quantile analyses and feature-interaction analyses did not reveal consistent, strongly monotonic relationships.

### 5.1 Buyer-Seller Asymmetry

Exploratory directional diagnostics revealed a difference between buyer-dominant and seller-dominant observations.

Buyer-dominant observations:
- Sample size: 10,175
- Mean aligned five-minute return: approximately -0.2565 bps
- Continuation frequency: 47.43%

Seller-dominant observations:
- Sample size: 10,066
- Mean aligned five-minute return: approximately +0.2002 bps
- Continuation frequency: 51.45%

The difference in mean aligned returns was approximately 0.4567 basis points.

This asymmetry was not consistent across all sessions. The result is exploratory and should not be interpreted as evidence of a persistent market effect.

One possible interpretation worth further investigation is an asymmetry in how buying and selling pressure translates into price movement. The idea that prices may rise more gradually but fall more sharply is a potential explanation, not an established finding.

## 6. Model Design and Validation

Three Ridge regression models were compared:

1. **Baseline:** Direction, absolute volume imbalance, and directional volume.
2. **Context:** Baseline features plus four prior-market-context features.
3. **Confirmation:** Context features plus seven order-flow-confirmation features.

All models use Ridge regression with alpha = 1.0. Missing values are median-imputed and features are standardized within a scikit-learn pipeline.

Validation follows an expanding chronological design:
- First ten sessions used for initial training.
- Each of the next five sessions held out individually for testing.
- Training expands to include all prior sessions before each test session.
- No random shuffling is performed.

The primary evaluation metrics are mean absolute error (MAE), pooled out-of-sample R², and directional accuracy.

## 7. Model Results

| Model | Test observations | MAE (bps) | OOS R² | Directional accuracy |
|---|---:|---:|---:|---:|
| Baseline | 6,782 | 5.7222 | -0.0004 | 51.05% |
| Context | 6,782 | 5.7238 | -0.0015 | 50.52% |
| Confirmation | 6,782 | 5.7262 | -0.0024 | 50.43% |

The baseline model produced the best pooled MAE, R², and directional accuracy, although differences between models were small.

Adding prior market context did not improve overall performance. Adding the confirmation features also failed to improve performance over the context model.

The confirmation model's MAE was approximately 0.0416% higher than the context model's MAE. Its pooled out-of-sample R² was also slightly lower, and directional accuracy declined by approximately 0.09 percentage points.

All three models produced negative pooled out-of-sample R², indicating that none outperformed the session-specific training-mean benchmark on squared-error performance.

## 8. Interpretation and Conclusion

The results do not support the original hypothesis in the tested sample.

Although the feature universe captured multiple dimensions of order flow, the additional confirmation features did not improve five-minute price-continuation predictions beyond basic pressure and prior-market context.

The exploratory buyer-seller asymmetry is an interesting observation, but its inconsistency across sessions and the lack of supporting predictive improvements prevent a stronger conclusion.

The findings suggest that the selected minute-level confirmation features, in their current form, do not provide meaningful incremental predictive value for this particular five-minute forecasting task.

Importantly, the result does not establish that order-flow information is generally uninformative. It only indicates that these features and this modeling approach did not demonstrate an improvement in the sample examined.

## 9. Limitations

Several limitations constrain interpretation:

1. **Limited sample:** The study covers one futures contract and 15 trading sessions, with only five out-of-sample test sessions.
2. **Overlapping targets:** Five-minute forward returns overlap across adjacent observations, creating serial dependence in the targets.
3. **Trade-only data:** The dataset does not include the complete order book or participant-level information.
4. **Model scope:** The analysis uses Ridge regression with a fixed regularization parameter and does not evaluate an extensive range of nonlinear models.
5. **No inferential significance testing:** The reported differences are descriptive and are not presented as statistically significant effects.
6. **No transaction-cost analysis:** The study does not simulate execution, fees, slippage, or market impact. Predictive performance should not be interpreted as trading profitability.

## 10. Future Research

Potential extensions include:
- Investigating the exploratory asymmetry between buyer- and seller-dominant observations.
- Evaluating alternative prediction horizons and less overlapping target constructions.
- Testing whether confirmation features behave differently under distinct volatility or volume regimes.
- Exploring nonlinear relationships and interactions with appropriate chronological validation.
- Incorporating order-book information if an authorized dataset becomes available.

These extensions should be treated as future research rather than established conclusions of this study.

## Final Conclusion

This project developed a minute-level order-flow feature universe and tested whether additional characteristics of executed trading activity improve short-term price-continuation predictions.

The confirmation features did not improve out-of-sample model performance in the tested sample. While exploratory directional asymmetries provide possible avenues for further investigation, the evidence does not justify claiming a predictive or profitable trading signal.

The primary contribution is the research framework: feature engineering, target construction, chronological validation, comparative modeling, and transparent evaluation of a hypothesis that was not supported by the results.
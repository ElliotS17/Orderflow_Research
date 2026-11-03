# MFT Orderflow Research

## Research Question

When does aggressive orderflow fail to produce a proportional price response, and does this contain predictive information about subsequent NQ futures returns?

## Dataset

- **Provider:** Databento
- **Dataset:** GLBX.MDP3
- **Schema:** Trades
- **Instrument:** NQZ6 (Nasdaq-100 E-mini futures)
- **Coverage:** October 12–30, 2026

## Project Structure

- `data/`: Raw and processed market data
- `notebooks/`: Research notebooks and documentation
- `src/`: Research pipeline and analysis scripts
  - `data/`: Trade dataset construction
  - `features/`: Minute-level and research feature engineering
  - `events/`: Event identification and dataset construction
  - `analysis/`: Outcome analysis, bootstrap inference, robustness checks, and regression
  - `diagnostics/`: Data inspection, validation, and exploratory diagnostics
- `figures/`: Research visualizations
- `results/`: Statistical and modeling outputs

## Current Research Status

- Trade data ingestion and validation completed.
- Original and deduplicated trade datasets constructed.
- One-minute features and research features generated.
- Strong-orderflow, weak-price-response events identified.
- Forward outcomes evaluated at 1-, 5-, and 15-minute horizons.
- Session bootstrap and robustness analyses completed.
- Exploratory OLS regression with session-clustered standard errors completed.

### Preliminary Findings

Initial results show a positive association between weak price response during strong directional orderflow and subsequent flow-aligned returns. This contrasts with the original reversal hypothesis and currently suggests continuation.

These findings are exploratory. Thresholds were selected using the full sample, the dataset covers only 15 trading sessions, and trades-only data cannot directly establish the liquidity mechanism.

### Next Steps

- Strengthen statistical inference with session-level regression bootstrap methods.
- Test alternative specifications and event definitions.
- Develop out-of-sample validation.
- Consolidate the research process and findings in a reproducible notebook.

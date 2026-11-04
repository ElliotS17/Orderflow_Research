# MFT Orderflow Research

## Project Objective

Develop a comprehensive universe of orderflow and market-context features using Nasdaq-100 E-mini futures (NQ) trade data.

The objective is to investigate whether observable buyer and seller activity contains predictive information about subsequent price movements.

The research process prioritizes creative feature engineering, statistical analysis, and independent hypothesis development.

## Dataset

- **Provider:** Databento
- **Dataset:** GLBX.MDP3
- **Schema:** Trades
- **Instrument:** NQZ6
- **Coverage:** October 12–30, 2026
- **Trading Sessions:** 15

The primary research dataset uses deduplicated trade records. Original records are retained for sensitivity analysis.

## Project Structure

- `data/`: Raw and processed market data
- `notebooks/`: Research notebooks and documentation
- `src/`: Active research pipeline
  - `data/`: Trade dataset construction
  - `features/`: Base aggregation and feature engineering
  - `diagnostics/`: Data inspection, validation, and feature diagnostics
- `archive/`: Previous exploratory research, including hypothesis-specific datasets, scripts, and results
- `figures/`: Research visualizations
- `results/`: Current research outputs

## Current Research Status

### Completed
- Databento trade data ingestion.
- Instrument filtering and chronological organization.
- Original and deduplicated trade datasets.
- One-minute feature aggregation.
- Initial exploratory orderflow feature engineering.
- Preservation and archival of previous hypothesis-specific research.

### Current Phase: Feature Universe Development

The active project is focused on developing a broad, hypothesis-independent feature universe.

Candidate feature families include:

- Directional orderflow pressure
- Trade intensity and activity
- Trade-size distributions
- Large-trade concentration
- Directional consistency
- Orderflow persistence and acceleration
- Price impact and price-flow relationships
- Market context, volatility, and time-of-day effects

Features will be documented, validated, and evaluated for redundancy and potential research value.

## Research Methodology

The project follows this sequence:

1. Validate the underlying trade data.
2. Construct base minute-level aggregations.
3. Engineer a comprehensive orderflow feature universe.
4. Document feature definitions and information availability.
5. Diagnose distributions, missingness, outliers, and correlations.
6. Develop research hypotheses based on observed feature behavior.
7. Construct hypothesis-specific datasets.
8. Evaluate hypotheses using appropriate statistical methods.
9. Document findings and limitations in a reproducible research deliverable.

## Data Limitations

The dataset contains individual trade records, not full order-book depth.

Consequently, the research cannot directly observe resting liquidity, queue position, order cancellations, or limit-order replenishment.

Conclusions must remain within the scope of observable trade activity and price behavior.

## Previous Research

Previous exploratory work examined the relationship between strong directional orderflow, price response, and subsequent returns.

That research has been archived and is not the starting point for the current feature-universe development phase.

Its findings are exploratory and are not treated as established conclusions.

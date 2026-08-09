# AI-Powered Football Forecasting System

An AI-driven football forecasting system that generates statistical match-outcome probabilities without using bookmaker prices.

## Core Design Principles
1. **Odds-Independence**: Predictions are based on stats, not market odds.
2. **Explainability**: Traceable logic.
3. **Continuous Learning**: Feedback loops.
4. **Forecasting-Only Output**: Match probabilities and model-health information, without recommendations.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Setup Environment Variables:
   Copy `.env.example` to `.env` and fill in keys.
   Local execution stores model artifacts under `src/ml/models` unless `MODELS_DIR` is set to another path. Docker images and Compose services set `MODELS_DIR=/app/data/models` explicitly. See `.env.example` for all configurable variables.

## Architecture

- **Data Layer**: Football-Data.co.uk (historical), Odds API (fixtures).
- **Feature Engineering**: Rolling stats, form indicators, differentials.
- **Models**: Poisson (goals), Negative Binomial (corners), Zero-Inflated Poisson (cards).
- **Forecast Surface**: Match probabilities with drift and model-health status.

## Daily Forecasting Operational Guide

Follow this sequence daily to ensure system integrity and generate match-probability forecasts.

### 1. Data Refresh (Morning)
Pull latest fixtures and recent results to keep the feature engine current.
```bash
# Fetch upcoming fixtures (Top 5 Leagues)
python -m src.cli fetch-upcoming --league PL
python -m src.cli fetch-upcoming --league PD
python -m src.cli fetch-upcoming --league SA
python -m src.cli fetch-upcoming --league BL1
python -m src.cli fetch-upcoming --league FL1

# Ingest latest resolved results (Labels)
python -m src.cli fetch-latest-season
python -m src.cli ingest-results
```

### 2. Integrity & Drift Guard (Safety First)
Before generating forecasts, verify that the statistical engine is stable.
```bash
# Calculate rolling drift and update safety status
python -m src.cli refresh-drift
```
> [!IMPORTANT]
> If `refresh-drift` returns **[FAIL]**, treat forecasts as unavailable until the degraded model state has been reviewed.

### 3. Prediction Generation
Generate predictions with the market-agnostic forecasting engine.
```bash
# Generate today's predictions
python -m src.cli show-predictions --league PL

# Show all available predictions
python -m src.cli show-predictions --league PL --all
```

### 4. Reconciliation (Post-Match)
Compare predictions with actual outcomes to feed the evaluation log.
```bash
# Reconcile for a specific date
python -m src.cli reconcile --date-str 2026-01-10
```
---

## 🛠️ System Architecture (3.0 Era)
- **Engine Status**: ❄️ **DEV-FROZEN** (Development baseline; not production-certified)
- **Forecasting Surface**: Market-agnostic match probabilities.
- **Safety Floor**: Absolute probability gate (0.60 - 0.70 depending on market).
- **Momentum Divergence Gate**: Dynamic risk adjustments based on league standings and recent form (Trap Detection & Surging Underdog modifiers).
---
*Production audit signoff is withdrawn pending remediation and deployment verification.*

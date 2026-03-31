# AI-Powered Football Betting Assistant

A production-grade, AI-driven football betting system that generates data-driven predictions without relying on bookmaker odds.

## Core Design Principles
1. **Odds-Independence**: Predictions are based on stats, not market odds.
2. **Explainability**: Traceable logic.
3. **Continuous Learning**: Feedback loops.
4. **Risk Management**: "Forbidden Fruit" slips with controlled risk.

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

## Architecture

- **Data Layer**: Football-Data.co.uk (historical), Odds API (fixtures).
- **Feature Engineering**: Rolling stats, form indicators, differentials.
- **Models**: Poisson (goals), Negative Binomial (corners), Zero-Inflated Poisson (cards).
- **Strategy**: Forbidden Fruit 3.0 with CSS-based slip construction.

## 🍎 Forbidden Fruit 3.0: Daily Operational Guide

Follow this sequence daily to ensure system integrity and generate certified prediction slips.

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
Before generating any slips, verify that the statistical engine is stable.
```bash
# Calculate rolling drift and update safety status
python -m src.cli refresh-drift
```
> [!IMPORTANT]
> If `refresh-drift` returns **[FAIL]**, the `show-predictions` command will warn you. Do not place bets during degraded model state.

### 3. Prediction Generation
Generate predictions with the market-agnostic decision engine.
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
- **Engine Status**: ❄️ **DEV-FROZEN** (Logic is sealed and certified)
- **Primary Strategy**: Market-Agnostic Competitive Selection.
- **Risk Control**: Cross-Market Dissonance Penalty (10% softening on conflicting signals).
- **Safety Floor**: Absolute probability gate (0.60 - 0.70 depending on market).
- **Momentum Divergence Gate**: Dynamic risk adjustments based on league standings and recent form (Trap Detection & Surging Underdog modifiers).
---
*Certified for Production by Antigravity on Jan 10, 2026.*

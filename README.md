# AI-Powered Football Betting Assistant

The new goals-only canonical loop is implemented but **not activated**. Its fixed
Model E replay passed; incomplete historical provider identities and the prior
history bridge still block adoption. The legacy nightly task stays disabled.
See [the canonical loop guide](docs/canonical-loop.md) for read-only inspection,
activation gates, backups and current limitations. The workflows below describe
the existing legacy and historical routes.

A personal-PC betting-prediction system with recorded, replayable prospective evidence.

## Local prospective operator workflow

Use the dedicated local command to inspect the existing Phase 19 predictions and
their scientific state:

```powershell
.\venv\Scripts\python.exe scripts\local_operator.py status
.\venv\Scripts\python.exe scripts\local_operator.py predictions --league BL1
.\venv\Scripts\python.exe scripts\local_operator.py audit
.\venv\Scripts\python.exe scripts\local_operator.py performance
```

The four recorded prediction cohorts are BL1, SA, FL1, and PD. PL has no admitted
prospective prediction. The command shows historical blockers separately and
refuses settlement while Phase 20 protected integrity fails. It does not offer
a new prediction cycle, scheduling, or model promotion. See
[`docs/local-operator-workflow.md`](docs/local-operator-workflow.md) for the
full command guide and current limits.

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
   Set `MODELS_DIR` in your `.env` to override the default model storage location (`src/ml/models`). See `.env.example` for all configurable variables.

## Architecture

- **Data Layer**: Football-Data.co.uk (historical), Odds API (fixtures).
- **Feature Engineering**: Rolling stats, form indicators, differentials.
- **Models**: Poisson (goals), Negative Binomial (corners), Zero-Inflated Poisson (cards).
- **Strategy**: Forbidden Fruit 3.0 with CSS-based slip construction.

## 🍎 Forbidden Fruit 3.0: Legacy CLI Guide

The commands below belong to the general CLI. They do not create Phase 19/20
prospective evidence or replace the local operator workflow above.

The earlier CLI sequence below remains for its separate slip workflow.

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

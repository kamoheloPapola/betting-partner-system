# Betting Partner System - User Guide

## Overview

A professional ML-powered betting prediction system with:
- Statistical models for goals, corners, and cards
- League-specific optimization
- Drift monitoring and alerting
- Production safety controls

---

## Quick Start

### Daily Workflow
```bash
# 1. Get latest results
python -m src.cli fetch-latest-season

# 2. Process results
python -m src.cli ingest-results

# 3. Resolve yesterday's predictions
python -m src.cli resolve-predictions

# 4. Generate today's predictions
python -m src.cli show-predictions

# 5. Review the suggested Forbidden Fruit slip (rendered with predictions)
python -m src.cli show-predictions --all
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `show-predictions` | Display predictions for upcoming matches |
| `show-predictions --all` | Display predictions and render the suggested accumulator |
| `fetch-latest-season` | Download latest match results |
| `ingest-results` | Process and store results |
| `resolve-predictions` | Mark predictions as WIN/LOSS |
| `check-drift` | Monitor model calibration |
| `audit-coverage` | Verify model health per league |
| `freeze-models` | Lock pipeline (production mode) |
| `refresh-offsets` | Recompute team biases |

---

## Model States

| State | Meaning | Operations Allowed |
|-------|---------|-------------------|
| `UNLOCKED` | Development mode | All operations |
| `LOCKED_vX.X` | Production mode | Inference only |

### Locking the System
```bash
python -m src.cli freeze-models --confirm
```

### Checking State
```bash
python -m src.cli freeze-models
# Shows current state and next version
```

---

## Monitoring

### Check for Drift
```bash
python -m src.cli check-drift
```

### View Model Health
```bash
python -m src.cli audit-coverage
```

---

## Configuration

### Environment Variables
```bash
# Slack alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Email alerts (optional)
ALERT_EMAIL_TO=alerts@example.com
ALERT_EMAIL_FROM=betting-system@localhost
SMTP_HOST=localhost
SMTP_PORT=25

# API Keys
ODDS_API_KEY=your-key
FOOTBALL_DATA_ORG_KEY=your-key
```

---

## API Endpoints (When Locked)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model state |
| `/drift-status` | GET | Recent drift alerts |
| `/api/v1/predictions/{league}` | GET | Match predictions |
| `/api/v1/slips/forbidden-fruit` | GET | Generate accumulator |
| `/api/v1/model/info` | GET | Model state info |

Start API:
```bash
uvicorn src.api.main:app --reload
```

---

## Things You Must NEVER Do

> ⚠️ **WARNING**: Violating these rules will destabilize the system.

### 1. Never Retrain While LOCKED
```bash
# This will be BLOCKED automatically:
python -m src.cli train poisson --league PL
# RuntimeError: Training is blocked. Model pipeline is LOCKED.
```

### 2. Never Modify Immune Markets
These markets are protected and must not be tuned:
- `dc_1x`, `dc_x2`, `dc_12` (Double Chance)
- `away_under_1_5`, `home_under_1_5` (High ROI)

### 3. Never Deploy API on Unlocked Model
The API enforces this automatically:
```
RuntimeError: API STARTUP BLOCKED: Model is UNLOCKED.
```

### 4. Never Tune Alpha Without Validation
All alpha changes must pass:
- ΔBrier ≥ 0.005
- Calibration slope ∈ [0.9, 1.1]

### 5. Never Skip Pre-Season Refresh
Before each season:
1. Unlock system
2. Run `refresh-offsets`
3. Validate with backtest
4. Lock system

---

## Troubleshooting

### "Model is UNLOCKED" Error
```bash
python -m src.cli freeze-models --confirm
```

### "Training is blocked" Error
System is locked (correct behavior). Unlock first:
```python
from src.config.model_state import unlock
unlock()
```

### Predictions Seem Off
1. Run `check-drift` to identify issues
2. Run `audit-coverage` to verify health
3. If needed, unlock and retrain

---

## Support

For issues, check:
1. Drift alerts: `data/monitoring/drift_alerts.csv`
2. State audit: `data/.model_state_audit.jsonl`
3. Team offsets: `data/models/team_offsets.csv`

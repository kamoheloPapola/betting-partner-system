---
description: Daily workflow for the betting prediction system
---

# Daily Workflow for Betting Prediction System

## Overview
This workflow runs every match day to generate predictions and track performance.

---

## Morning Routine (Before Matches)

### 1. Fetch Latest Results
```bash
python -m src.cli.main fetch-latest-season
```
Downloads latest match results from Football-Data.co.uk.

### 2. Ingest Results
```bash
python -m src.cli.main ingest-results
```
Processes new results and updates the database.

### 3. Resolve Previous Predictions (Optional)
```bash
python -m src.cli.main resolve-predictions
```
Marks previous predictions as WIN/LOSS/PUSH based on actual outcomes.

### 4. Generate Predictions
```bash
python -m src.cli.main show-predictions
```
Shows predictions for upcoming matches with probabilities and edges.

### 5. Get Forbidden Fruit Selections (Optional)
```bash
python -m src.cli.main forbidden-fruit
```
Generates high-confidence selections using the Forbidden Fruit strategy.

---

## Weekly Maintenance (Once Per Week)

### 1. Check for Model Drift
```bash
python -m src.cli.main check-drift
```
Monitors calibration drift, sharpness decay, and market degradation.

### 2. Refresh Drift Metrics
```bash
python -m src.cli.main refresh-drift --window 30
```
Updates rolling drift statistics.

### 3. Audit Model Coverage
```bash
python -m src.cli.main audit-coverage
```
Verifies all leagues have healthy local models.

---

## Pre-Season Maintenance (Once Per Season)

### 1. Unlock System (If Locked)
```python
from src.config.model_state import unlock
unlock()
```

### 2. Refresh Team Offsets
```bash
python -m src.cli.main refresh-offsets
```
Recomputes team-level corner/card biases.

### 3. Retrain Models (If Needed)
```bash
python -m src.cli.main train poisson --mode production
python -m src.cli.main train nb --mode production
```

### 4. Run Backtest Validation
```bash
python -m src.cli.main backtest --league PL --test-season 2024
python -m src.cli.main backtest --league BL1 --test-season 2024
```

### 5. Lock System
```bash
python -m src.cli.main freeze-models --confirm
```
Locks pipeline to prevent accidental changes during season.

---

## Quick Reference

| Task | Command |
|------|---------|
| Get predictions | `show-predictions` |
| Fetch new data | `fetch-latest-season` |
| Process results | `ingest-results` |
| Resolve outcomes | `resolve-predictions` |
| Check model health | `audit-coverage` |
| Check drift | `check-drift` |
| Lock system | `freeze-models --confirm` |

---

## Emergency: Model Issues

If predictions seem off:
1. Run `check-drift` to identify issues
2. Run `audit-coverage` to verify model health
3. If locked, unlock first: `from src.config.model_state import unlock; unlock()`
4. Retrain affected models: `train poisson --league PL --mode production`
5. Validate with backtest before re-locking

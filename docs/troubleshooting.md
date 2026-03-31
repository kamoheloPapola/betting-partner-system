
# Troubleshooting Guide

## Common Errors

### 1. `ModelNotFoundError`
**Error:** "No production model found for poisson_home_base in PL"
**Cause:** The model registry has no entry for this specific league/model combination.
**Fix:**
- Run training for that league: `python -m src.cli train --league PL --mode production`.
- If new league, ensure data exists (`fetch-latest-season`).

### 2. `DataValidationError`
**Error:** "FutureWarning: Downcasting object dtype arrays..."
**Cause:** Old Pandas versions or deprecated syntax.
**Fix:** Update dependencies or check `prediction.py` for explicit type casting (`pd.to_numeric`). Note: This was patched in v2.1.

### 3. "Drift Guardrail Triggered"
**Error:** Predictions return empty or warning about high drift.
**Cause:** Recent match results deviate significantly from historical averages (e.g. goalfest week).
**Fix:**
- Run `python -m src.cli refresh-drift` to recalibrate the baseline.
- If persistent, verify data integrity (`ingest-results`).

### 4. "No matches found for filter: today"
**Cause:** It is off-season or no fixtures are scheduled.
**Fix:**
- Check `--date` argument.
- Verify `fetch-upcoming` has been run recently.

## Debugging

**Enable Debug Logging:**
Set `LOG_LEVEL=DEBUG` in environment or modify `src/config/settings.py`.

**Run Integration Tests:**
```bash
pytest tests/integration/
```

**Check Event Logs:**
Inspect `data/events/events_YYYY-MM-DD.jsonl` for raw failure events.

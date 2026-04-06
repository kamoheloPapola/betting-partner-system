#!/bin/bash
# Daily automation script — runs on Render Cron at 6am UTC
# Fetches fresh fixtures, then runs daily pipeline for all leagues

set -e
cd /app

echo "[daily] Fetching fresh fixtures..."
python scripts/fetch_fresh_data.py

echo "[daily] Running daily pipeline for all leagues..."
for LEAGUE in PL BL1 FL1 SA PD; do
    echo "[daily] Processing $LEAGUE..."
    python -m src.cli run-daily-pipeline \
        --league "$LEAGUE" \
        --all \
        --no-alert-on-failure \
        || echo "[daily] WARNING: $LEAGUE pipeline failed, continuing..."
done

echo "[daily] Done."

#!/bin/bash
# LEGACY/UNUSED: superseded by Dockerfile.scheduler and deploy/cron/betting-nightly.
# Retained for audit history only. Do not execute.

# Former Render Cron claim: run at 6am UTC.
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

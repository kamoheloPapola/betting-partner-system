#!/bin/bash
# Download trained models from GitHub Releases on Render startup.
# Skip if models already present (warm restart protection).

set -e

MODELS_DIR="/app/models"
MANIFEST_PATH="/app/src/ml/models/manifest.json"
RELEASE_BASE="https://github.com/kamoheloPapola/betting-partner-system/releases/download/v1.0-models"

# Model files - flat list matching what exists in models/
MODEL_FILES=(
    "away_goals_model.pkl"
    "away_goals_xgb_v1.joblib"
    "cards_model.pkl"
    "cards_xgb_v1.joblib"
    "corners_model.pkl"
    "corners_xgb_v1.joblib"
    "goals_model.pkl"
    "home_goals_model.pkl"
    "home_goals_xgb_v1.joblib"
    "match_outcome_model.pkl"
    "feature_columns.json"
    "feature_baselines.json"
    "training_metadata.json"
    "training_report.json"
    "backtest_results.csv"
    "feature_importance.csv"
)

mkdir -p "$MODELS_DIR"

# Skip if already downloaded
if [ -f "$MODELS_DIR/goals_model.pkl" ]; then
    echo "[startup] Models already present - skipping download."
    exit 0
fi

echo "[startup] Downloading models from GitHub Releases (v1.0-models)..."

for FILE in "${MODEL_FILES[@]}"; do
    URL="$RELEASE_BASE/$FILE"
    DEST="$MODELS_DIR/$FILE"
    echo "[startup] Fetching $FILE..."
    curl -L --fail --silent --show-error -o "$DEST" "$URL"
    SIZE=$(du -sh "$DEST" 2>/dev/null | cut -f1)
    echo "[startup] OK: $FILE ($SIZE)"
done

# Also download the pruned manifest (committed to repo, but belt-and-suspenders)
echo "[startup] Verifying manifest..."
if [ ! -f "$MANIFEST_PATH" ]; then
    echo "[startup] WARNING: manifest.json not found at $MANIFEST_PATH"
else
    echo "[startup] Manifest OK."
fi

echo "[startup] All models downloaded successfully."

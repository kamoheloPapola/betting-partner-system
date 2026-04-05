#!/bin/bash
# Download trained models from GitHub Releases on Render startup.
# Skip if models already present (warm restart protection).

set -e

MODELS_DIR="/app/models"
MANIFEST_SRC="/app/src/ml/models/manifest.json"
MANIFEST_DEST="$MODELS_DIR/manifest.json"
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

if [ -f "$MODELS_DIR/goals_model.pkl" ]; then
    echo "[startup] Models already present - skipping download."
else
    echo "[startup] Downloading models from GitHub Releases (v1.0-models)..."

    for FILE in "${MODEL_FILES[@]}"; do
        URL="$RELEASE_BASE/$FILE"
        DEST="$MODELS_DIR/$FILE"
        echo "[startup] Fetching $FILE..."
        curl -L --fail --silent --show-error -o "$DEST" "$URL"
        SIZE=$(du -sh "$DEST" 2>/dev/null | cut -f1)
        echo "[startup] OK: $FILE ($SIZE)"
    done
fi

echo "[startup] Ensuring manifest is available..."
if [ -f "$MANIFEST_SRC" ]; then
    cp "$MANIFEST_SRC" "$MANIFEST_DEST"
    echo "[startup] Manifest synced to $MANIFEST_DEST."
else
    echo "[startup] Repo manifest missing, fetching release asset..."
    curl -L --fail --silent --show-error -o "$MANIFEST_DEST" "$RELEASE_BASE/manifest.json"
    SIZE=$(du -sh "$MANIFEST_DEST" 2>/dev/null | cut -f1)
    echo "[startup] OK: manifest.json ($SIZE)"
fi

echo "[startup] All models ready."

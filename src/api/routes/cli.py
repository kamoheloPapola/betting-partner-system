from __future__ import annotations

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, Body, Depends

from src.api.auth import require_cli_admin
from src.api.cache import MODEL_HEALTH_CACHE_KEY, prediction_cache, prediction_cache_key
from src.config.model_state import get_model_state
from src.ml.registry import ModelRegistry
from src.monitoring.drift_orchestrator import DriftOrchestrator
from src.predictions.predictor import Predictor

router = APIRouter(
    prefix="/cli",
    tags=["cli"],
    dependencies=[Depends(require_cli_admin)],
)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)

if not (PROJECT_ROOT / "scripts").is_dir():
    raise RuntimeError(
        f"PROJECT_ROOT resolved to {PROJECT_ROOT} but scripts/ dir not found. "
        "Check file location - cli.py expects to live at src/api/routes/cli.py"
    )


def _manifest_models(registry: ModelRegistry) -> List[Dict[str, Any]]:
    models = []
    for key, meta in registry.manifest.items():
        if not isinstance(meta, dict) or not meta.get("name"):
            continue
        models.append(
            {
                "key": key,
                "name": meta.get("name"),
                "league": meta.get("league"),
                "version": meta.get("version"),
                "status": meta.get("status"),
            }
        )
    return models


def _parse_predictions(raw_predictions: Any) -> List[Dict[str, Any]]:
    if isinstance(raw_predictions, dict):
        candidate_predictions = raw_predictions.get("predictions", [])
        if isinstance(candidate_predictions, list):
            raw_predictions = candidate_predictions
        else:
            raw_predictions = []

    parsed = []
    for prediction in raw_predictions:
        home_prob = float(prediction.get("home") or prediction.get("home_win") or 0.0)
        draw_prob = float(prediction.get("draw") or 0.0)
        away_prob = float(prediction.get("away") or prediction.get("away_win") or 0.0)
        parsed.append(
            {
                "home_team": prediction.get("home_team", ""),
                "away_team": prediction.get("away_team", ""),
                "home_prob": home_prob,
                "draw_prob": draw_prob,
                "away_prob": away_prob,
            }
        )
    return parsed


@router.get("/help")
def cli_help() -> Dict[str, Any]:
    return {
        "commands": [
            {"name": "help", "description": "Show this help"},
            {"name": "status", "description": "System status overview"},
            {"name": "predict", "description": "Generate predictions. Args: --league PL"},
            {"name": "train", "description": "Retrain models. Args: --leagues PL BL1 ..."},
            {"name": "model-health", "description": "Check model registry and loaded models"},
            {"name": "inspect-drift", "description": "Show current drift state"},
            {"name": "reset-drift", "description": "Clear persisted STOP drift state"},
            {"name": "invalidate-cache", "description": "Clear cached predictions. Args: --league PL"},
            {"name": "cache-stats", "description": "Show active in-memory cache entries"},
            {"name": "list-models", "description": "List all models in registry"},
        ]
    }


@router.get("/status")
def cli_status() -> Dict[str, Any]:
    registry = ModelRegistry()
    models = _manifest_models(registry)
    return {
        "api": "online",
        "model_state": get_model_state(),
        "models_loaded": len(models),
        "active_models": len(registry.manifest.get("active_models", {})),
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/health")
def cli_health() -> Dict[str, str]:
    return {"status": "ok"}


@router.get("/model-health")
def cli_model_health() -> Dict[str, Any]:
    registry = ModelRegistry()
    models = _manifest_models(registry)
    return {
        "model_state": get_model_state(),
        "total_models": len(models),
        "active_models": registry.manifest.get("active_models", {}),
        "shadow_models": registry.manifest.get("shadow_models", {}),
    }


@router.get("/inspect-drift")
def cli_inspect_drift() -> Dict[str, Any]:
    drift = DriftOrchestrator()
    return drift.inspect_global_state()


@router.post("/reset-drift")
def cli_reset_drift() -> Dict[str, Any]:
    drift = DriftOrchestrator()
    cleared = []
    for file_path in [drift.status_file, drift.confidence_state_file]:
        if file_path.exists():
            file_path.unlink()
            cleared.append(str(file_path))

    return {
        "status": "ok",
        "message": "Drift state cleared." if cleared else "No drift state to clear.",
        "cleared": cleared,
    }


@router.post("/invalidate-cache")
def cli_invalidate_cache(
    body: Dict[str, Any] = Body(default_factory=dict),
    league: str | None = None,
) -> Dict[str, Any]:
    league_value = str(body.get("league") or league or "").strip().upper() or None
    cleared_keys: List[str] = []
    existing_keys = prediction_cache.stats().get("cached_keys", [])

    if league_value:
        league_key = prediction_cache_key(league_value)
        prediction_cache.invalidate(league_key)
        prediction_cache.invalidate(MODEL_HEALTH_CACHE_KEY)
        cleared_keys.extend([league_key, MODEL_HEALTH_CACHE_KEY])
        for key in existing_keys:
            if str(key).startswith("slip_"):
                prediction_cache.invalidate(str(key))
                cleared_keys.append(str(key))
    else:
        cleared_keys = existing_keys
        prediction_cache.invalidate()

    try:
        ModelRegistry().reload()
    except Exception:
        pass

    return {
        "status": "ok",
        "message": f"Cache cleared for {league_value or 'all leagues'}",
        "cleared_keys": cleared_keys,
    }


@router.get("/cache-stats")
def cli_cache_stats() -> Dict[str, Any]:
    return prediction_cache.stats()


@router.post("/predict")
def cli_predict(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    predictor = Predictor()
    raw_predictions = predictor.predict_for_show_predictions(
        league=body.get("league"),
        date="upcoming",
        show_all=True,
        timezone="LOCAL",
        simulate=True,
    )
    parsed = _parse_predictions(raw_predictions)
    return {"count": len(parsed), "predictions": parsed}


@router.post("/train")
def cli_train(body: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    leagues_value = body.get("leagues", "PL BL1 FL1 SA PD")
    if isinstance(leagues_value, list):
        leagues = [str(league).strip() for league in leagues_value if str(league).strip()]
    else:
        leagues = [league.strip() for league in str(leagues_value).split() if league.strip()]

    log: List[str] = []
    for league in leagues:
        log.append(f"[INFO] Training {league}...")
        try:
            result = subprocess.run(
                [sys.executable, "scripts/train_batch.py", "--leagues", league],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode == 0:
                log.append(f"[OK]   {league} training complete")
            else:
                error_msg = (result.stderr or result.stdout or "Unknown error")[:500]
                log.append(f"[ERR] {league}: {error_msg}")
                logger.error("Training failed for %s: %s", league, error_msg)
        except Exception as exc:
            log.append(f"[ERR]  {league}: {str(exc)}")

    return {"log": log, "summary": f"Training complete for {len(leagues)} leagues."}


@router.get("/list-models")
def cli_list_models() -> Dict[str, Any]:
    registry = ModelRegistry()
    return {"models": _manifest_models(registry)}

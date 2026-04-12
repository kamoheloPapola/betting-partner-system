"""
FastAPI application entry point.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.api.cache import MODEL_HEALTH_CACHE_KEY, prediction_cache, prediction_cache_key, slip_cache_key
from src.api.schemas import (
    ForbiddenFruitSlipLeg,
    ForbiddenFruitSlipResponse,
    HealthCheck,
    MatchPrediction,
    PredictionTriggerRequest,
    PredictionTriggerResponse,
    TriggerPrediction,
)
from src.api.routes.cli import router as cli_router
from src.api.routes.frontend import router as frontend_router
from src.config import DATA_DIR, MODELS_DIR
from src.config.model_state import get_model_state, is_locked
from src.core.exceptions import ConfigurationError, DataValidationError
from src.ml.model_db import ModelHistoryDB
from src.ml.registry import ModelRegistry
from src.monitoring.drift_orchestrator import DriftOrchestrator
from src.monitoring.telemetry import capture_alert, capture_exception, init_sentry
from src.predictions.predictor import Predictor
from src.strategies.drift_guard import DriftGuardrail
from src.strategies.slip_builder import ForbiddenFruitSlipBuilder

logger = logging.getLogger(__name__)
_LAST_GLOBAL_DRIFT_STATUS: Optional[str] = None
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"
DEFAULT_TRAINING_LEAGUES = ["PL", "BL1", "FL1", "SA", "PD"]
MODEL_CONFIGS = [
    {"name": "poisson_home_base"},
    {"name": "poisson_away_base"},
    {"name": "nb_home_corners_base"},
    {"name": "nb_away_corners_base"},
]

try:
    from sentry_sdk.integrations.fastapi import FastApiIntegration
except Exception:  # pragma: no cover - optional dependency at runtime.
    FastApiIntegration = None

if FastApiIntegration is not None:
    init_sentry(component="prediction-api", integrations=[FastApiIntegration()])
else:
    init_sentry(component="prediction-api")

PROBABILITY_KEYS = (
    "home",
    "draw",
    "away",
    "u25",
    "o25",
    "btts",
    "btts_no",
    "over_1_5",
    "home_under_1_5",
    "away_under_1_5",
    "corn_u11",
    "corn_o75",
    "corn_1x2_h",
    "corn_1x2_d",
    "corn_1x2_a",
    "card_u45",
    "card_o25",
    "card_u55",
    "dc_1x",
    "dc_x2",
    "dc_12",
    "u35",
    "expected_home_goals",
    "expected_away_goals",
    "goal_model_home_lambda",
    "goal_model_away_lambda",
    "ensemble_divergence",
    "divergence_pct",
    "mc_entropy",
    "mc_tail_mass",
    "mc_n_simulations",
)
BOOL_PROBABILITY_KEYS = {"ensemble_divergence"}


def _enforce_locked_state() -> None:
    """Verify model is locked at startup."""
    skip_lock_check = os.getenv("SKIP_MODEL_LOCK_CHECK", "").strip().lower() == "true"
    state = get_model_state()
    if skip_lock_check:
        logger.warning(
            "Skipping model lock check because SKIP_MODEL_LOCK_CHECK=true; current model state is %s",
            state,
        )
        return
    if not state.startswith("LOCKED"):
        raise RuntimeError(
            f"API STARTUP BLOCKED: Model is UNLOCKED ({state}). "
            "API must only run on locked models to prevent inference/training divergence. "
            "Run 'freeze-models --confirm' first."
        )
    logger.info("API starting with locked model: %s", state)


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_valid_float(*values: Any, default: Optional[float] = 0.0) -> Optional[float]:
    for candidate in values:
        parsed = _coerce_float(candidate)
        if parsed is not None:
            return parsed
    return default


def _extract_manifest_brier(meta: Dict[str, Any]) -> Optional[float]:
    metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics"), dict) else {}
    return _first_valid_float(
        metrics.get("brier_score"),
        meta.get("brier_score"),
        default=None,
    )


def _resolve_history_brier(
    model_db: Optional[ModelHistoryDB],
    *,
    model_name: str,
    preferred_leagues: List[str],
    version: str,
) -> Optional[float]:
    if model_db is None:
        return None

    seen: set[str] = set()
    for league in preferred_leagues:
        normalized_league = str(league).strip()
        if not normalized_league or normalized_league in seen:
            continue
        seen.add(normalized_league)

        try:
            events = model_db.fetch_events(
                model_name=model_name,
                league=normalized_league,
                limit=50,
            )
        except Exception:
            continue

        for event in events:
            if version and str(event.get("version", "")) != version:
                continue
            brier = _coerce_float(event.get("brier_score"))
            if brier is not None:
                return brier

        for event in events:
            brier = _coerce_float(event.get("brier_score"))
            if brier is not None:
                return brier

    return None


def _resolve_drift_status(
    drift: Any,
    *,
    market: str,
    global_status: str,
) -> str:
    status_getter = getattr(drift, "get_status", None)
    if callable(status_getter):
        try:
            status = status_getter(market)
        except TypeError:
            status = status_getter()
        if status is not None:
            return str(status)

    market_drift = getattr(drift, "market_status", {})
    if isinstance(market_drift, dict):
        return str(market_drift.get(market, global_status))
    return global_status


def _normalize_guard_status(status: Any) -> str:
    normalized = str(status or "UNKNOWN").strip().upper()
    if normalized == "OK":
        return DriftOrchestrator.GO
    if normalized in {DriftOrchestrator.GO, DriftOrchestrator.WATCH, DriftOrchestrator.STOP}:
        return normalized
    return "UNKNOWN"


def _read_prediction_guard_status() -> str:
    try:
        return _normalize_guard_status(DriftGuardrail().check_drift())
    except Exception as exc:
        logger.warning("Prediction drift guard status unavailable: %s", exc)
        return "UNKNOWN"


def _empty_predictions_message(*, league: str, drift_status: str, total_predictions: int) -> Optional[str]:
    if total_predictions > 0:
        if drift_status == DriftOrchestrator.WATCH:
            return f"Predictions generated under WATCH drift status for {league}. Use caution."
        return None
    if drift_status == DriftOrchestrator.STOP:
        return f"Predictions are currently blocked by the drift guardrail for {league}."
    return f"No predictions available for {league} right now."


def _empty_slip_message(*, league: str, drift_status: str, total_legs: int) -> Optional[str]:
    if total_legs > 0:
        if drift_status == DriftOrchestrator.WATCH:
            return f"Slip built under WATCH drift status for {league}. Review carefully."
        return None
    if drift_status == DriftOrchestrator.STOP:
        return f"Slip generation is currently blocked by the drift guardrail for {league}."
    return f"No qualifying slip is available for {league} right now."


def _serialize_trigger_prediction(prediction: Dict[str, Any]) -> TriggerPrediction:
    home_prob = _first_valid_float(prediction.get("home"), prediction.get("home_win")) or 0.0
    draw_prob = _first_valid_float(prediction.get("draw")) or 0.0
    away_prob = _first_valid_float(prediction.get("away"), prediction.get("away_win")) or 0.0
    btts_prob = _first_valid_float(prediction.get("btts"), prediction.get("btts_yes")) or 0.0
    over_25_prob = _first_valid_float(prediction.get("o25"), prediction.get("over_2_5")) or 0.0
    confidence = _first_valid_float(
        prediction.get("confidence"),
        max(home_prob, draw_prob, away_prob),
    )
    return TriggerPrediction(
        home_team=str(prediction.get("home_team", "")),
        away_team=str(prediction.get("away_team", "")),
        home_win_prob=home_prob,
        draw_prob=draw_prob,
        away_win_prob=away_prob,
        btts_prob=btts_prob,
        over_25_prob=over_25_prob,
        confidence=confidence,
        ensemble_divergence=bool(prediction.get("ensemble_divergence", False)),
    )


def _extract_probabilities(prediction: Dict[str, Any]) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    for key in PROBABILITY_KEYS:
        if key not in prediction:
            continue
        if key in BOOL_PROBABILITY_KEYS:
            value = prediction.get(key)
            extracted[key] = bool(value) if value is not None else None
        else:
            extracted[key] = _coerce_float(prediction.get(key))
    return extracted


def _serialize_prediction(prediction: Dict[str, Any]) -> MatchPrediction:
    kickoff = prediction.get("kickoff_utc") or prediction.get("time") or prediction.get("date")
    return MatchPrediction(
        match_id=str(prediction.get("match_id", "unknown")),
        home_team=str(prediction.get("home_team", "")),
        away_team=str(prediction.get("away_team", "")),
        kickoff=kickoff,
        league=str(prediction.get("league", "")),
        probabilities=_extract_probabilities(prediction),
    )


def _serialize_slip_leg(leg: Dict[str, Any]) -> ForbiddenFruitSlipLeg:
    return ForbiddenFruitSlipLeg(
        match_id=leg.get("id"),
        match=leg.get("match", ""),
        market=leg.get("market") or leg.get("market_name", ""),
        probability=float(leg.get("probability", leg.get("confidence", 0.0))),
        confidence=float(leg.get("confidence", 0.0)),
        league=str(leg.get("league", "")),
        date=leg.get("date"),
        action_tier=leg.get("action_tier"),
        tier=leg.get("tier"),
        odds=_coerce_float(leg.get("odds")),
        implied_probability=_coerce_float(leg.get("implied_probability")),
        edge=_coerce_float(leg.get("edge")),
        ev=_coerce_float(leg.get("ev")),
        passes_value_threshold=leg.get("passes_value_threshold"),
        value_reason=leg.get("value_reason"),
    )


def _raise_environment_mismatch(exc: ConfigurationError) -> None:
    raise HTTPException(
        status_code=503,
        detail=f"Model environment mismatch: {exc}",
    ) from exc


def _load_or_compute_predictions(league: Optional[str]) -> List[Dict[str, Any]]:
    cache_key = prediction_cache_key(league)
    cached_predictions = prediction_cache.get(cache_key)
    if isinstance(cached_predictions, list):
        drift_status = _read_prediction_guard_status()
        if drift_status == DriftOrchestrator.STOP:
            logger.warning(
                "Cache invalidated for league=%s: drift status is STOP.",
                league or "ALL",
            )
            prediction_cache.delete(cache_key)
        else:
            return cached_predictions

    predictor = Predictor()
    raw_predictions = predictor.predict_upcoming(league=league)
    if raw_predictions:
        prediction_cache.set(cache_key, raw_predictions)
    return raw_predictions


def _generate_forbidden_fruit_slip(
    *,
    league: Optional[str],
    min_prob: float,
    max_selections: int,
) -> ForbiddenFruitSlipResponse:
    drift_status = _read_prediction_guard_status()
    cache_key = slip_cache_key(league, min_prob, max_selections)
    cached_slip = prediction_cache.get(cache_key)
    if isinstance(cached_slip, ForbiddenFruitSlipResponse):
        if drift_status == DriftOrchestrator.STOP:
            logger.warning(
                "Slip cache invalidated for league=%s: drift status is STOP.",
                league or "ALL",
            )
            prediction_cache.delete(cache_key)
        else:
            return cached_slip

    league_label = str(league or "ALL").upper()
    if drift_status == DriftOrchestrator.STOP:
        response = ForbiddenFruitSlipResponse(
            generated_at=datetime.now(),
            model_state=get_model_state(),
            slip=[],
            drift_status=drift_status,
            blocked=True,
            message=_empty_slip_message(
                league=league_label,
                drift_status=drift_status,
                total_legs=0,
            ),
        )
        prediction_cache.set(cache_key, response)
        return response

    raw_predictions = _load_or_compute_predictions(league)
    builder = ForbiddenFruitSlipBuilder()
    slip = builder.generate(
        raw_predictions,
        min_probability=min_prob,
        max_selections=max_selections,
    )
    response = ForbiddenFruitSlipResponse(
        generated_at=datetime.now(),
        model_state=get_model_state(),
        slip=[_serialize_slip_leg(leg) for leg in slip],
        drift_status=drift_status,
        blocked=False,
        message=_empty_slip_message(
            league=league_label,
            drift_status=drift_status,
            total_legs=len(slip),
        ),
    )
    prediction_cache.set(cache_key, response)
    return response


def _load_latest_reliability_for_league(league: str) -> List[Dict[str, Any]]:
    """
    Return latest reliability payload per market for the requested league.
    """
    monitoring_dir = DATA_DIR / "monitoring"
    if not monitoring_dir.exists():
        return []

    latest_by_market: Dict[str, Dict[str, Any]] = {}
    league_upper = str(league).upper()

    for path in sorted(monitoring_dir.glob("reliability_*.json")):
        payload: Optional[Dict[str, Any]] = None
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            continue

        if payload is None:
            continue
        payload_league = str(payload.get("league", "")).upper()
        if payload_league != league_upper:
            continue

        market = str(payload.get("market", "unknown"))
        generated_at = str(payload.get("generated_at", ""))
        current = latest_by_market.get(market)
        if current is None or generated_at >= str(current.get("generated_at", "")):
            latest = dict(payload)
            latest["source_file"] = str(Path(path).name)
            latest_by_market[market] = latest

    return [latest_by_market[mkt] for mkt in sorted(latest_by_market)]


def _report_drift_stop_transition(global_drift: Dict[str, Any]) -> None:
    global _LAST_GLOBAL_DRIFT_STATUS

    current_status = str(global_drift.get("status", DriftOrchestrator.STOP))
    previous_status = _LAST_GLOBAL_DRIFT_STATUS
    _LAST_GLOBAL_DRIFT_STATUS = current_status

    if previous_status is None or previous_status == current_status:
        return
    if current_status != DriftOrchestrator.STOP:
        return

    capture_alert(
        event_name="drift_status_stop_transition",
        message="Drift status transitioned to STOP",
        level="error",
        context={
            "previous_status": previous_status,
            "current_status": current_status,
            "evaluated_at": global_drift.get("evaluated_at"),
            "alerts": global_drift.get("alerts"),
        },
    )


_enforce_locked_state()

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def _lifespan(app: FastAPI):
    import asyncio
    import os

    if os.getenv("RENDER"):

        async def _warm() -> None:
            await asyncio.sleep(10)
            leagues = ["PL", "BL1", "FL1", "SA", "PD"]
            for league in leagues:
                try:
                    key = prediction_cache_key(league)
                    if prediction_cache.get(key) is None:
                        predictor = Predictor()
                        data = predictor.predict_upcoming(league=league)
                        prediction_cache.set(key, data)
                        logger.info("[startup] Cache warmed: %s (%d predictions)", league, len(data))
                except Exception as exc:
                    logger.warning("[startup] Cache warm failed for %s: %s", league, exc)
            logger.info("[startup] Cache warm-up complete.")

        asyncio.create_task(_warm())
    yield


app = FastAPI(
    title="Betting Partner API",
    version="2.1.0",
    description="Inference-only API for betting predictions",
    lifespan=_lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/assets", StaticFiles(directory="assets"), name="assets")
app.include_router(frontend_router)
app.include_router(cli_router)


@app.get("/")
def get_root() -> RedirectResponse:
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard")
def get_dashboard() -> FileResponse:
    """Serve the terminal dashboard UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthCheck)
def health_check() -> HealthCheck:
    return HealthCheck(
        status="healthy",
        model_version=get_model_state(),
        last_update=datetime.now(),
    )


@app.get("/api/v1/health")
def api_health_check() -> Dict[str, str]:
    """Lightweight uptime endpoint for orchestrator health checks."""
    return {"status": "ok"}


@app.get("/drift-status")
def get_drift_status() -> Dict[str, Any]:
    """Get current drift monitoring status."""
    from src.config import DATA_DIR
    import pandas as pd

    log_file = DATA_DIR / "monitoring" / "drift_alerts.csv"

    if not log_file.exists():
        return {
            "status": "no_alerts",
            "last_check": None,
            "alerts": [],
        }

    try:
        df = pd.read_csv(log_file)
        recent = df.tail(10).to_dict(orient="records")
        return {
            "status": "monitored",
            "total_alerts": len(df),
            "recent_alerts": recent,
        }
    except Exception as exc:
        capture_exception(exc, context={"endpoint": "/drift-status"})
        return {"status": "error", "error": str(exc)}


def _manifest_model_path(meta: Dict[str, Any]) -> Optional[Path]:
    candidate = meta.get("path") or meta.get("filename")
    if not candidate:
        return None

    path = Path(str(candidate))
    if path.is_absolute():
        return path
    return MODELS_DIR / path


def _serving_league_from_key(serving_key: str, meta: Dict[str, Any]) -> str:
    league = meta.get("league")
    if league not in {None, ""}:
        return str(league)

    suffix = str(serving_key).rsplit("_", 1)[-1]
    return suffix if suffix else "Global"


def _fast_model_health_snapshot() -> Dict[str, Any]:
    cached_snapshot = prediction_cache.get(MODEL_HEALTH_CACHE_KEY)
    if isinstance(cached_snapshot, dict):
        return cached_snapshot

    global_status = "UNKNOWN"
    evaluated_at = None

    manifest_path = ModelRegistry.MANIFEST_FILE
    generated_at = datetime.now().isoformat()
    if not manifest_path.exists():
        result = {
            "generated_at": generated_at,
            "status": "no_manifest",
            "global_drift_status": global_status,
            "global_drift_evaluated_at": evaluated_at,
            "model_count": 0,
            "markets": {},
            "models": [],
            "message": "manifest.json not found",
        }
        prediction_cache.set(MODEL_HEALTH_CACHE_KEY, result, ttl_seconds=300)
        return result

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    active_models = manifest.get("active_models", {}) if isinstance(manifest, dict) else {}
    models: List[Dict[str, Any]] = []
    markets: Dict[str, List[Dict[str, Any]]] = {}

    if isinstance(active_models, dict):
        for serving_key, manifest_key in active_models.items():
            meta = manifest.get(manifest_key) if isinstance(manifest, dict) else None
            if not isinstance(meta, dict):
                continue

            model_path = _manifest_model_path(meta)
            feature_count = meta.get("n_features")
            if feature_count is None and isinstance(meta.get("features"), list):
                feature_count = len(meta["features"])

            status = meta.get("status") or "ok"
            if model_path is not None and not model_path.exists():
                status = "missing_file"

            entry = {
                "key": str(serving_key),
                "manifest_key": str(manifest_key),
                "league": _serving_league_from_key(str(serving_key), meta),
                "model_league": str(meta.get("league") or "Global"),
                "market": str(meta.get("name") or serving_key),
                "version": str(meta.get("version", "unknown")),
                "brier_score": _extract_manifest_brier(meta),
                "drift_status": global_status,
                "last_trained": meta.get("training_date") or meta.get("trained_at") or meta.get("registered_at"),
                "trained_at": meta.get("trained_at") or meta.get("training_date"),
                "registered_at": meta.get("registered_at"),
                "features": feature_count,
                "status": status,
                "path": str(model_path) if model_path is not None else None,
            }
            models.append(entry)
            markets.setdefault(entry["market"], []).append(
                {
                    "league": entry["league"],
                    "model_league": entry["model_league"],
                    "version": entry["version"],
                    "brier_score": entry["brier_score"],
                    "drift_status": entry["drift_status"],
                    "status": entry["status"],
                }
            )

    for entries in markets.values():
        entries.sort(key=lambda item: (str(item.get("league") or ""), str(item.get("model_league") or "")))

    result = {
        "generated_at": generated_at,
        "status": "ok",
        "global_drift_status": global_status,
        "global_drift_evaluated_at": evaluated_at,
        "model_count": len(models),
        "markets": dict(sorted(markets.items())),
        "models": models,
    }
    prediction_cache.set(MODEL_HEALTH_CACHE_KEY, result, ttl_seconds=300)
    return result


@app.get("/model-health")
@app.get("/api/v1/model-health")
def get_model_health() -> Dict[str, Any]:
    """Fast productive model snapshot backed by the manifest file."""
    try:
        return _fast_model_health_snapshot()
    except Exception as exc:
        capture_exception(exc, context={"endpoint": "/model-health"})
        return {
            "generated_at": datetime.now().isoformat(),
            "status": "error",
            "message": str(exc),
            "global_drift_status": "UNKNOWN",
            "global_drift_evaluated_at": None,
            "model_count": 0,
            "markets": {},
            "models": [],
        }


@app.get("/api/v1/predictions/{league}", response_model=List[MatchPrediction])
def get_predictions(league: str, limit: Optional[int] = 20) -> List[MatchPrediction]:
    if limit is not None and limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    try:
        raw_predictions = _load_or_compute_predictions(league)
        if limit is not None:
            raw_predictions = raw_predictions[:limit]
        return [_serialize_prediction(prediction) for prediction in raw_predictions]
    except ConfigurationError as exc:
        logger.error("Prediction environment mismatch for %s: %s", league, exc)
        _raise_environment_mismatch(exc)
    except DataValidationError as exc:
        logger.warning("Prediction request rejected for %s: %s", league, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Prediction error for %s: %s", league, exc)
        capture_exception(exc, context={"endpoint": "/api/v1/predictions/{league}", "league": league})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/v1/predictions/trigger", response_model=PredictionTriggerResponse)
@limiter.limit("10/minute")
def trigger_predictions(request: Request, payload: PredictionTriggerRequest) -> PredictionTriggerResponse:
    if payload.limit is not None and payload.limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    try:
        drift_status = _read_prediction_guard_status()
        if drift_status == DriftOrchestrator.STOP:
            return PredictionTriggerResponse(
                generated_at=datetime.now(),
                league=str(payload.league).upper(),
                total_predictions=0,
                total=0,
                predictions=[],
                drift_status=drift_status,
                blocked=True,
                message=_empty_predictions_message(
                    league=str(payload.league).upper(),
                    drift_status=drift_status,
                    total_predictions=0,
                ),
                reason="drift_guardrail_stop",
            )

        predictor = Predictor()
        raw_predictions = predictor.predict_for_show_predictions(
            league=payload.league,
            date="upcoming",
            show_all=True,
            timezone="LOCAL",
            simulate=True,
            limit=payload.limit,
        )
        empty_reason: Optional[str] = None
        empty_message: Optional[str] = None
        total: Optional[int] = None
        if isinstance(raw_predictions, dict):
            empty_reason = str(raw_predictions.get("reason", "")).strip() or None
            empty_message = str(raw_predictions.get("message", "")).strip() or None
            total_value = raw_predictions.get("total")
            if isinstance(total_value, int):
                total = total_value
            candidate_predictions = raw_predictions.get("predictions", [])
            raw_predictions = candidate_predictions if isinstance(candidate_predictions, list) else []

        serialized = [_serialize_trigger_prediction(prediction) for prediction in raw_predictions]
        effective_total = total if total is not None else len(serialized)
        message = _empty_predictions_message(
            league=str(payload.league).upper(),
            drift_status=drift_status,
            total_predictions=effective_total,
        )
        if effective_total == 0 and empty_message:
            message = empty_message

        return PredictionTriggerResponse(
            generated_at=datetime.now(),
            league=str(payload.league).upper(),
            total_predictions=len(serialized),
            total=effective_total,
            predictions=serialized,
            drift_status=drift_status,
            blocked=False,
            message=message,
            reason=empty_reason,
        )
    except ConfigurationError as exc:
        logger.error("Prediction trigger environment mismatch for %s: %s", payload.league, exc)
        _raise_environment_mismatch(exc)
    except DataValidationError as exc:
        logger.warning("Prediction trigger request rejected for %s: %s", payload.league, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Prediction trigger error for %s: %s", payload.league, exc)
        capture_exception(
            exc,
            context={"endpoint": "/api/v1/predictions/trigger", "league": payload.league},
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get(
    "/api/v1/slips/forbidden-fruit",
    response_model=ForbiddenFruitSlipResponse,
)
def get_forbidden_fruit_slip(
    min_prob: float = 0.65,
    max_selections: int = 4,
) -> ForbiddenFruitSlipResponse:
    if not 0.0 <= min_prob <= 1.0:
        raise HTTPException(status_code=400, detail="min_prob must be between 0 and 1")
    if max_selections < 2:
        raise HTTPException(status_code=400, detail="max_selections must be >= 2")

    try:
        return _generate_forbidden_fruit_slip(
            league=None,
            min_prob=min_prob,
            max_selections=max_selections,
        )
    except ConfigurationError as exc:
        logger.error("Forbidden Fruit environment mismatch: %s", exc)
        _raise_environment_mismatch(exc)
    except DataValidationError as exc:
        logger.warning("Forbidden Fruit request rejected: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Forbidden Fruit error: %s", exc)
        capture_exception(exc, context={"endpoint": "/api/v1/slips/forbidden-fruit"})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/slips/{league}", response_model=ForbiddenFruitSlipResponse)
@limiter.limit("30/minute")
def get_latest_slips(
    request: Request,
    league: str,
    min_prob: float = 0.65,
    max_selections: int = 4,
) -> ForbiddenFruitSlipResponse:
    if not 0.0 <= min_prob <= 1.0:
        raise HTTPException(status_code=400, detail="min_prob must be between 0 and 1")
    if max_selections < 2:
        raise HTTPException(status_code=400, detail="max_selections must be >= 2")

    try:
        return _generate_forbidden_fruit_slip(
            league=league,
            min_prob=min_prob,
            max_selections=max_selections,
        )
    except ConfigurationError as exc:
        logger.error("League slip environment mismatch for %s: %s", league, exc)
        _raise_environment_mismatch(exc)
    except DataValidationError as exc:
        logger.warning("League slip request rejected for %s: %s", league, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("League slip error for %s: %s", league, exc)
        capture_exception(exc, context={"endpoint": "/api/v1/slips/{league}", "league": league})
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v1/model/info")
def get_model_info() -> Dict[str, Any]:
    from src.config.immune_markets import IMMUNE_MARKETS
    from src.config.model_state import NEXT_LOCK_VERSION

    return {
        "state": get_model_state(),
        "is_locked": is_locked(),
        "next_version": NEXT_LOCK_VERSION,
        "immune_markets": list(IMMUNE_MARKETS),
    }


@app.get("/api/v1/calibration/{league}")
def get_calibration_data(league: str) -> Dict[str, Any]:
    """
    Return latest reliability diagram payloads per market for a league.
    """
    latest = _load_latest_reliability_for_league(league)
    if not latest:
        return {
            "league": league.upper(),
            "status": "NO_DATA",
            "reliability": [],
        }

    return {
        "league": league.upper(),
        "status": "OK",
        "reliability": latest,
    }

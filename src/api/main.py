"""
FastAPI application entry point.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ForbiddenFruitSlipLeg,
    ForbiddenFruitSlipResponse,
    HealthCheck,
    MatchPrediction,
    PredictionTriggerRequest,
    PredictionTriggerResponse,
)
from src.config import DATA_DIR, DEFAULT_TRAINING_LEAGUES
from src.config.model_state import get_model_state, is_locked
from src.core.exceptions import ConfigurationError, DataValidationError
from src.ml.registry import ModelRegistry
from src.ml.training.model_configs import MODEL_CONFIGS
from src.monitoring.drift_orchestrator import DriftOrchestrator
from src.monitoring.telemetry import capture_alert, capture_exception, init_sentry
from src.predictions.predictor import Predictor
from src.strategies.slip_builder import ForbiddenFruitSlipBuilder

logger = logging.getLogger(__name__)
_LAST_GLOBAL_DRIFT_STATUS: Optional[str] = None

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


def _generate_forbidden_fruit_slip(
    *,
    league: Optional[str],
    min_prob: float,
    max_selections: int,
) -> ForbiddenFruitSlipResponse:
    predictor = Predictor()
    raw_predictions = predictor.predict_upcoming(league=league)
    builder = ForbiddenFruitSlipBuilder()
    slip = builder.generate(
        raw_predictions,
        min_probability=min_prob,
        max_selections=max_selections,
    )
    return ForbiddenFruitSlipResponse(
        generated_at=datetime.now(),
        model_state=get_model_state(),
        slip=[_serialize_slip_leg(leg) for leg in slip],
    )


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

app = FastAPI(
    title="Betting Partner API",
    version="2.1.0",
    description="Inference-only API for betting predictions",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/model-health")
@app.get("/api/v1/model-health")
def get_model_health() -> Dict[str, Any]:
    """Current productive model health by market and serving scope."""
    registry = ModelRegistry()
    drift = DriftOrchestrator()
    global_drift = drift.inspect_global_state()
    drift.load_confidence_state()
    _report_drift_stop_transition(global_drift)

    global_status = str(global_drift.get("status", DriftOrchestrator.STOP))
    market_drift = drift.market_status if isinstance(drift.market_status, dict) else {}
    model_names = [
        str(cfg["name"])
        for cfg in MODEL_CONFIGS
        if isinstance(cfg, dict) and cfg.get("name")
    ]
    serving_leagues = list(dict.fromkeys([*DEFAULT_TRAINING_LEAGUES, "Global"]))

    markets: Dict[str, List[Dict[str, Any]]] = {}
    for market in model_names:
        entries: List[Dict[str, Any]] = []
        for serving_league in serving_leagues:
            meta = registry.get_production_model_for_league(serving_league, market)
            if not isinstance(meta, dict):
                continue

            metrics = meta.get("metrics", {}) if isinstance(meta.get("metrics"), dict) else {}
            brier_score = _coerce_float(metrics.get("brier_score") if "brier_score" in metrics else meta.get("brier_score"))
            last_trained = meta.get("trained_at") or meta.get("registered_at")
            entries.append(
                {
                    "league": serving_league,
                    "model_league": str(meta.get("league") or "Global"),
                    "version": str(meta.get("version", "unknown")),
                    "brier_score": brier_score,
                    "drift_status": str(market_drift.get(market, global_status)),
                    "last_trained": last_trained,
                }
            )

        if entries:
            markets[market] = entries

    return {
        "generated_at": datetime.now().isoformat(),
        "global_drift_status": global_status,
        "global_drift_evaluated_at": global_drift.get("evaluated_at"),
        "markets": markets,
    }


@app.get("/api/v1/predictions/{league}", response_model=List[MatchPrediction])
def get_predictions(league: str, limit: Optional[int] = 20) -> List[MatchPrediction]:
    if limit is not None and limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    try:
        predictor = Predictor()
        raw_predictions = predictor.predict_upcoming(league=league, limit=limit)
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
def trigger_predictions(request: PredictionTriggerRequest) -> PredictionTriggerResponse:
    if request.limit is not None and request.limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    try:
        predictor = Predictor()
        raw_predictions = predictor.predict_upcoming(league=request.league, limit=request.limit)
        serialized = [_serialize_prediction(prediction) for prediction in raw_predictions]
        return PredictionTriggerResponse(
            generated_at=datetime.now(),
            league=str(request.league).upper(),
            total_predictions=len(serialized),
            predictions=serialized,
        )
    except ConfigurationError as exc:
        logger.error("Prediction trigger environment mismatch for %s: %s", request.league, exc)
        _raise_environment_mismatch(exc)
    except DataValidationError as exc:
        logger.warning("Prediction trigger request rejected for %s: %s", request.league, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Prediction trigger error for %s: %s", request.league, exc)
        capture_exception(
            exc,
            context={"endpoint": "/api/v1/predictions/trigger", "league": request.league},
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
def get_latest_slips(
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

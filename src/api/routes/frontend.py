from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from src.api.cache import prediction_cache, prediction_cache_key
from src.core.exceptions import ConfigurationError, DataValidationError
from src.predictions.predictor import Predictor

router = APIRouter(tags=["frontend"])

LEAGUE_NAMES = {
  "PL": "Premier League",
  "BL1": "Bundesliga",
  "FL1": "Ligue 1",
  "SA": "Serie A",
  "PD": "La Liga",
}


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_valid_float(*values: Any, default: Optional[float] = None) -> Optional[float]:
    for candidate in values:
        parsed = _coerce_float(candidate)
        if parsed is not None:
            return parsed
    return default


def _parse_date(value: Any) -> Optional[date]:
    if value in {None, ""}:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _iso_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            return str(value)
    return value


def _confidence_tier(probability: float) -> str:
    if probability >= 0.65:
        return "HIGH"
    if probability >= 0.5:
        return "MEDIUM"
    return "LOW"


def _team_short_name(name: str) -> str:
    parts = str(name).strip().split()
    if not parts:
        return ""
    if len(parts) <= 2:
        return parts[0]
    return " ".join(parts[:2])


def _form_from_rating(rating: Any) -> List[str]:
    parsed = _coerce_float(rating)
    if parsed is None:
        return []

    clamped = max(0.0, min(1.0, parsed))
    wins = round(clamped * 5)
    draws = 1 if 0.35 < clamped < 0.65 else 0
    losses = max(0, 5 - wins - draws)
    return (["W"] * wins + ["D"] * draws + ["L"] * losses)[:5]


def _h2h_from_probabilities(home_prob: float, draw_prob: float, away_prob: float, sample_size: int) -> Dict[str, int]:
    if sample_size <= 0:
        return {"home_wins": 0, "draws": 0, "away_wins": 0}

    raw = {
        "home_wins": home_prob * sample_size,
        "draws": draw_prob * sample_size,
        "away_wins": away_prob * sample_size,
    }
    floors = {key: int(value) for key, value in raw.items()}
    remainder = sample_size - sum(floors.values())
    ranked = sorted(raw.items(), key=lambda item: item[1] - floors[item[0]], reverse=True)
    for key, _value in ranked[:remainder]:
        floors[key] += 1
    return floors


def _normalize_prediction_row(prediction: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = prediction.get("input_snapshot") if isinstance(prediction.get("input_snapshot"), dict) else {}

    home_prob = _first_valid_float(prediction.get("home"), prediction.get("home_win"), default=0.0) or 0.0
    draw_prob = _first_valid_float(prediction.get("draw"), default=0.0) or 0.0
    away_prob = _first_valid_float(prediction.get("away"), prediction.get("away_win"), default=0.0) or 0.0
    btts_prob = _first_valid_float(prediction.get("btts"), prediction.get("btts_yes"))
    over25_prob = _first_valid_float(prediction.get("o25"), prediction.get("over_2_5"))
    corners_prob = max(
        value
        for value in [
            _coerce_float(prediction.get("corn_o75")) or 0.0,
            _coerce_float(prediction.get("corn_u11")) or 0.0,
            _coerce_float(prediction.get("corn_1x2_h")) or 0.0,
            _coerce_float(prediction.get("corn_1x2_d")) or 0.0,
            _coerce_float(prediction.get("corn_1x2_a")) or 0.0,
        ]
    )
    cards_prob = max(
        value
        for value in [
            _coerce_float(prediction.get("card_o25")) or 0.0,
            _coerce_float(prediction.get("card_u45")) or 0.0,
            _coerce_float(prediction.get("card_u55")) or 0.0,
        ]
    )

    expected_home = _first_valid_float(
        prediction.get("expected_home_goals"),
        prediction.get("goal_model_home_lambda"),
    )
    expected_away = _first_valid_float(
        prediction.get("expected_away_goals"),
        prediction.get("goal_model_away_lambda"),
    )
    expected_goals = None
    if expected_home is not None or expected_away is not None:
        expected_goals = float(expected_home or 0.0) + float(expected_away or 0.0)

    confidence = max(home_prob, draw_prob, away_prob, btts_prob or 0.0, over25_prob or 0.0, corners_prob, cards_prob)
    h2h_count = int(_first_valid_float(snapshot.get("h2h_match_count"), prediction.get("h2h_match_count"), default=0) or 0)
    h2h = _h2h_from_probabilities(home_prob, draw_prob, away_prob, min(5, h2h_count) if h2h_count else 0)
    kickoff = prediction.get("kickoff_utc") or prediction.get("time") or snapshot.get("kickoff_utc") or snapshot.get("date")
    league_code = str(prediction.get("league", "")).upper()

    return {
        "id": str(prediction.get("match_id", "unknown")),
        "league": league_code,
        "league_name": LEAGUE_NAMES.get(league_code, league_code),
        "date": _iso_value(kickoff),
        "kickoff": _iso_value(kickoff),
        "home_team": str(prediction.get("home_team", "")),
        "away_team": str(prediction.get("away_team", "")),
        "home_team_short": _team_short_name(str(prediction.get("home_team", ""))),
        "away_team_short": _team_short_name(str(prediction.get("away_team", ""))),
        "home_prob": home_prob,
        "draw_prob": draw_prob,
        "away_prob": away_prob,
        "expected_goals": expected_goals,
        "expected_home_goals": expected_home,
        "expected_away_goals": expected_away,
        "btts_prob": btts_prob,
        "over25_prob": over25_prob,
        "corners_mean": _first_valid_float(snapshot.get("h2h_avg_corners"), snapshot.get("total_corners")),
        "corners_prob": corners_prob,
        "cards_prob": cards_prob,
        "home_form": _form_from_rating(snapshot.get("home_form_rating")),
        "away_form": _form_from_rating(snapshot.get("away_form_rating")),
        "h2h": h2h,
        "confidence": confidence,
        "confidence_tier": _confidence_tier(confidence),
    }


def _market_probability(prediction: Dict[str, Any], market: Optional[str]) -> float:
    if market == "1X2":
        return max(
            float(prediction.get("home_prob", 0.0)),
            float(prediction.get("draw_prob", 0.0)),
            float(prediction.get("away_prob", 0.0)),
        )
    if market == "btts":
        return float(prediction.get("btts_prob") or 0.0)
    if market == "over25":
        return float(prediction.get("over25_prob") or 0.0)
    if market == "corners":
        return float(prediction.get("corners_prob") or 0.0)
    if market == "cards":
        return float(prediction.get("cards_prob") or 0.0)
    return float(prediction.get("confidence") or 0.0)


def _load_cached_predictions(league: Optional[str]) -> tuple[List[Dict[str, Any]], bool, str]:
    cache_key = prediction_cache_key(league)
    cached_rows = prediction_cache.get(cache_key)
    cache_hit = cached_rows is not None

    if isinstance(cached_rows, list):
        return cached_rows, cache_hit, cache_key

    predictor = Predictor()
    rows = predictor.predict_upcoming(league=league)
    if rows:
        prediction_cache.set(cache_key, rows)
    return rows, cache_hit, cache_key


@router.get("/predictions")
def frontend_predictions(
    league: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    market: Optional[str] = None,
    min_confidence: Optional[float] = None,
    limit: Optional[int] = 50,
) -> Dict[str, Any]:
    if limit is not None and limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")

    from_date = _parse_date(date_from)
    to_date = _parse_date(date_to)
    threshold = None
    if min_confidence is not None:
        threshold = min_confidence / 100 if min_confidence > 1 else min_confidence

    try:
        rows, cache_hit, cache_key = _load_cached_predictions(league)
        normalized = [_normalize_prediction_row(row) for row in rows]
        filtered = []
        for prediction in normalized:
            kickoff_date = _parse_date(prediction.get("kickoff"))
            if from_date and kickoff_date and kickoff_date < from_date:
                continue
            if to_date and kickoff_date and kickoff_date > to_date:
                continue
            if market and _market_probability(prediction, market) <= 0:
                continue
            if threshold is not None and _market_probability(prediction, market) < threshold:
                continue
            filtered.append(prediction)

        filtered.sort(key=lambda item: str(item.get("kickoff") or ""))
        limited = filtered[:limit] if limit is not None else filtered
        return {
            "generated_at": datetime.now().isoformat(),
            "total": len(filtered),
            "count": len(limited),
            "predictions": limited,
            "cached": cache_hit,
            "cache_key": cache_key,
        }
    except ConfigurationError as exc:
        raise HTTPException(status_code=503, detail=f"Model environment mismatch: {exc}") from exc
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

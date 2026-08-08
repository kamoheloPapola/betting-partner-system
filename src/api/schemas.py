"""
API response schemas aligned to the live prediction payloads.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel


class MatchPrediction(BaseModel):
    """
    API contract for a single predicted match.

    `probabilities` maps directly to the prediction adapter output keys
    (`home`, `draw`, `away`, `o25`, `u25`, and optional derived markets).
    """

    match_id: str
    home_team: str
    away_team: str
    kickoff: datetime
    league: str
    probabilities: Dict[str, Optional[float | bool]]


class PredictionTriggerRequest(BaseModel):
    """Request payload for manual prediction triggering."""

    league: str
    limit: Optional[int] = None


class TriggerPrediction(BaseModel):
    """Flat prediction payload returned by manual trigger endpoint."""

    home_team: str
    away_team: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    btts_prob: float
    over_25_prob: float
    confidence: float
    ensemble_divergence: bool


class PredictionTriggerResponse(BaseModel):
    """Manual prediction trigger response payload."""

    generated_at: datetime
    league: str
    total_predictions: int
    total: Optional[int] = None
    predictions: List[TriggerPrediction]
    drift_status: Optional[str] = None
    blocked: bool = False
    message: Optional[str] = None
    reason: Optional[str] = None


class HealthCheck(BaseModel):
    status: str
    model_version: str
    last_update: datetime

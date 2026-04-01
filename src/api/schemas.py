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


class PredictionTriggerResponse(BaseModel):
    """Manual prediction trigger response payload."""

    generated_at: datetime
    league: str
    total_predictions: int
    predictions: List[MatchPrediction]


class ForbiddenFruitSlipLeg(BaseModel):
    """Single leg returned by the slip builder."""

    match_id: Optional[str] = None
    match: str
    market: str
    probability: float
    confidence: float
    league: str
    date: Optional[datetime] = None
    action_tier: Optional[str] = None
    tier: Optional[int] = None
    odds: Optional[float] = None
    implied_probability: Optional[float] = None
    edge: Optional[float] = None
    ev: Optional[float] = None
    passes_value_threshold: Optional[bool] = None
    value_reason: Optional[str] = None


class ForbiddenFruitSlipResponse(BaseModel):
    """Slip-building response returned by the API."""

    generated_at: datetime
    model_state: str
    slip: List[ForbiddenFruitSlipLeg]


class HealthCheck(BaseModel):
    status: str
    model_version: str
    last_update: datetime

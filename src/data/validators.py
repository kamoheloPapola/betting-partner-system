"""
Pydantic Data Models.

Defines validation schemas for raw API data and processed match records.
Used throughout the ingestion and feature engineering pipelines.

Usage:
    from src.data.validators import RawMatch, ProcessedMatch
    
    match = ProcessedMatch(
        match_id="abc123",
        date=datetime.now(),
        home_team="Arsenal",
        away_team="Chelsea",
        ...
    )
"""
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Define public API
__all__ = ["Team", "Score", "RawMatch", "ProcessedMatch"]


class Team(BaseModel):
    """
    Team entity from API response.
    
    Represents a football team with optional display information.
    """
    id: int
    name: str
    shortName: Optional[str] = None
    tla: Optional[str] = None
    crest: Optional[str] = None


class Score(BaseModel):
    """
    Match score container.
    
    Stores home and away scores as floats to handle fractional xG values.
    """
    home: float
    away: float


class RawMatch(BaseModel):
    """
    Represents a match as it comes from the API (normalized structure).
    
    Contains the raw payload for debugging and reprocessing.
    """
    model_config = ConfigDict(extra='ignore')
    
    source_id: str
    date: datetime
    home_team: str
    away_team: str
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    status: str
    competition: str
    season: int
    raw_data: Dict[str, Any]  # Original full payload

    @field_validator('home_score', 'away_score', mode='before')
    @classmethod
    def handle_none_score(cls, v: Any) -> Optional[float]:
        """Convert score to float, preserving None for unfinished matches."""
        if v is None:
            return None
        return float(v)


class ProcessedMatch(BaseModel):
    """
    Represents a match ready for feature engineering.
    
    All required fields must be present. Used as input to the feature pipeline.
    """
    model_config = ConfigDict(extra='forbid')
    
    match_id: str
    date: datetime
    home_team: str
    away_team: str
    home_score: float
    away_score: float
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    status: str
    competition: str
    season: int
    
    # Metadata
    source: str
    last_updated: datetime = Field(default_factory=datetime.now)

    @field_validator('home_score', 'away_score')
    @classmethod
    def scores_must_be_non_negative(cls, v: float) -> float:
        """Ensure scores are non-negative."""
        if v < 0:
            raise ValueError('Scores must be non-negative')
        return v


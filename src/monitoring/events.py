"""
Prediction Event Logging.

Provides structured event tracking for model predictions, gate rejections,
and model fallbacks. Events are persisted as JSONL files with daily rotation.
"""
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np

from src.config import DATA_DIR

# Define public API
__all__ = ["PredictionEvent", "log_event", "load_events", "get_prediction_metrics"]

logger = logging.getLogger(__name__)

# Events Directory (lazy initialization)
EVENTS_DIR = DATA_DIR / "events"


def _ensure_events_dir() -> None:
    """Ensure the events directory exists (lazy initialization)."""
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PredictionEvent:
    """Structured event for prediction tracking."""
    timestamp: datetime
    event_type: Literal["prediction_generated", "gate_rejected", "model_fallback"]
    league: str
    match_id: str
    model_version: str
    confidence: float
    metadata: Dict[str, Any]

    def to_json(self) -> str:
        """Serialize event to JSON string."""
        data = asdict(self)
        # Handle datetime serialization
        data['timestamp'] = self.timestamp.isoformat()
        return json.dumps(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionEvent':
        """Deserialize from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


def log_event(event: PredictionEvent) -> None:
    """
    Log an event to the daily event log file.
    
    Events are stored in DATA_DIR/events/events_{YYYY-MM-DD}.jsonl
    """
    try:
        _ensure_events_dir()
        filename = f"events_{event.timestamp.strftime('%Y-%m-%d')}.jsonl"
        filepath = EVENTS_DIR / filename
        
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(event.to_json() + "\n")
            
    except Exception as e:
        # Fallback logging if file write fails (should not block main flow)
        logger.error(f"Failed to log monitoring event: {e}")


def load_events(date_str: str) -> List[PredictionEvent]:
    """
    Load events for a specific date (YYYY-MM-DD).
    
    Returns empty list if file not found.
    """
    filename = f"events_{date_str}.jsonl"
    filepath = EVENTS_DIR / filename
    
    events: List[PredictionEvent] = []
    if not filepath.exists():
        return events
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    events.append(PredictionEvent.from_dict(data))
    except Exception as e:
        logger.error(f"Error loading events from {filename}: {e}")
        
    return events


def get_prediction_metrics(date: str) -> Dict[str, float]:
    """
    Aggregate metrics for a given date.
    
    Returns:
        Dict with total_predictions, avg_confidence, fallback_rate.
    """
    events = load_events(date)
    if not events:
        return {
            "total_predictions": 0,
            "avg_confidence": 0.0,
            "fallback_rate": 0.0
        }
        
    pred_events = [e for e in events if e.event_type == "prediction_generated"]
    fallback_events = [e for e in events if e.event_type == "model_fallback"]
    
    count = len(pred_events)
    confidences = [e.confidence for e in pred_events]
    
    return {
        "total_predictions": float(count),
        "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "fallback_rate": len(fallback_events) / len(events) if events else 0.0
    }


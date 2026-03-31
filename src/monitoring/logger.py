"""
Prediction Logger.

Provides structured CSV logging for prediction records with schema enforcement
and daily file rotation.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.config import DATA_DIR

# Define public API
__all__ = ["PredictionLogger"]

logger = logging.getLogger(__name__)


class PredictionLogger:
    """
    Logs prediction records to daily CSV files.
    
    Enforces a standard schema to prevent CSV corruption and handles
    column name variants transparently.
    """
    
    UNIVERSE_COLUMNS: List[str] = [
        'match_id', 'league', 'date', 'home_team', 'away_team', 
        'market', 'probability', 'model_version', 'timestamp'
    ]
    
    def __init__(self) -> None:
        self.base_dir = DATA_DIR / "predictions"
        self._dir_initialized = False
        
    def _ensure_dir(self) -> None:
        """Lazily create the predictions directory."""
        if not self._dir_initialized:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._dir_initialized = True
        
    def log(self, records: List[Dict[str, Any]]) -> None:
        """
        Append prediction records to the daily log.
        
        Args:
            records: List of prediction dictionaries.
        """
        if not records:
            return

        self._ensure_dir()
        
        now = datetime.now(timezone.utc)
        today_str = now.strftime("%Y-%m-%d")
        file_path = self.base_dir / f"predictions_{today_str}.csv"
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Mapping variants (e.g., match_hash -> match_id)
        rename_map = {
            'match_hash': 'match_id',
            'predicted_probability': 'probability',
            'kickoff_date': 'date'
        }
        df = df.rename(columns=rename_map)

        # Ensure timestamp if missing
        if 'timestamp' not in df.columns:
            df['timestamp'] = now.isoformat()
            
        # Ensure all required columns exist
        for col in self.UNIVERSE_COLUMNS:
            if col not in df.columns:
                df[col] = None
                
        # Slice to standard columns only
        df = df[self.UNIVERSE_COLUMNS]
            
        # Append to CSV
        header = not file_path.exists()
        try:
            df.to_csv(file_path, mode='a', header=header, index=False)
            logger.info(f"Logged {len(records)} predictions to {file_path}")
        except Exception as e:
            logger.error(f"Failed to log predictions: {e}")


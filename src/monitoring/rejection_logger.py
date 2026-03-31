"""
Rejection Logger.

Logs every NO_BET decision with reason for later analysis.
Learn more from rejections than from bets.
"""
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# === PATHS ===
REJECTION_LOG_PATH = DATA_DIR / "rejection_log.csv"
REJECTION_SUMMARY_PATH = DATA_DIR / "rejection_summary.json"

# === REJECTION REASONS ===
class RejectionReason:
    SHARPNESS_DISABLED = "SHARPNESS_DISABLED"
    CALIBRATOR_DISABLED = "CALIBRATOR_DISABLED"
    LOW_TIER = "LOW_TIER"
    EV_FAIL = "EV_FAIL"
    DRIFT_FREEZE = "DRIFT_FREEZE"
    TIER_INVERSION = "TIER_INVERSION"
    ZERO_STAKE = "ZERO_STAKE"
    ODDS_SANITY = "ODDS_SANITY"
    EXPOSURE_LIMIT = "EXPOSURE_LIMIT"
    MANUAL_KILL = "MANUAL_KILL"

LOG_COLUMNS = [
    'timestamp',
    'match',
    'market',
    'p_raw',
    'reason',
    'details'
]


class RejectionLogger:
    """
    Logs every rejected betting opportunity.
    
    Helps debug over-restrictive logic and understand filter performance.
    """
    
    def __init__(self):
        self._ensure_log_exists()
        self.daily_counts = {}
    
    def _ensure_log_exists(self):
        if not REJECTION_LOG_PATH.exists():
            REJECTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(REJECTION_LOG_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(LOG_COLUMNS)
    
    def log_rejection(
        self,
        match: str,
        market: str,
        p_raw: float,
        reason: str,
        details: str = ""
    ):
        """Log a rejected opportunity."""
        row = {
            'timestamp': datetime.now().isoformat(),
            'match': match,
            'market': market,
            'p_raw': round(p_raw, 4),
            'reason': reason,
            'details': details
        }
        
        with open(REJECTION_LOG_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writerow(row)
        
        # Track daily counts
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in self.daily_counts:
            self.daily_counts[today] = {}
        
        if reason not in self.daily_counts[today]:
            self.daily_counts[today][reason] = 0
        self.daily_counts[today][reason] += 1
        
        logger.debug(f"REJECTED: {match}:{market} - {reason} ({details})")
    
    def get_daily_summary(self) -> dict:
        """Get today's rejection summary."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.daily_counts.get(today, {})
    
    def save_summary(self):
        """Persist daily summary."""
        with open(REJECTION_SUMMARY_PATH, 'w') as f:
            json.dump(self.daily_counts, f, indent=2)
    
    def get_stats(self) -> dict:
        """Get aggregate rejection stats."""
        import pandas as pd
        
        try:
            df = pd.read_csv(REJECTION_LOG_PATH)
            if len(df) == 0:
                return {'total': 0}
            
            return {
                'total': len(df),
                'by_reason': df['reason'].value_counts().to_dict(),
                'by_market': df['market'].value_counts().to_dict()
            }
        except Exception as e:
            return {'error': str(e)}


# === SINGLETON ===
_rejection_logger: Optional[RejectionLogger] = None


def get_rejection_logger() -> RejectionLogger:
    """Get singleton rejection logger."""
    global _rejection_logger
    if _rejection_logger is None:
        _rejection_logger = RejectionLogger()
    return _rejection_logger

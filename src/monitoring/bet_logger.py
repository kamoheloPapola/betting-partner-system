"""
Bet Logging Module.

Every bet must log for auditability:
- p_raw, p_cal, tier, EV, odds, stake, market, status
"""
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

# === PATHS ===
BET_LOG_PATH = DATA_DIR / "betting_log.csv"
BET_LOG_JSON = DATA_DIR / "betting_log_recent.json"

# === CSV COLUMNS ===
LOG_COLUMNS = [
    'timestamp',
    'bet_id',
    'match',
    'market',
    'p_raw',
    'p_cal',
    'tier',
    'ev',
    'odds',
    'stake',
    'market_status',
    'outcome',      # Filled in after settlement
    'pnl',          # Filled in after settlement
    'notes'
]


class BetLogger:
    """
    Structured bet logging for audit trail.
    
    Every bet is logged with full context for post-hoc analysis.
    """
    
    def __init__(self):
        self._ensure_log_exists()
        self.recent_bets: list = []
    
    def _ensure_log_exists(self):
        """Create CSV with headers if doesn't exist."""
        if not BET_LOG_PATH.exists():
            BET_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(BET_LOG_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(LOG_COLUMNS)
    
    def log_bet(
        self,
        bet_id: str,
        match: str,
        market: str,
        p_raw: float,
        p_cal: float,
        tier: str,
        ev: float,
        odds: float,
        stake: float,
        market_status: str,
        notes: str = ""
    ):
        """Log a bet to CSV and in-memory buffer."""
        row = {
            'timestamp': datetime.now().isoformat(),
            'bet_id': bet_id,
            'match': match,
            'market': market,
            'p_raw': round(p_raw, 4),
            'p_cal': round(p_cal, 4),
            'tier': tier,
            'ev': round(ev, 4),
            'odds': round(odds, 3),
            'stake': round(stake, 2),
            'market_status': market_status,
            'outcome': '',
            'pnl': '',
            'notes': notes
        }
        
        # Append to CSV
        with open(BET_LOG_PATH, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writerow(row)
        
        # Keep in memory for quick access
        self.recent_bets.append(row)
        if len(self.recent_bets) > 100:
            self.recent_bets = self.recent_bets[-100:]
        
        # Also save recent to JSON for quick inspection
        self._save_recent_json()
        
        logger.info(f"BET LOGGED: {match} | {market} | p_cal={p_cal:.3f} | tier={tier} | stake={stake:.2f}")
    
    def record_outcome(self, bet_id: str, outcome: str, pnl: float):
        """Update bet with outcome and P&L."""
        import pandas as pd
        
        try:
            df = pd.read_csv(BET_LOG_PATH)
            mask = df['bet_id'] == bet_id
            
            if mask.sum() > 0:
                df.loc[mask, 'outcome'] = outcome
                df.loc[mask, 'pnl'] = pnl
                df.to_csv(BET_LOG_PATH, index=False)
                logger.info(f"Outcome recorded: {bet_id} -> {outcome} ({pnl:+.2f})")
            else:
                logger.warning(f"Bet {bet_id} not found in log")
                
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
    
    def _save_recent_json(self):
        """Save recent bets to JSON for quick inspection."""
        with open(BET_LOG_JSON, 'w') as f:
            json.dump(self.recent_bets, f, indent=2)
    
    def get_recent(self, n: int = 20) -> list:
        """Get recent bets."""
        return self.recent_bets[-n:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get betting statistics."""
        import pandas as pd
        
        try:
            df = pd.read_csv(BET_LOG_PATH)
            
            if len(df) == 0:
                return {'total_bets': 0}
            
            stats = {
                'total_bets': len(df),
                'by_tier': df['tier'].value_counts().to_dict(),
                'by_market': df['market'].value_counts().to_dict(),
                'avg_stake': df['stake'].mean(),
                'total_staked': df['stake'].sum(),
                'avg_ev': df['ev'].mean(),
            }
            
            # Add outcome stats if available
            settled = df[df['outcome'].notna() & (df['outcome'] != '')]
            if len(settled) > 0:
                wins = (settled['outcome'] == 'WIN').sum()
                stats['settled'] = len(settled)
                stats['win_rate'] = wins / len(settled)
                stats['total_pnl'] = settled['pnl'].sum()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}


# === SINGLETON ===
_bet_logger: Optional[BetLogger] = None


def get_bet_logger() -> BetLogger:
    """Get singleton bet logger."""
    global _bet_logger
    if _bet_logger is None:
        _bet_logger = BetLogger()
    return _bet_logger

"""
Market Stability Score (MSS).

Reflects inherent volatility of different market types.
Ref-sensitive markets (corners, cards) get lower base scores.
"""
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


logger = logging.getLogger(__name__)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))


# Base stability scores by market type
MARKET_BASE_MSS = {
    'goals_u25': 0.90,
    'goals_o25': 0.90,
    'u25': 0.90,
    'o25': 0.90,
    'u35': 0.90,
    'home_under_1_5': 0.70, # New: Team specific markets are more volatile
    'away_under_1_5': 0.70,
    'btts': 0.80,
    'btts_yes': 0.80,
    'btts_no': 0.80,
    'corn_o75': 0.70,
    'corn_u11': 0.70,
    'card_o25': 0.60, # Lowered to penalize card overconfidence
    'card_u55': 0.60,
    'card_u45': 0.60,
}


def calculate_mss(
    market: str,
    referee_known: bool = True,
    team_volatility: float = 0.5
) -> float:
    """
    Calculate Market Stability Score.
    
    Args:
        market: Market identifier (e.g., 'corn_o75', 'card_u55')
        referee_known: Whether referee is known for the match
        team_volatility: Team tactical variance (0-1), higher = more volatile
        
    Returns:
        MSS ∈ [0.5, 1.0]
    """
    # Get base score for market type
    base = MARKET_BASE_MSS.get(market, 0.75)
    
    # Referee unknown penalty for card markets
    if not referee_known and 'card' in market:
        base -= 0.15
    
    # Tempo sensitivity: volatile teams in over markets
    if team_volatility > 0.7:
        if any(x in market for x in ['_o', 'over', 'btts_yes']):
            base -= 0.08
    
    return clamp(base, 0.5, 1.0)


# ---------------------------------------------------------------------------
# Team Volatility — computed from historical goal variance
# ---------------------------------------------------------------------------
_volatility_cache: Optional[Dict[str, float]] = None
_DEFAULT_VOLATILITY = 0.5


def _load_volatility_map() -> Dict[str, float]:
    """
    Compute per-team volatility (std-dev of total goals, normalised 0-1).

    Scans processed match CSVs, computes the standard deviation of total
    goals per team, then min-max normalises across all teams so the most
    consistent team ≈ 0.0 and the most volatile ≈ 1.0.
    """
    from src.config import PROCESSED_DATA_DIR

    matches_dir = PROCESSED_DATA_DIR / "matches"
    if not matches_dir.exists():
        logger.warning("Processed matches dir not found — using default volatility")
        return {}

    frames = []
    for f in matches_dir.glob("*.csv"):
        if f.name.endswith("_upcoming.csv"):
            continue
        try:
            df = pd.read_csv(f, usecols=["home_team", "away_team", "home_score", "away_score"])
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return {}

    all_df = pd.concat(frames, ignore_index=True)
    all_df["total_goals"] = pd.to_numeric(all_df["home_score"], errors="coerce") + \
                            pd.to_numeric(all_df["away_score"], errors="coerce")
    all_df = all_df.dropna(subset=["total_goals"])

    # Collect goal totals per team (home + away combined)
    home = all_df.groupby("home_team")["total_goals"].apply(list)
    away = all_df.groupby("away_team")["total_goals"].apply(list)

    team_goals: Dict[str, list] = {}
    for team, goals in home.items():
        team_goals.setdefault(str(team), []).extend(goals)
    for team, goals in away.items():
        team_goals.setdefault(str(team), []).extend(goals)

    # Compute std-dev per team (need ≥ 2 matches)
    team_std: Dict[str, float] = {}
    for team, goals in team_goals.items():
        if len(goals) >= 2:
            team_std[team] = pd.Series(goals).std()

    if not team_std:
        return {}

    # Min-max normalise to [0, 1]
    min_v = min(team_std.values())
    max_v = max(team_std.values())
    spread = max_v - min_v if max_v > min_v else 1.0

    return {t: clamp((v - min_v) / spread, 0.0, 1.0) for t, v in team_std.items()}


def get_team_volatility(home_team: str, away_team: str) -> float:
    """
    Get combined team volatility score ∈ [0, 1].

    Uses the average of home and away team volatilities computed from
    the historical standard deviation of total goals per match.
    Results are cached after first computation.
    """
    global _volatility_cache
    if _volatility_cache is None:
        _volatility_cache = _load_volatility_map()

    if not _volatility_cache:
        return _DEFAULT_VOLATILITY

    h = _volatility_cache.get(home_team.upper(), _DEFAULT_VOLATILITY)
    a = _volatility_cache.get(away_team.upper(), _DEFAULT_VOLATILITY)
    return (h + a) / 2.0


"""
Model Agreement Score (MAS).

Measures consensus between different prediction sources.
High disagreement = low confidence (one model might be wrong).
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(value, max_val))


def calculate_mas(
    probabilities: List[float]
) -> float:
    """
    Calculate Model Agreement Score using range-weighted standard deviation.
    
    Args:
        probabilities: List of probabilities from different models
            [p_model, p_baseline, p_prior]
            
    Returns:
        MAS ∈ [0.4, 1.0]
    """
    if len(probabilities) < 2:
        return 0.75  # Default when only one model
    
    ps = np.array(probabilities)
    
    # Range-weighted disagreement
    spread = np.max(ps) - np.min(ps)
    std = np.std(ps)
    
    # Weighted combination: spread more important than std
    disagreement = 0.6 * spread + 0.4 * std
    
    return clamp(1 - disagreement, 0.4, 1.0)


# ---------------------------------------------------------------------------
# Per-league baseline probabilities — computed from historical data
# ---------------------------------------------------------------------------
# Hardcoded fallbacks (used when no data is available)
_FALLBACK_BASELINES: Dict[str, float] = {
    'goals_u25': 0.45,
    'goals_o25': 0.55,
    'u25': 0.45,
    'o25': 0.55,
    'btts': 0.50,
    'btts_yes': 0.50,
    'btts_no': 0.50,
    'corn_o75': 0.65,
    'corn_u11': 0.70,
    'card_o25': 0.60,
}

# Cache: {league -> {market -> rate}}
_baseline_cache: Optional[Dict[str, Dict[str, float]]] = None


def _compute_league_baselines() -> Dict[str, Dict[str, float]]:
    """
    Compute actual historical rates per league from processed match CSVs.

    Returns a dict of {league: {market: probability}}.
    """
    from src.config import PROCESSED_DATA_DIR

    matches_dir = PROCESSED_DATA_DIR / "matches"
    if not matches_dir.exists():
        logger.warning("Processed matches dir not found — using fallback baselines")
        return {}

    frames = []
    for f in matches_dir.glob("*.csv"):
        if f.name.endswith("_upcoming.csv"):
            continue
        try:
            df = pd.read_csv(f)
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return {}

    all_df = pd.concat(frames, ignore_index=True)

    # Coerce numeric columns
    for col in ("home_score", "away_score"):
        all_df[col] = pd.to_numeric(all_df[col], errors="coerce")
    all_df = all_df.dropna(subset=["home_score", "away_score"])
    all_df["total_goals"] = all_df["home_score"] + all_df["away_score"]

    result: Dict[str, Dict[str, float]] = {}

    for league, grp in all_df.groupby("league"):
        n = len(grp)
        if n < 10:
            continue

        rates: Dict[str, float] = {}

        # Goals markets
        rates["goals_o25"] = (grp["total_goals"] > 2.5).mean()
        rates["o25"] = rates["goals_o25"]
        rates["goals_u25"] = 1.0 - rates["goals_o25"]
        rates["u25"] = rates["goals_u25"]

        # BTTS
        rates["btts_yes"] = ((grp["home_score"] > 0) & (grp["away_score"] > 0)).mean()
        rates["btts"] = rates["btts_yes"]
        rates["btts_no"] = 1.0 - rates["btts_yes"]

        # Corners (if available)
        if "total_corners" in grp.columns:
            tc = pd.to_numeric(grp["total_corners"], errors="coerce").dropna()
            if len(tc) >= 10:
                rates["corn_o75"] = (tc > 7.5).mean()
                rates["corn_u11"] = (tc < 11.5).mean()

        # Cards (if available)
        if "match_total_cards" in grp.columns:
            mc = pd.to_numeric(grp["match_total_cards"], errors="coerce").dropna()
            if len(mc) >= 10:
                rates["card_o25"] = (mc > 2.5).mean()

        result[str(league)] = rates

    return result


def get_baseline_probability(market: str, league: str) -> float:
    """
    Get historical baseline probability for a market in a given league.

    Loads actual per-league rates from processed match data on first
    call, caches the result. Falls back to hardcoded defaults when
    data is not available.
    """
    global _baseline_cache
    if _baseline_cache is None:
        _baseline_cache = _compute_league_baselines()

    # Try league-specific rate first
    if league in _baseline_cache and market in _baseline_cache[league]:
        return _baseline_cache[league][market]

    # Fallback to hardcoded defaults
    return _FALLBACK_BASELINES.get(market, 0.50)


def get_prior_probability(market: str) -> float:
    """
    Get uninformed prior probability for a market.
    
    Returns appropriate prior based on market structure.
    """
    # Binary markets have 0.5 prior
    return 0.50

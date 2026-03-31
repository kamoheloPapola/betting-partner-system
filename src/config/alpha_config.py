"""
Alpha Configuration for Hyperparameter Tuning.

Defines per-league regularization parameters optimized on Brier score.
All values locked after grid search validation.

Rules:
- Only adopt new alpha if ΔBrier ≥ MIN_DELTA_BRIER
- Calibration slope must be in [0.9, 1.1]
- Default alpha used for leagues without specific tuning
"""
from typing import Dict, Tuple, Final

# Define public API
__all__ = [
    "ALPHA_MAP", 
    "DEFAULT_ALPHA", 
    "MIN_DELTA_BRIER", 
    "CALIBRATION_SLOPE_RANGE",
    "GRID_SEARCH_ALPHAS"
]

# --- LOCKED ALPHA VALUES ---
# Updated via grid search validation only
# Last updated: 2026-01-10
ALPHA_MAP: Dict[str, Dict[str, float]] = {
    # League -> {model_type -> alpha}
    "PL": {
        "poisson_home_base": 0.01,
        "poisson_away_base": 0.01,
        "nb_home_corners_base": 0.05,
        "nb_away_corners_base": 0.05,
        "poisson_total_cards_base": 0.01,
    },
    "BL1": {
        "poisson_home_base": 0.01,
        "poisson_away_base": 0.01,
        "nb_home_corners_base": 0.05,
        "nb_away_corners_base": 0.05,
        "poisson_total_cards_base": 0.01,
    },
    "SA": {
        "poisson_home_base": 0.01,
        "poisson_away_base": 0.01,
        "nb_home_corners_base": 0.05,
        "nb_away_corners_base": 0.05,
        "poisson_total_cards_base": 0.01,
    },
    "PD": {
        "poisson_home_base": 0.01,
        "poisson_away_base": 0.01,
        "nb_home_corners_base": 0.05,
        "nb_away_corners_base": 0.05,
        "poisson_total_cards_base": 0.01,
    },
    "FL1": {
        "poisson_home_base": 0.01,
        "poisson_away_base": 0.01,
        "nb_home_corners_base": 0.05,
        "nb_away_corners_base": 0.05,
        "poisson_total_cards_base": 0.01,
    },
}

# Fallback for untrained leagues
DEFAULT_ALPHA: Final[float] = 0.01

# --- ADOPTION CRITERIA ---
# New alpha only adopted if improvement exceeds this threshold
MIN_DELTA_BRIER: Final[float] = 0.005

# Calibration slope must be in this range for adoption
CALIBRATION_SLOPE_RANGE: Final[Tuple[float, float]] = (0.9, 1.1)

# --- GRID SEARCH PARAMETERS ---
GRID_SEARCH_ALPHAS: Final[Tuple[float, ...]] = (0.001, 0.005, 0.01, 0.05, 0.1)


def get_alpha(league: str, model_type: str) -> float:
    """
    Get the optimal alpha for a league/model combination.
    
    Args:
        league: League code (e.g., 'PL', 'BL1')
        model_type: Model name (e.g., 'poisson_home_base')
        
    Returns:
        Tuned alpha value, or DEFAULT_ALPHA if not configured.
    """
    league_config = ALPHA_MAP.get(league, {})
    return league_config.get(model_type, DEFAULT_ALPHA)

"""
System Thresholds Configuration.

Centralized constants for probability gates, market lines, model promotion
thresholds, and strategy-specific parameters. All betting logic should
reference these values rather than hardcoding numbers.

Usage:
    from src.config.thresholds import Thresholds
    
    if probability > Thresholds.GATE_PROB_GOALS:
        # Selection passes gate
"""
from typing import Final

# Define public API
__all__ = ["Thresholds", "css_risk_band"]


class Thresholds:
    """
    Centralized thresholds and configuration constants for the betting system.
    
    All values are class-level constants. Do not modify at runtime.
    """
    
    # =========================================================================
    # DATA & TRAINING
    # =========================================================================
    XG_MIN_COVERAGE: Final[float] = 0.30          # Phase 6 Enhancement Gate
    MIN_TRAINING_SAMPLES: Final[int] = 1000       # Minimum samples for reliable training
    ENHANCED_XG_SAMPLES: Final[int] = 100         # Minimum samples to trigger enhanced xG
    
    # Model Promotion (from registry)
    PROMOTION_GOALS: Final[int] = 2000
    PROMOTION_CORNERS: Final[int] = 1500
    PROMOTION_CARDS: Final[int] = 1500
    
    # =========================================================================
    # MARKET LINES (Prediction & Display)
    # =========================================================================
    CORNERS_U11_LINE: Final[float] = 11.5         # Standard corners under line
    CORNERS_O9_LINE: Final[float] = 9.5           # Corners over 9.5 line
    CORNERS_O75_LINE: Final[float] = 7.5          # Virtual corners line
    GOALS_O2_LINE: Final[float] = 2.5             # Standard goals over/under
    CARDS_U55_LINE: Final[float] = 5.5            # Cards under 5.5 line
    CARDS_O25_LINE: Final[float] = 2.5            # Cards over 2.5 line
    
    # =========================================================================
    # SELECTION GATE (Primary Filtering)
    # =========================================================================
    GATE_PROB_GOALS: Final[float] = 0.74
    GATE_PROB_O15: Final[float] = 0.76
    GATE_PROB_CORNERS: Final[float] = 0.85
    GATE_PROB_CARDS: Final[float] = 0.70
    GATE_PROB_DC: Final[float] = 0.77
    GATE_PROB_1X2: Final[float] = 0.73
    GATE_PROB_DEFAULT: Final[float] = 0.75
    
    GATE_MIN_EDGE: Final[float] = 0.08            # 8% edge requirement
    MAX_DAILY_SELECTIONS: Final[int] = 5
    
    # =========================================================================
    # BASELINES (Conservative Hurdles)
    # =========================================================================
    BASE_HOME_WIN: Final[float] = 0.38
    BASE_AWAY_WIN: Final[float] = 0.30
    BASE_DRAW: Final[float] = 0.28
    BASE_O25: Final[float] = 0.50
    BASE_U25: Final[float] = 0.50
    BASE_BTTS: Final[float] = 0.52
    BASE_DC: Final[float] = 0.66
    BASE_TEAM_U15: Final[float] = 0.60  # Corrected from 0.75 - prior value exceeded cap (0.80), making 8% edge unreachable
    BASE_CORN_U11: Final[float] = 0.65
    BASE_CARD_O25: Final[float] = 0.55
    BASE_CARD_U55: Final[float] = 0.78
    
    # =========================================================================
    # STRATEGY GATES (Forbidden Fruit 3.0)
    # =========================================================================
    GLOBAL_MIN_CONF: Final[float] = 0.60          # Absolute floor for safety
    CSS_SURVIVABILITY_THRESHOLD: Final[float] = 0.15
    MIN_PROB_GATE: Final[float] = 0.62            # Match-level probability gate
    
    # Market Hard Gates (Strategy specific)
    STRAT_GATE_1X2: Final[float] = 0.70
    STRAT_GATE_CORNERS: Final[float] = 0.70
    STRAT_GATE_CARDS: Final[float] = 0.65
    STRAT_GATE_CARDS_O25: Final[float] = 0.60
    STRAT_GATE_UND_U15: Final[float] = 0.64
    STRAT_GATE_BTTS: Final[float] = 0.70
    STRAT_GATE_GOALS_O15: Final[float] = 0.70
    STRAT_GATE_GOALS_O25: Final[float] = 0.65
    STRAT_GATE_GOALS_U25: Final[float] = 0.65
    STRAT_GATE_DC: Final[float] = 0.78
    STRAT_GATE_CORNERS_O75: Final[float] = 0.70
    STRAT_GATE_CARDS_U55: Final[float] = 0.75
    
    # =========================================================================
    # TIERING & CONFIDENCE
    # =========================================================================
    TIER_1_CONF: Final[float] = 0.75
    TIER_2_CONF: Final[float] = 0.70
    
    # Tempo Thresholds
    TEMPO_O15_STABLE: Final[float] = 0.70
    TEMPO_O15_LOW_RISK: Final[float] = 0.55
    TEMPO_O75_HIGH: Final[float] = 0.65
    TEMPO_O75_SLOW: Final[float] = 0.45
    
    # =========================================================================
    # MOMENTUM DIVERGENCE
    # =========================================================================
    MOMENTUM_STRONG_POSITIVE: Final[float] = 0.60
    MOMENTUM_POSITIVE: Final[float] = 0.20
    MOMENTUM_STRONG_NEGATIVE: Final[float] = -0.60
    MOMENTUM_NEGATIVE: Final[float] = -0.20
    MOMENTUM_UNDERDOG_RELAXATION: Final[float] = 0.06  # Relaxation for surging underdogs
    
    # =========================================================================
    # CSS MATH
    # =========================================================================
    CSS_CORR_DECAY: Final[float] = 0.15
    CSS_TEMPORAL_DECAY: Final[float] = 0.3
    
    # =========================================================================
    # MAINTENANCE & DRIFT
    # =========================================================================
    DRIFT_WARNING_THRESHOLD: Final[float] = 0.15
    
    # Action 2 Implementation: Match Intensity Damping
    # Applied to Cup games or high-volatility fixtures to prevent overconfidence.
    INTENSITY_VARIANCE_MULT: Final[float] = 1.6


def css_risk_band(css: float) -> str:
    """
    Classify CSS score into human-readable risk band.
    
    Args:
        css: Combo Survivability Score (0.0 to 1.0).
        
    Returns:
        Risk classification: 'VERY LOW', 'LOW', 'MEDIUM', or 'HIGH'.
    """
    if css >= 0.45:
        return "VERY LOW"
    if css >= 0.30:
        return "LOW"
    if css >= 0.20:
        return "MEDIUM"
    return "HIGH"

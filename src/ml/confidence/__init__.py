"""
Production Confidence Pipeline.

Separates probability (chance event happens) from confidence (reliability of estimate).
"""
from src.ml.confidence.uncertainty import estimate_uncertainty
from src.ml.confidence.data_reliability import calculate_drs
from src.ml.confidence.market_stability import calculate_mss
from src.ml.confidence.model_agreement import calculate_mas
from src.ml.confidence.calibration_penalty import calculate_hce_penalty
from src.ml.confidence.calculator import ConfidenceCalculator, get_confidence_calculator

__all__ = [
    'estimate_uncertainty',
    'calculate_drs',
    'calculate_mss',
    'calculate_mas',
    'calculate_hce_penalty',
    'ConfidenceCalculator',
    'get_confidence_calculator',
]

# Action thresholds
CONFIDENCE_THRESHOLDS = {
    'AUTO_BET': 0.80,
    'REDUCED_STAKE': 0.70,
    'MANUAL_REVIEW': 0.60,
    'NO_BET': 0.0,
}

# Stake multipliers per confidence tier
CONFIDENCE_STAKE_MULTIPLIERS = {
    'AUTO_BET': 1.0,
    'REDUCED_STAKE': 0.6,
    'MANUAL_REVIEW': 0.3,
    'NO_BET': 0.0,
}


def get_action_tier(confidence: float) -> str:
    """Get action tier for a confidence score."""
    if confidence >= CONFIDENCE_THRESHOLDS['AUTO_BET']:
        return 'AUTO_BET'
    elif confidence >= CONFIDENCE_THRESHOLDS['REDUCED_STAKE']:
        return 'REDUCED_STAKE'
    elif confidence >= CONFIDENCE_THRESHOLDS['MANUAL_REVIEW']:
        return 'MANUAL_REVIEW'
    else:
        return 'NO_BET'

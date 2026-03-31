"""
Confidence Calculator.

Combines all confidence components into final score.
Uses geometric dampening for early-phase stability.
"""
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.ml.confidence.uncertainty import estimate_uncertainty
from src.ml.confidence.data_reliability import calculate_drs
from src.ml.confidence.market_stability import calculate_mss
from src.ml.confidence.model_agreement import calculate_mas, get_baseline_probability, get_prior_probability
from src.ml.confidence.calibration_penalty import calculate_hce_penalty


@dataclass
class ConfidenceResult:
    """Result from confidence calculation."""
    confidence: float
    action_tier: str
    stake_multiplier: float
    components: Dict[str, float]


class ConfidenceCalculator:
    """
    Production-grade confidence calculator.
    
    Confidence = pow(DRS * MSS * MAS * HCE * (1 - uncertainty), 0.85)
    """
    
    GEOMETRIC_DAMPENING = 0.85
    
    def __init__(self):
        pass
    
    def calculate(
        self,
        prob: float,
        market: str,
        league: str,
        # Uncertainty inputs
        mu: float = 0.0,
        variance: float = 0.0,
        # DRS inputs
        h2h_count: int = 0,
        days_since_h2h: float = 0.0,
        missing_features: List[str] = None,
        # MSS inputs
        referee_known: bool = True,
        team_volatility: float = 0.5,
        # MAS inputs
        baseline_prob: float = None,
        # HCE inputs
        days_since_calibration: int = 0
    ) -> ConfidenceResult:
        """
        Calculate confidence score for a prediction.
        
        Args:
            prob: Model probability
            market: Market identifier
            league: League code
            ... (component-specific inputs)
            
        Returns:
            ConfidenceResult with score, action tier, and components
        """
        missing_features = missing_features or []
        
        # === Calculate Components ===
        
        # 1. Uncertainty
        uncertainty = estimate_uncertainty(mu, variance) if mu > 0 else 0.3
        
        # 2. Data Reliability Score
        drs = calculate_drs(h2h_count, days_since_h2h, missing_features)
        
        # 3. Market Stability Score
        mss = calculate_mss(market, referee_known, team_volatility)
        
        # 4. Model Agreement Score
        baseline = baseline_prob if baseline_prob is not None else get_baseline_probability(market, league)
        prior = get_prior_probability(market)
        mas = calculate_mas([prob, baseline, prior])
        
        # 5. Historical Calibration Error
        hce = calculate_hce_penalty(league, market, prob, days_since_calibration)
        
        # === Final Formula ===
        raw_conf = drs * mss * mas * hce * (1 - uncertainty)
        
        # Geometric dampening for early-phase stability
        confidence = pow(raw_conf, self.GEOMETRIC_DAMPENING)
        confidence = max(0.0, min(confidence, 1.0))
        
        # === Determine Action ===
        action_tier = self._get_action_tier(confidence)
        stake_mult = self._get_stake_multiplier(action_tier)
        
        return ConfidenceResult(
            confidence=round(confidence, 4),
            action_tier=action_tier,
            stake_multiplier=stake_mult,
            components={
                'uncertainty': round(uncertainty, 4),
                'drs': round(drs, 4),
                'mss': round(mss, 4),
                'mas': round(mas, 4),
                'hce': round(hce, 4),
                'raw': round(raw_conf, 4),
            }
        )
    
    def _get_action_tier(self, confidence: float) -> str:
        """Get action tier for confidence score."""
        if confidence >= 0.80:
            return 'AUTO_BET'
        elif confidence >= 0.70:
            return 'REDUCED_STAKE'
        elif confidence >= 0.60:
            return 'MANUAL_REVIEW'
        else:
            return 'NO_BET'
    
    def _get_stake_multiplier(self, tier: str) -> float:
        """Get stake multiplier for tier."""
        return {
            'AUTO_BET': 1.0,
            'REDUCED_STAKE': 0.6,
            'MANUAL_REVIEW': 0.3,
            'NO_BET': 0.0,
        }.get(tier, 0.0)


# === Singleton ===
_calculator: Optional[ConfidenceCalculator] = None


def get_confidence_calculator() -> ConfidenceCalculator:
    """Get singleton confidence calculator."""
    global _calculator
    if _calculator is None:
        _calculator = ConfidenceCalculator()
    return _calculator


"""
Derived Markets Engine.

Calculates derived probabilities for markets that are mathematical
functions of primary markets (e.g., Double Chance from 1X2).
"""
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class DoubleChanceEngine:
    """
    Derives Double Chance probabilities with consistency checks.
    
    Principles:
    - 1X = P(Home) + P(Draw)
    - X2 = P(Draw) + P(Away)
    - 12 = P(Home) + P(Away)
    
    Includes safety assertions to ensure inputs are sane.
    """
    
    @staticmethod
    def calculate(
        prob_home: float, 
        prob_draw: float, 
        prob_away: float,
        strict: bool = True
    ) -> Dict[str, float]:
        """
        Derive Double Chance keys.
        
        Args:
            prob_home, prob_draw, prob_away: 1X2 Probabilities (should sum approx to 1.0)
            strict: If True, enforces normalization and bounds.
        """
        # 1. Normalize if requested (or just check sum)
        total = prob_home + prob_draw + prob_away
        if abs(total - 1.0) > 0.05 and strict:
            logger.warning(f"1X2 Probs do not sum to 1.0 (Sum: {total:.3f}). Normalizing for DC calculation.")
            prob_home /= total
            prob_draw /= total
            prob_away /= total
            
        # 2. Derive
        p_1x = prob_home + prob_draw
        p_x2 = prob_draw + prob_away
        p_12 = prob_home + prob_away
        
        # 3. Assertions
        try:
            DoubleChanceEngine._assert_consistency(p_1x, p_x2, p_12, prob_home, prob_draw, prob_away)
        except AssertionError as e:
            logger.error(f"DC Consistency Check Failed: {e}")
            # If strictly broken, maybe raise? For now, log.
            if strict:
                raise e

        return {
            "dc_1x": min(1.0, p_1x),
            "dc_x2": min(1.0, p_x2),
            "dc_12": min(1.0, p_12)
        }

    @staticmethod
    def _assert_consistency(p_1x, p_x2, p_12, ph, pd, pa):
        epsilon = 1e-9
        # P(1X) >= P(H)
        assert p_1x >= ph - epsilon, f"1X ({p_1x:.3f}) < Home ({ph:.3f})"
        assert p_1x >= pd - epsilon, f"1X ({p_1x:.3f}) < Draw ({pd:.3f})"
        
        # P(X2) >= P(A)
        assert p_x2 >= pa - epsilon, f"X2 ({p_x2:.3f}) < Away ({pa:.3f})"
        assert p_x2 >= pd - epsilon, f"X2 ({p_x2:.3f}) < Draw ({pd:.3f})"
        
        # P(12) >= P(H)
        assert p_12 >= ph - epsilon, f"12 ({p_12:.3f}) < Home ({ph:.3f})"
        assert p_12 >= pa - epsilon, f"12 ({p_12:.3f}) < Away ({pa:.3f})"


class BTTSNoEngine:
    """
    Calculates probability of Both Teams To Score: NO.
    
    BTTS_NO = P(Home scores 0) + P(Away scores 0) - P(Both score 0)
    
    Using Poisson: P(X=0) = e^(-lambda)
    """
    
    @staticmethod
    def calculate(
        home_expected_goals: float,
        away_expected_goals: float
    ) -> Dict[str, float]:
        """
        Calculate BTTS NO probability.
        
        Args:
            home_expected_goals: Expected goals for home team (lambda_home)
            away_expected_goals: Expected goals for away team (lambda_away)
            
        Returns:
            Dict with btts_no probability
        """
        import math
        
        # Poisson P(X=0) = e^(-lambda)
        p_home_zero = math.exp(-home_expected_goals)
        p_away_zero = math.exp(-away_expected_goals)
        
        # BTTS NO = At least one team scores 0
        # P(BTTS NO) = P(Home=0 OR Away=0) = P(Home=0) + P(Away=0) - P(Both=0)
        p_both_zero = p_home_zero * p_away_zero
        p_btts_no = p_home_zero + p_away_zero - p_both_zero
        
        # Clamp to valid probability range
        p_btts_no = max(0.0, min(1.0, p_btts_no))
        
        logger.debug(f"BTTS NO: home_lambda={home_expected_goals:.2f}, away_lambda={away_expected_goals:.2f}, P={p_btts_no:.3f}")
        
        return {
            "btts_no": p_btts_no,
            "btts_yes": 1.0 - p_btts_no
        }


class GoalsUnderEngine:
    """
    Calculates probability of total goals being under a threshold.
    
    Uses Poisson distribution for combined goal expectation.
    P(Total <= k) = sum(P(i) for i in 0..k) where P(i) = (lambda^i * e^-lambda) / i!
    """
    
    @staticmethod
    def calculate(
        home_expected_goals: float,
        away_expected_goals: float,
        threshold: float = 3.5
    ) -> Dict[str, float]:
        """
        Calculate probability of total goals under threshold.
        
        Args:
            home_expected_goals: Expected goals for home team
            away_expected_goals: Expected goals for away team
            threshold: Goal threshold (e.g., 3.5 for U3.5)
            
        Returns:
            Dict with under probability
        """
        import math
        
        # Combined lambda for total goals
        total_lambda = home_expected_goals + away_expected_goals
        
        # Calculate P(Total <= floor(threshold)) using Poisson CDF
        k = int(threshold)  # For 3.5, we need P(Total <= 3)
        
        prob_under = 0.0
        for i in range(k + 1):
            # Poisson PMF: P(X=i) = (lambda^i * e^-lambda) / i!
            prob_under += (total_lambda ** i * math.exp(-total_lambda)) / math.factorial(i)
        
        # Clamp to valid probability range
        prob_under = max(0.0, min(1.0, prob_under))
        
        logger.debug(f"Goals U{threshold}: total_lambda={total_lambda:.2f}, P={prob_under:.3f}")
        
        return {
            f"goals_u{threshold}": prob_under,
            f"goals_o{threshold}": 1.0 - prob_under
        }
    
    @staticmethod
    def calculate_u35(home_expected_goals: float, away_expected_goals: float) -> float:
        """Convenience method for U3.5."""
        result = GoalsUnderEngine.calculate(home_expected_goals, away_expected_goals, 3.5)
        return result["goals_u3.5"]
    
    @staticmethod
    def calculate_u25(home_expected_goals: float, away_expected_goals: float) -> float:
        """Convenience method for U2.5."""
        result = GoalsUnderEngine.calculate(home_expected_goals, away_expected_goals, 2.5)
        return result["goals_u2.5"]

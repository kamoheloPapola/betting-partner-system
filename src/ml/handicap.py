import numpy as np
from typing import Literal

class EuropeanHandicap:
    """
    Calculates European Handicap probabilities using joint Poisson distributions.
    
    Definition:
    EH (Handicap) Win: The selection's score + handicap > Opponent's score.
    
    Unlike Asian Handicap, there are no pushes. A draw after handicap is a Loss.
    """
    
    @staticmethod
    def win_probability(
        lambda_home: float,
        lambda_away: float,
        handicap: int,
        underdog: Literal["home", "away"],
        max_goals: int = 10
    ) -> float:
        """
        Calculates P(Underdog + Handicap > Favorite)
        
        Guardrails:
        1. lambda_diff >= 0.6 (Only valid for clear favorites)
        2. Prob <= 0.85 (Reject manufactured certainties)
        """
        # 1. Guardrail: Strong Favorite Only
        # We only offer EH+2 on the 'Underdog', so we expect the OTHER team to be favored.
        # But the method is generic, so we just check the absolute difference.
        if abs(lambda_home - lambda_away) < 0.6:
            return 0.0 # Reject: Match too close for EH+2 value
            
        from scipy.stats import poisson
        
        # 2. Compute Joint Distribution
        h_probs = poisson.pmf(np.arange(max_goals + 1), lambda_home)
        a_probs = poisson.pmf(np.arange(max_goals + 1), lambda_away)
        
        # Outer product P(h, a) = P(h) * P(a)
        matrix = np.outer(h_probs, a_probs)
        
        # 3. Apply Handicap Condition
        prob = 0.0
        rows, cols = matrix.shape
        
        for h in range(rows):
            for a in range(cols):
                if underdog == "home":
                    # Home + Handicap > Away
                    if h + handicap > a:
                        prob += matrix[h, a]
                else:
                    # Away + Handicap > Home
                    if a + handicap > h:
                        prob += matrix[h, a]
                        
        # 4. Guardrail: Probability Ceiling
        if prob > 0.85:
            return 0.0 # Reject: Too certain (model likely lying/overfitting)
            
        return prob

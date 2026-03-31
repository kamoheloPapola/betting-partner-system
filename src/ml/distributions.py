"""
Statistical Distribution Engines.

Converts model outputs (lambda values) into market probabilities using
various distribution models:
- PoissonEngine: Goals (independent Poisson)
- NegativeBinomialEngine: Corners (over-dispersed counts)
- ZeroInflatedEngine: Cards (sparse counts with excess zeros)

All engines include confidence dampening and probability clamping
for calibrated outputs.
"""
import numpy as np
import scipy.stats as stats
from src.config.thresholds import Thresholds
from scipy.stats import poisson
from typing import Dict, Tuple

class PoissonEngine:
    """
    Converts expected goals (lambdas) into market probabilities.
    Ensures consistency: All probabilities flow from the goal distributions.
    """
    
    def __init__(self, max_goals: int = 10, rho: float = -0.13):
        self.max_goals = max_goals # Truncate distribution at 10 goals for performance
        self.rho = rho

    def _dixon_coles_tau(self, home_goals: int, away_goals: int, home_xg: float, away_xg: float) -> float:
        """
        Dixon-Coles low-score correction factor.

        Applies only to:
        - 0-0
        - 1-0
        - 0-1
        - 1-1
        """
        if home_goals == 0 and away_goals == 0:
            return 1.0 - (home_xg * away_xg * self.rho)
        if home_goals == 1 and away_goals == 0:
            return 1.0 + (away_xg * self.rho)
        if home_goals == 0 and away_goals == 1:
            return 1.0 + (home_xg * self.rho)
        if home_goals == 1 and away_goals == 1:
            return 1.0 - self.rho
        return 1.0

    def calculate_score_matrix(self, home_xg: float, away_xg: float) -> np.ndarray:
        """
        Generates a 2D matrix of probabilities for scorelines P(h, a).
        P(Score_Home=h) * P(Score_Away=a) assuming independence.
        Includes Dixon-Coles low-score correction for 0-0, 1-0, 0-1, 1-1.
        """
        home_probs = poisson.pmf(np.arange(self.max_goals + 1), home_xg)
        away_probs = poisson.pmf(np.arange(self.max_goals + 1), away_xg)
        
        # Outer product to get joint probabilities
        matrix = np.outer(home_probs, away_probs)

        # Dixon-Coles correction on low-score cells
        if matrix.shape[0] > 1 and matrix.shape[1] > 1:
            matrix[0, 0] *= self._dixon_coles_tau(0, 0, home_xg, away_xg)
            matrix[1, 0] *= self._dixon_coles_tau(1, 0, home_xg, away_xg)
            matrix[0, 1] *= self._dixon_coles_tau(0, 1, home_xg, away_xg)
            matrix[1, 1] *= self._dixon_coles_tau(1, 1, home_xg, away_xg)

        # Re-normalize to ensure a valid probability distribution
        total = matrix.sum()
        if total > 0:
            matrix /= total

        return matrix

    def calculate_probabilities(self, home_xg: float, away_xg: float) -> Dict[str, float]:
        """
        Derives all major betting market probabilities from the score matrix.
        Includes safety guardrails: Lambda clamping, dampening, and prob bounds.
        """
        # 1. Lambda Guardrails
        home_xg = max(min(home_xg, 3.5), 0.2)
        away_xg = max(min(away_xg, 3.5), 0.2)
        
        matrix = self.calculate_score_matrix(home_xg, away_xg)
        
        # 2. Raw Probabilities
        # Match Outcome (1X2)
        prob_home = np.sum(np.tril(matrix, -1))
        prob_draw = np.sum(np.diag(matrix))
        prob_away = np.sum(np.triu(matrix, 1))
        
        # Over/Under Goals (2.5 & 1.5)
        h_idx, a_idx = np.indices(matrix.shape)
        total_goals = h_idx + a_idx
        prob_over_2_5 = np.sum(matrix[total_goals > 2.5])
        prob_under_2_5 = np.sum(matrix[total_goals < 2.5])
        prob_over_1_5 = np.sum(matrix[total_goals > 1.5])
        prob_under_3_5 = np.sum(matrix[total_goals < 3.5])  # New: U3.5 (low variance)
        
        # BTTS
        prob_btts_yes = np.sum(matrix[1:, 1:])
        prob_btts_no = 1.0 - prob_btts_yes
        
        # 3. Confidence Dampening (Binary Shrinkage towards 0.5)
        # Apply to key markets as requested to reduce sharpness
        prob_over_2_5 = self._dampen(prob_over_2_5)
        prob_under_2_5 = 1.0 - prob_over_2_5 # Maintain coherency
        
        # Optimization Markets: Moderate dampening (alpha=0.85) for stability
        prob_over_1_5 = self._dampen(prob_over_1_5, alpha=0.85)
        
        prob_btts_yes = self._dampen(prob_btts_yes)
        prob_btts_no = 1.0 - prob_btts_yes
        
        # For 1X2, usage symmetric multinomial dampening to preserve neutrality
        prob_home, prob_draw, prob_away = self._dampen_multinomial(
            [prob_home, prob_draw, prob_away], alpha=0.80
        )
        
        # 4. Probability Floor & Ceiling
        prob_home = self._clamp(prob_home)
        prob_draw = self._clamp(prob_draw)
        prob_away = self._clamp(prob_away)
        
        prob_over_2_5 = self._clamp(prob_over_2_5)
        prob_under_2_5 = self._clamp(prob_under_2_5)
        prob_over_1_5 = self._clamp(prob_over_1_5)
        prob_under_3_5 = self._clamp(self._dampen(prob_under_3_5, alpha=0.85))  # Light dampening for U3.5
        
        prob_btts_yes = self._clamp(prob_btts_yes)
        prob_btts_no = self._clamp(prob_btts_no)
        
        prob_home_u15 = self._dampen(np.sum(matrix[0:2, :]))
        prob_away_u15 = self._dampen(np.sum(matrix[:, 0:2]))

        return {
            "home_win": prob_home,
            "draw": prob_draw,
            "away_win": prob_away,
            "over_2_5": prob_over_2_5,
            "under_2_5": prob_under_2_5,
            "over_1_5": prob_over_1_5,
            "under_3_5": prob_under_3_5,  # New: low variance market
            "btts_yes": prob_btts_yes,
            "btts_no": prob_btts_no,
            "home_under_1_5": self._clamp(prob_home_u15),
            "away_under_1_5": self._clamp(prob_away_u15),
            "projected_home_goals": home_xg,
            "projected_away_goals": away_xg
        }

    def _dampen(self, p: float, alpha: float = 0.80) -> float:
        """
        Shrinks probability towards 0.5 (Maximum Entropy for binary).
        New P = 0.5 + alpha * (Old P - 0.5)
        Reduced dampening (0.70 -> 0.80) to preserve more edge.
        """
        return 0.5 + alpha * (p - 0.5)

    def _dampen_multinomial(self, probs: list[float], alpha: float = 0.80) -> list[float]:
        """
        Shrinks multinomial probabilities symmetrically towards 1/K (Standard Balanced State).
        For K=3 (1X2), target is 0.333.
        """
        K = len(probs)
        target = 1.0 / K
        return [target + alpha * (p - target) for p in probs]

    def _clamp(self, p: float, min_p: float = 0.05, max_p: float = 0.92) -> float:
        """
        Clamps probability to avoid 0.0 or 1.0 certainty.
        Relaxed ceiling (0.85 -> 0.92) to restore elite expressibility.
        """
        return max(min(p, max_p), min_p)

class NegativeBinomialEngine:
    """
    Models count data with over-dispersion (Variance > Mean).
    Ideal for Corners, where clustering occurs.
    """
    
    def __init__(self, max_count: int = 25):
        self.max_count = max_count

    def convert_params(self, mu: float, variance: float):
        """
        Converts Mean/Variance notation to n/p for scipy.stats.nbinom.
        Constraint: Variance must be > Mean.
        """
        if variance <= mu:
            # Fallback to Poisson-like behavior (low dispersion)
            # We enforce a minimal over-dispersion gap for stability
            variance = mu * 1.05
            
        p = mu / variance
        n = (mu * p) / (1 - p)
        return n, p

    def calculate_probabilities(self, mu_home: float, var_home: float, mu_away: float, var_away: float) -> Dict[str, float]:
        """
        Calculates markets derived from Negative Binomial distributions.
        """
        from scipy.stats import nbinom
        
        # 1. Parameter Conversion
        # Calibration Fix: Smooth Variance scaling (prevents non-monotonic jumps)
        mu_sum = mu_home + mu_away
        
        # Sigmoid-style transition for the 9.0 threshold
        # Boost is ~1.5 at low sums, decays to 1.0 as sum crosses 9.0
        # Formula: 1.0 + 0.5 / (1 + exp(1.5 * (mu_sum - 8.5)))
        boost = 1.0 + 0.5 / (1 + np.exp(1.5 * (mu_sum - 8.5)))
        
        var_home *= boost
        var_away *= boost

        n_h, p_h = self.convert_params(mu_home, var_home)
        n_a, p_a = self.convert_params(mu_away, var_away)
        
        # 2. PMFs
        x = np.arange(self.max_count + 1)
        pmf_home = nbinom.pmf(x, n_h, p_h)
        pmf_away = nbinom.pmf(x, n_a, p_a)
        
        # 3. Joint Matrix (assuming independence initially)
        matrix = np.outer(pmf_home, pmf_away)
        
        # 3b. Structural Normalization (No cosmetic inflation)
        matrix /= matrix.sum()
        
        # 4. Markets
        # Home/Away Corners (1X2 Analytical)
        prob_home_win = np.sum(np.tril(matrix, -1))
        prob_away_win = np.sum(np.triu(matrix, 1))
        prob_draw = np.sum(np.diag(matrix))
        
        # Dampen toward 1/3 symmetric (Multinomial safe)
        prob_home_win, prob_draw, prob_away_win = self._dampen_multinomial(
            [prob_home_win, prob_draw, prob_away_win], alpha=0.80
        )
            
        # Final Conservation Step (Clamping can break Sum=1.0)
        p1x2 = [self._clamp(prob_home_win), self._clamp(prob_draw), self._clamp(prob_away_win)]
        sum_p = sum(p1x2)
        p1x2 = [p / sum_p for p in p1x2]
        
        # Total Corners
        h_idx, a_idx = np.indices(matrix.shape)
        total = h_idx + a_idx
        
        # Standard Lines (Independent binary markets)
        # CALIBRATION FIX: U11.5 is High Freq (~72%) -> Light damping (0.90) preserves edge
        prob_under_11_5 = self._dampen(np.sum(matrix[total < Thresholds.CORNERS_U11_LINE]), alpha=0.90)
        prob_over_11_5 = 1.0 - prob_under_11_5
        
        # Optimization Market: Moderate dampening for stability
        prob_over_7_5 = self._dampen(np.sum(matrix[total > 7.5]), alpha=0.85)
        
        # Team Corners - Expanded Lines (2.5 to 6.5)
        # Home
        prob_home_u25 = self._dampen(np.sum(pmf_home[:3]))
        prob_home_u35 = self._dampen(np.sum(pmf_home[:4]))
        prob_home_u45 = self._dampen(np.sum(pmf_home[:5]))
        prob_home_u55 = self._dampen(np.sum(pmf_home[:6]))
        
        # Away
        prob_away_u25 = self._dampen(np.sum(pmf_away[:3]))
        prob_away_u35 = self._dampen(np.sum(pmf_away[:4]))
        prob_away_u45 = self._dampen(np.sum(pmf_away[:5]))
        prob_away_u55 = self._dampen(np.sum(pmf_away[:6]))
        
        return {
            "corners_home_win": p1x2[0],
            "corners_draw": p1x2[1],
            "corners_away_win": p1x2[2],
            "corners_home_more": p1x2[0], # Alias
            "corners_away_more": p1x2[2], # Alias
            "corners_under_11_5": self._clamp(prob_under_11_5),
            "corners_over_11_5": self._clamp(prob_over_11_5),
            "corners_over_7_5": self._clamp(prob_over_7_5),
            
            # Home Under
            "corners_home_under_2_5": self._clamp(prob_home_u25),
            "corners_home_under_3_5": self._clamp(prob_home_u35),
            "corners_home_under_4_5": self._clamp(prob_home_u45),
            "corners_home_under_5_5": self._clamp(prob_home_u55),
            
            # Home Over (Derived)
            "corners_home_over_2_5": self._clamp(1.0 - prob_home_u25),
            "corners_home_over_3_5": self._clamp(1.0 - prob_home_u35),
            "corners_home_over_4_5": self._clamp(1.0 - prob_home_u45),
            "corners_home_over_5_5": self._clamp(1.0 - prob_home_u55),

            # Away Under
            "corners_away_under_2_5": self._clamp(prob_away_u25),
            "corners_away_under_3_5": self._clamp(prob_away_u35),
            "corners_away_under_4_5": self._clamp(prob_away_u45),
            "corners_away_under_5_5": self._clamp(prob_away_u55),
            
            # Away Over (Derived)
            "corners_away_over_2_5": self._clamp(1.0 - prob_away_u25),
            "corners_away_over_3_5": self._clamp(1.0 - prob_away_u35),
            "corners_away_over_4_5": self._clamp(1.0 - prob_away_u45),
            "corners_away_over_5_5": self._clamp(1.0 - prob_away_u55)
        }

    def _dampen(self, p: float, alpha: float = 0.80) -> float:
        return 0.5 + alpha * (p - 0.5)

    def _dampen_multinomial(self, probs: list[float], alpha: float = 0.80) -> list[float]:
        """
        Shrinks multinomial probabilities symmetrically towards 1/K.
        """
        K = len(probs)
        target = 1.0 / K
        return [target + alpha * (p - target) for p in probs]

    def _clamp(self, p: float, min_p: float = 0.05, max_p: float = 0.92) -> float:
        return max(min(p, max_p), min_p)

class ZeroInflatedEngine:
    """
    Models sparse count data (Cards) where '0' is more frequent than Poisson predicts.
    """
    
    # Market-scoped clamp ceilings (prevent overconfidence per market)
    CLAMP_CEILING_U55: float = 0.88  # Cards U5.5 - aligned with 74.0% base rate
    CLAMP_CEILING_DEFAULT: float = 0.92
    
    # Damping Policies (Explicit Separation)
    # High Freq: O2.5 (77.1% base rate) -> Light damping (0.90) preserves edge
    ALPHA_HIGH_FREQ: float = 0.90 
    
    # Mid Freq / Under Markets: U4.5, U5.5 (58-74% base rate)
    # Use slightly conservative but consistent damping
    ALPHA_MID_FREQ: float = 0.88

    def __init__(self, max_count: int = 15):
        self.max_count = max_count
        
    def pmf(self, k: np.ndarray, mu: float, pi: float) -> np.ndarray:
        """
        ZIP PMF:
        P(k=0) = pi + (1-pi)*Poi(0)
        P(k>0) = (1-pi)*Poi(k)
        """
        poi_pmf = poisson.pmf(k, mu)
        
        # Adjust for Inflation
        probs = (1 - pi) * poi_pmf
        probs[0] += pi
        
        return probs

    def calculate_probabilities(self, mu_total: float, pi_zero: float = 0.03, var_mult: float = 1.0) -> Dict[str, float]:
        """
        Calculates Card market probabilities.
        Typically we model Total Cards directly rather than H/A separately due to ref influence.
        
        Args:
            mu_total: Expected total cards.
            pi_zero: Zero-inflation parameter.
            var_mult: Variance multiplier for intensity damping (1.0 = no effect).
        """
        x = np.arange(self.max_count + 1)
        
        # Action 2: Apply intensity variance scaling
        # For ZIP, increasing "variance" means spreading the distribution.
        # We simulate this by scaling mu to widen the tail.
        adjusted_mu = mu_total * var_mult
        
        probs = self.pmf(x, adjusted_mu, pi_zero)
        
        # Markets
        # Under 4.5 Cards - Default Dynamic Policy
        prob_under_4_5 = self._apply_policy(np.sum(probs[:5]), "CARDS_U45")
        prob_over_4_5 = 1.0 - prob_under_4_5

        # Over 2.5 Cards - High Frequency Policy
        # Explicitly decoupled from U5.5 logic using Policy Guard
        prob_over_2_5 = self._apply_policy(np.sum(probs[3:]), "CARDS_O25")

        # Under 5.5 Cards - Rare Event Policy (with ceiling)
        # Uses default dynamic damping but with strict ceiling
        prob_under_5_5 = self._apply_policy(np.sum(probs[:6]), "CARDS_U55")
        
        return {
            "cards_under_4_5": self._clamp(prob_under_4_5),
            "cards_over_4_5": self._clamp(prob_over_4_5),
            "cards_over_2_5": self._clamp(prob_over_2_5),
            "cards_under_5_5": self._clamp(prob_under_5_5, max_p=self.CLAMP_CEILING_U55)
        }

    def _apply_policy(self, p: float, market_type: str) -> float:
        """
        Policy Assertion Guard: Enforces correct damping policy per market type.
        
        Prevents misrouting and makes violations fail-fast.
        """
        # Valid markets set
        assert market_type in {"CARDS_O25", "CARDS_U55", "CARDS_U45"}, f"Unknown market policy: {market_type}"
        
        if market_type == "CARDS_O25":
            # High Frequency Policy -> Must use ALPHA_HIGH_FREQ
            return self._dampen(p, alpha=self.ALPHA_HIGH_FREQ)
            
        elif market_type == "CARDS_U55" or market_type == "CARDS_U45":
            # Mid Frequency Under Markets -> Use dedicated alpha
            return self._dampen(p, alpha=self.ALPHA_MID_FREQ)
            
        return self._dampen(p)

    def _dampen(self, p: float, alpha: float = None) -> float:
        """
        Confidence-aware damping: applies stronger regression for high-confidence predictions.
        
        Args:
            p: Probability to dampen
            alpha: Optional override. If None, uses dynamic logic.
        
        Dynamic Logic (Low Frequency / Rare Event Policy):
        - p > 0.75: Use alpha=0.72 (aggressive)
        - p <= 0.75: Use alpha=0.80 (standard)
        """
        if alpha is None:
            if p > 0.75:
                alpha = 0.72
            else:
                alpha = 0.80
                
        return 0.5 + alpha * (p - 0.5)

    def _clamp(self, p: float, min_p: float = 0.05, max_p: float = None) -> float:
        """Market-scoped clamping with configurable ceiling."""
        max_p = max_p if max_p is not None else self.CLAMP_CEILING_DEFAULT
        return max(min(p, max_p), min_p)

class CardsNBEngine:
    """
    Models cards using Negative Binomial for Home/Away components.
    User Weighted Formula: HY + AY + 2*(HR + AR).
    """
    
    def __init__(self, max_count: int = 15):
        self.max_count = max_count
        self.nb_engine = NegativeBinomialEngine(max_count=max_count)

    def calculate_probabilities(self, mu_home: float, var_home: float, mu_away: float, var_away: float) -> Dict[str, float]:
        from scipy.stats import nbinom
        
        n_h, p_h = self.nb_engine.convert_params(mu_home, var_home)
        n_a, p_a = self.nb_engine.convert_params(mu_away, var_away)
        
        x = np.arange(self.max_count + 1)
        pmf_h = nbinom.pmf(x, n_h, p_h)
        pmf_a = nbinom.pmf(x, n_a, p_a)
        
        matrix = np.outer(pmf_h, pmf_a)
        matrix /= matrix.sum()
        
        h_idx, a_idx = np.indices(matrix.shape)
        total = h_idx + a_idx
        
        # Over 2.5 Cards (Primary Market)
        prob_over_2_5 = self.nb_engine._dampen(np.sum(matrix[total > 2.5]), alpha=0.85)
        
        # Under 4.5 Cards (Legacy Audit Point)
        prob_under_4_5 = self.nb_engine._dampen(np.sum(matrix[total < 4.5]), alpha=0.85)

        # Under 5.5 Cards (User Request)
        prob_under_5_5 = self.nb_engine._dampen(np.sum(matrix[total < 5.5]), alpha=0.85)
        
        return {
            "cards_over_2_5": self.nb_engine._clamp(prob_over_2_5),
            "cards_under_4_5": self.nb_engine._clamp(prob_under_4_5),
            "cards_under_5_5": self.nb_engine._clamp(prob_under_5_5),
            "projected_home_cards": mu_home,
            "projected_away_cards": mu_away
        }

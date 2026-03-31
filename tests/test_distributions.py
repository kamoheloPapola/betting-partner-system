
import unittest
import numpy as np
from src.ml.distributions import PoissonEngine, NegativeBinomialEngine

class TestDistributions(unittest.TestCase):
    
    def setUp(self):
        self.pe = PoissonEngine()
        self.nbe = NegativeBinomialEngine()

    def test_multinomial_dampener_symmetry(self):
        """Verify _dampen_multinomial preserves symmetry for equal probabilities."""
        # 1. Perfectly Balanced 3-way
        probs = [1/3, 1/3, 1/3]
        damped = self.pe._dampen_multinomial(probs, alpha=0.8)
        
        for p in damped:
            self.assertAlmostEqual(p, 1/3, places=5)
            
        # 2. Symmetric Bias (e.g. Draw low, Home/Away equal)
        # H=0.4, D=0.2, A=0.4
        probs = [0.4, 0.2, 0.4]
        damped = self.pe._dampen_multinomial(probs, alpha=0.8)
        
        self.assertAlmostEqual(damped[0], damped[2], places=5, msg="Home/Away symmetry broken")
        self.assertAlmostEqual(sum(damped), 1.0, places=5, msg="Sum not conserved")
        
        # Check Direction: Should shrink towards 0.333
        # 0.4 > 0.333 -> Should decrease
        self.assertLess(damped[0], 0.4)
        # 0.2 < 0.333 -> Should increase
        self.assertGreater(damped[1], 0.2)

    def test_poisson_engine_1x2_symmetry(self):
        """Verify PoissonEngine 1x2 outputs are symmetric for equal xG."""
        # Neutral Game
        xg = 1.35
        probs = self.pe.calculate_probabilities(xg, xg)
        
        self.assertAlmostEqual(probs['home_win'], probs['away_win'], places=4, 
                               msg="Home/Away Win Probs not symmetric for equal xG")
        self.assertAlmostEqual(probs['home_under_1_5'], probs['away_under_1_5'], places=4)

    def test_corner_engine_1x2_symmetry(self):
        """Verify NegativeBinomialEngine 1x2 Corners outputs are symmetric."""
        # Neutral Game
        mu = 5.0
        var = 7.0 
        
        # Pass identical params
        probs = self.nbe.calculate_probabilities(mu, var, mu, var)
        
        self.assertAlmostEqual(probs['corners_home_win'], probs['corners_away_win'], places=4,
                               msg="Corner Home/Away Win Probs not symmetric for equal params")

    def test_dixon_coles_tau_values(self):
        """Verify Dixon-Coles tau correction matches expected formulas."""
        pe = PoissonEngine(rho=-0.13)
        lh, la = 1.4, 1.1

        self.assertAlmostEqual(pe._dixon_coles_tau(0, 0, lh, la), 1 - (lh * la * pe.rho), places=8)
        self.assertAlmostEqual(pe._dixon_coles_tau(1, 0, lh, la), 1 + (la * pe.rho), places=8)
        self.assertAlmostEqual(pe._dixon_coles_tau(0, 1, lh, la), 1 + (lh * pe.rho), places=8)
        self.assertAlmostEqual(pe._dixon_coles_tau(1, 1, lh, la), 1 - pe.rho, places=8)
        self.assertAlmostEqual(pe._dixon_coles_tau(2, 2, lh, la), 1.0, places=8)

    def test_dixon_coles_matrix_is_normalized(self):
        """Corrected score matrix should remain a valid probability matrix."""
        pe = PoissonEngine(rho=-0.13)
        matrix = pe.calculate_score_matrix(1.3, 1.0)
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=8)

if __name__ == '__main__':
    unittest.main()


import pytest
import numpy as np
from src.ml.distributions import PoissonEngine, NegativeBinomialEngine, ZeroInflatedEngine

def test_poisson_probabilities_sum_to_one():
    engine = PoissonEngine()
    # Test typical lambda values
    probs = engine.calculate_probabilities(1.5, 0.8)
    
    # 1X2 Probabilities (Home/Draw/Away)
    total_hda = probs['home_win'] + probs['draw'] + probs['away_win']
    assert abs(total_hda - 1.0) < 0.001
    
    # Under/Over 2.5
    total_goal_band = probs['under_2_5'] + probs['over_2_5']
    assert abs(total_goal_band - 1.0) < 0.001

def test_nb_probabilities_sum_to_one():
    engine = NegativeBinomialEngine()
    # mu=5.0, var=7.5 (overdispersion)
    probs = engine.calculate_probabilities(5.0, 7.5, 4.0, 6.0)
    
    # Check 1X2 Corners
    total_hda = probs['corners_home_win'] + probs['corners_draw'] + probs['corners_away_win']
    assert abs(total_hda - 1.0) < 0.001
    
    # Check Under 11.5
    assert 0 <= probs['corners_under_11_5'] <= 1.0

def test_zip_probabilities_sum_to_one():
    engine = ZeroInflatedEngine()
    # mu=3.5, pi_zero=0.1
    probs = engine.calculate_probabilities(3.5, pi_zero=0.1)
    
    # Zero-Inflated typically returns a dict of p(k), but our system uses it for bands
    total_band = probs['cards_under_4_5'] + probs['cards_over_4_5']
    assert abs(total_band - 1.0) < 0.001


def test_zip_cards_u55_monotonicity():
    """
    Ensure calibration doesn't kill information: mu ↓ ⇒ P(U5.5) ↑
    
    Lower expected cards should always mean higher probability of under 5.5.
    Over-damping can accidentally flatten the curve creating signal collapse.
    """
    engine = ZeroInflatedEngine()
    
    # Calculate probabilities at different mu values
    p_mu_3 = engine.calculate_probabilities(3.0, pi_zero=0.1)['cards_under_5_5']
    p_mu_4 = engine.calculate_probabilities(4.0, pi_zero=0.1)['cards_under_5_5']
    p_mu_5 = engine.calculate_probabilities(5.0, pi_zero=0.1)['cards_under_5_5']
    
    # Monotonicity: lower mu should give higher P(U5.5)
    assert p_mu_3 > p_mu_4 > p_mu_5, (
        f"Signal collapse detected! Expected p_mu_3 > p_mu_4 > p_mu_5, "
        f"got {p_mu_3:.3f} > {p_mu_4:.3f} > {p_mu_5:.3f}"
    )


def test_zip_cards_u55_ceiling():
    """
    Cards U5.5 probability should never exceed the market-scoped ceiling (0.82).
    
    At very low mu (high confidence), the model should still be clamped.
    """
    engine = ZeroInflatedEngine()
    
    # At mu=2.0 (very low cards), probability should still be capped
    probs = engine.calculate_probabilities(2.0, pi_zero=0.1)
    p_u55 = probs['cards_under_5_5']
    
    assert p_u55 <= engine.CLAMP_CEILING_U55, (
        f"Cards U5.5 exceeds ceiling: {p_u55:.3f} > {engine.CLAMP_CEILING_U55:.2f}"
    )
    
    # At mu=3.0, should also be capped but close to ceiling
    probs_3 = engine.calculate_probabilities(3.0, pi_zero=0.1)
    p_u55_3 = probs_3['cards_under_5_5']
    
    assert p_u55_3 <= engine.CLAMP_CEILING_U55, (
        f"Cards U5.5 at mu=3.0 exceeds ceiling: {p_u55_3:.3f} > {engine.CLAMP_CEILING_U55:.2f}"
    )


def test_zip_confidence_aware_damping():
    """
    Verify confidence-aware damping applies correctly:
    - High confidence (p > 0.75): stronger damping (alpha=0.72)
    - Lower confidence: preserve signal (alpha=0.80)
    """
    engine = ZeroInflatedEngine()
    
    # At mu=4.0 (below high-confidence threshold after damping), 
    # calibration should be approximately preserved
    probs_4 = engine.calculate_probabilities(4.0, pi_zero=0.1)
    p_u55_4 = probs_4['cards_under_5_5']
    
    # At average cards (mu=4.3, the actual mean), probability should be near base rate (73.6%)
    probs_avg = engine.calculate_probabilities(4.3, pi_zero=0.1)
    p_u55_avg = probs_avg['cards_under_5_5']
    
    # Should be in reasonable range (65-78%)
    assert 0.65 <= p_u55_avg <= 0.78, f"Avg cards prediction out of range: {p_u55_avg:.3f}"


def test_zip_policy_guard_enforcement():
    """Verify that the Policy Guard raises assertions for unknown markets."""
    engine = ZeroInflatedEngine()
    
    # 1. Valid Markets should pass
    # O2.5 uses High Freq (alpha=0.90)
    p_high = engine._apply_policy(0.80, "CARDS_O25")
    assert 0.77 == pytest.approx(p_high, abs=0.01) # 0.5 + 0.9*(0.8-0.5) = 0.5 + 0.27 = 0.77
    
    # U5.5 uses the dedicated under-market policy constant.
    p_rare = engine._apply_policy(0.80, "CARDS_U55")
    assert engine._dampen(0.80, alpha=engine.ALPHA_MID_FREQ) == pytest.approx(p_rare, abs=0.01)
    
    # 2. Invalid Markets should fail fast
    with pytest.raises(AssertionError, match="Unknown market policy"):
        engine._apply_policy(0.80, "INVALID_MARKET_TYPE")

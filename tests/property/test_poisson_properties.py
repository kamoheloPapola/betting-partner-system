
import pytest
from hypothesis import given, strategies as st
from src.ml.distributions import PoissonEngine

@given(
    home_lambda=st.floats(min_value=0.1, max_value=5.0),
    away_lambda=st.floats(min_value=0.1, max_value=5.0)
)
def test_poisson_probabilities_always_valid(home_lambda, away_lambda):
    """
    Property: Probabilities must ALWAYS sum to ~1.0, regardless of lambda inputs.
    """
    engine = PoissonEngine()
    probs = engine.calculate_probabilities(home_lambda, away_lambda)
    
    # Check 1: Completeness (Sum of Outcomes ≈ 1.0)
    # Note: 'under_2_5' etc are derived subsets, so we only sum disjoint outcomes (1X2)
    total_1x2 = probs['home_win'] + probs['draw'] + probs['away_win']
    assert 0.99 <= total_1x2 <= 1.01, f"1X2 Sum violation: {total_1x2} (Inputs: {home_lambda}, {away_lambda})"
    
    # Check 2: Range Validity [0, 1]
    for key, value in probs.items():
        if key.startswith("projected_"):
            continue
        assert 0.0 <= value <= 1.0, f"Probability range violation for {key}: {value}"
        
    # Check 3: Consistency
    # If the clamped projected lambdas remain meaningfully separated,
    # home win should still dominate away win.
    if probs['projected_home_goals'] > probs['projected_away_goals'] + 1.0:
        assert probs['home_win'] > probs['away_win'], (
            "Home win should be favored when projected home goals are significantly higher"
        )


import pytest
import math
from src.strategies.css_math import calculate_css, count_correlated_pairs

# --- HELPER FIXTURES ---

@pytest.fixture
def high_stable_leg():
    return {
        'market_name': 'CORNERS_U11.5', # Weight 0.95
        'confidence': 0.85,
        'league': 'PL',
        'date': '2025-01-01T15:00:00'
    }

@pytest.fixture
def medium_leg():
    return {
        'market_name': 'CORNERS_O7.5', # Weight 0.90
        'confidence': 0.70,
        'league': 'PD',
        'date': '2025-01-02T15:00:00'
    }

@pytest.fixture
def volatile_leg():
    return {
        'market_name': '1X2_HOME', # Weight 0.75
        'confidence': 0.65,
        'league': 'SA',
        'date': '2025-01-03T15:00:00'
    }

# --- TESTS ---

def test_single_selection_combo(high_stable_leg):
    """Test CSS for a single selection."""
    legs = [high_stable_leg]
    css, details = calculate_css(legs, corr_count=0)
    
    # Assertions
    assert 0 < css < 1.0
    assert details['corr_penalty'] == 1.0 # No correlation
    # For single leg with p=0.85, w=0.95:
    # s_i = 0.85 * 0.95 * (1 - (1-0.85)) = 0.85 * 0.95 * 0.85 = 0.686375
    # surv = 1 - (1-0.85) = 0.85
    # css = 0.686375 * 1 * 0.85 = 0.5834...
    expected_s_i = 0.85 * 0.95 * 0.85
    assert math.isclose(details['s_i_product'], expected_s_i, rel_tol=1e-5)

def test_css_bounds_0_to_1(high_stable_leg, medium_leg):
    """CSS must be strictly between 0 and 1."""
    legs = [high_stable_leg, medium_leg]
    css, _ = calculate_css(legs, corr_count=0)
    assert 0 < css < 1.0

def test_zero_probability_handling():
    """CSS must be 0 if any p_i is 0."""
    legs = [{
        'market_name': 'Corners Under 11.5',
        'confidence': 0.0,
        'league': 'PL'
    }]
    css, _ = calculate_css(legs, corr_count=0)
    assert css == 0.0

def test_css_strictly_lt_product_pi(high_stable_leg, medium_leg):
    """Assert CSS < product(p_i). Stability penalties must lower the raw joint probability."""
    legs = [high_stable_leg, medium_leg]
    raw_joint_p = high_stable_leg['confidence'] * medium_leg['confidence']
    
    css, _ = calculate_css(legs, corr_count=0)
    
    assert css < raw_joint_p, f"CSS {css} should be less than raw joint P {raw_joint_p}"

def test_hard_rejection_gate():
    """Assert rejection if CSS < 0.18."""
    # Construct a weak slip
    weak_leg = {
        'market_name': 'Risky Bet',
        'confidence': 0.40, # Low conf
        'league': 'PL'
    }
    # Even single leg logic
    css, _ = calculate_css([weak_leg], corr_count=0)
    
    # This assertion is logic-level (what the caller should do), 
    # but here we verify the score itself is low enough to trigger it.
    assert css < 0.18

def test_monotonicity_volatility_increases(high_stable_leg):
    """CSS strictly decreases when volatility increases (confidence decreases)."""
    # Baseline
    legs_high = [high_stable_leg]
    css_high, _ = calculate_css(legs_high, corr_count=0)
    
    # Lower confidence (higher volatility)
    worse_leg = high_stable_leg.copy()
    worse_leg['confidence'] = 0.70 # Was 0.85
    legs_low = [worse_leg]
    css_low, _ = calculate_css(legs_low, corr_count=0)
    
    assert css_low < css_high, "CSS should drop when confidence/stability drops"

def test_monotonicity_correlation_increases(high_stable_leg, medium_leg):
    """CSS strictly decreases when correlation increases."""
    legs = [high_stable_leg, medium_leg]
    
    # CSS with 0 correlation
    css_0, _ = calculate_css(legs, corr_count=0)
    
    # CSS with 1 correlation pair
    css_1, _ = calculate_css(legs, corr_count=1)
    
    assert css_1 < css_0, "CSS should drop when correlation count increases"

def test_invariance_to_order(high_stable_leg, medium_leg):
    """CSS is invariant to selection order."""
    legs_ab = [high_stable_leg, medium_leg]
    legs_ba = [medium_leg, high_stable_leg]
    
    css_ab, _ = calculate_css(legs_ab, corr_count=0)
    css_ba, _ = calculate_css(legs_ba, corr_count=0)
    
    assert css_ab == css_ba

# --- CORRELATION COUNTING LOGIC TESTS ---

def test_count_correlated_pairs_logic():
    # 2 legs, same league, same time -> 1 pair
    l1 = {'league': 'PL', 'date': '2025-01-01T15:00:00'}
    l2 = {'league': 'PL', 'date': '2025-01-01T15:00:00'}
    assert count_correlated_pairs([l1, l2]) == 1
    
    # 2 legs, diff league, same time -> 0 pair
    l3 = {'league': 'PD', 'date': '2025-01-01T15:00:00'}
    assert count_correlated_pairs([l1, l3]) == 0
    
    # 3 legs, all same -> 3 pairs (1-2, 1-3, 2-3)
    l4 = {'league': 'PL', 'date': '2025-01-01T15:00:00'}
    assert count_correlated_pairs([l1, l2, l4]) == 3


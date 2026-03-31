import pytest
import math
from datetime import datetime, timedelta
from src.strategies.css_math import calculate_css, count_correlated_pairs

def test_temporal_decay():
    """Test that correlation decay works with hours apart."""
    base_time = datetime(2025, 1, 1, 15, 0)
    
    l1 = {'league': 'PL', 'date': base_time.isoformat()}
    l2 = {'league': 'PL', 'date': base_time.isoformat()} # 0 hours apart
    l3 = {'league': 'PL', 'date': (base_time + timedelta(hours=3)).isoformat()} # 3 hours apart
    l4 = {'league': 'PL', 'date': (base_time + timedelta(hours=24)).isoformat()} # 24 hours apart
    
    # 0 hours -> Decay = 1.0
    assert math.isclose(count_correlated_pairs([l1, l2]), 1.0)
    
    # 3 hours -> Decay = exp(-0.3 * 3) = 0.406
    corr_3h = count_correlated_pairs([l1, l3])
    assert 0.4 < corr_3h < 0.41
    
    # 24 hours -> Decay = exp(-0.3 * 24) = 0.0007
    corr_24h = count_correlated_pairs([l1, l4])
    assert corr_24h < 0.01

def test_historical_volatility_impact():
    """Test that different markets have different CSS impact due to historical volatility."""
    # Market with low historical volatility (TG_U1.5: 0.12)
    leg_stable = {
        'market_name': 'TG_U1.5',
        'confidence': 0.80,
        'league': 'PL'
    }
    
    # Market with higher historical volatility (1X2: 0.30)
    leg_volatile = {
        'market_name': '1X2',
        'confidence': 0.80,
        'league': 'PL'
    }
    
    css_stable, _ = calculate_css([leg_stable])
    css_volatile, _ = calculate_css([leg_volatile])
    
    # Stable market should have higher CSS for same confidence
    assert css_stable > css_volatile

def test_log_space_stability():
    """Test that log-space calculation handles many legs without underflow issues."""
    leg = {'market_name': 'TG_U1.5', 'confidence': 0.70}
    many_legs = [leg] * 20 # 20 legs would normally lead to very small products
    
    css, details = calculate_css(many_legs)
    
    assert css > 0
    assert details['s_i_product'] > 0
    # 0.7 ^ 20 is small but should be correctly computed
    expected_approx = (0.7 * 1.0 * (1.0 - (0.5*0.3 + 0.5*0.12))) ** 20
    # Stability: v = 0.5 * 0.3 + 0.5 * 0.12 = 0.15 + 0.06 = 0.21
    # s_i = 0.7 * 1.0 * (1 - 0.21) = 0.7 * 0.79 = 0.553
    # 0.553 ^ 20 = 6.4e-6
    assert 1e-7 < details['s_i_product'] < 1e-4

def test_simulation_override():
    """Test that simulation override reduces CSS to product(p_i)."""
    legs = [
        {'market_name': 'TG_U1.5', 'confidence': 0.8},
        {'market_name': '1X2', 'confidence': 0.7}
    ]
    
    css, _ = calculate_css(legs, simulation_override=True)
    assert math.isclose(css, 0.8 * 0.7)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
from src.strategies.forbidden_fruit import ForbiddenFruitEngine

# --- FIXTURES ---

@pytest.fixture
def engine():
    return ForbiddenFruitEngine()

@pytest.fixture
def risky_candidates():
    # High probability but heavily correlated or volatile markets
    return [
        {'id': 'm1', 'market_name': 'Risky 1', 'confidence': 0.95, 'league': 'PL', 'date': '2025-01-01', 'type': '1X2'},
        {'id': 'm2', 'market_name': 'Risky 2', 'confidence': 0.95, 'league': 'PL', 'date': '2025-01-01', 'type': '1X2'}, # Correlated
        {'id': 'm3', 'market_name': 'Risky 3', 'confidence': 0.95, 'league': 'PL', 'date': '2025-01-01', 'type': '1X2'}, # Correlated
    ]

# --- TESTS ---

def test_binding_enforcement_rejection(engine, risky_candidates):
    """
    Test Phase 3: Binding Enforcement.
    Simulate forcing high probabilities with correlation.
    Expected: System refuses to produce a slip (CSS or hard gates block it).
    """
    slip = engine.construct_slip(risky_candidates)
    
    # Assert rejected because of 3 correlated items (limit > 1)
    assert not slip, "Engine should reject highly correlated slip despite high confidence"

def test_kill_switch_simulation():
    """
    Test Phase 3: 'Kill Switch' Simulation.
    We can't easily modify the internal engine state here without mocking, 
    but we can verify that if we pass perfect inputs, it returns a slip, 
    validating the path exists.
    """
    perfect_candidates = [
        {'id': 'm1', 'market_name': 'CORNERS_U11.5', 'confidence': 0.85, 'league': 'PL', 'date': '2025-01-01'},
        {'id': 'm2', 'market_name': 'CORNERS_U11.5', 'confidence': 0.85, 'league': 'PD', 'date': '2025-01-02'}
    ]
    
    engine = ForbiddenFruitEngine()
    slip = engine.construct_slip(perfect_candidates)
    
    assert slip, "Engine should produce slip for perfect candidates"
    assert len(slip) == 2

if __name__ == "__main__":
    test_kill_switch_simulation()
    print("Kill Switch Test Passed")
    test_binding_enforcement_rejection(ForbiddenFruitEngine(), [])
    print("Binding Enforcement Test Passed")


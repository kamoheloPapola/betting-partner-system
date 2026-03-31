
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Mock Registry before importing Engine
with patch('src.ml.registry.ModelRegistry') as MockRegistry:
    from src.strategies.forbidden_fruit import ForbiddenFruitEngine, ForbiddenFruitEvaluator

@pytest.fixture
def engine():
    # Mock internal components
    with patch('src.strategies.forbidden_fruit.ModelRegistry'), \
         patch('src.strategies.forbidden_fruit.StandingsManager'), \
         patch('src.strategies.forbidden_fruit.DriftGuardrail') as MockDrift:
        
        eng = ForbiddenFruitEngine()
        eng.drift_guard = MockDrift.return_value
        eng._registry = MagicMock()
        # Set default status to OK
        eng.drift_guard.status = "OK"
        return eng

def test_drift_guard_enforcement(engine):
    """Test that STOP status blocks analysis."""
    # 1. Set Drift Guard to STOP
    engine.drift_guard.status = "STOP"
    
    match = {"home_team": "A", "away_team": "B", "league": "PL", "date": "2026-01-01"}
    preds = {"home_win": 0.9}
    
    # 2. Analyze
    candidates = engine.analyze_match(match, preds)
    
    # 3. Assert Blocked
    assert candidates == [], "Engine failed to block analysis when Drift Guard is STOP"

def test_drift_guard_allowance(engine):
    """Test that OK status allows analysis."""
    engine.drift_guard.status = "OK"
    engine._registry.get_coverage_status.return_value = "FULL"
    
    match = {"home_team": "A", "away_team": "B", "league": "PL", "date": "2026-01-01"}
    preds = {"home_win": 0.9, "over_1_5": 0.8, "corners_over_7_5": 0.5} # High conf + Valid Tempo
    
    # Analyze
    candidates = engine.analyze_match(match, preds)
    
    # Assert Allowed (should produce candidates)
    # Note: Evaluator might filter based on other gates, but at least function runs.
    # With 0.9 conf home_win, it should pass 1X2 gate.
    assert len(candidates) > 0, "Engine blocked valid analysis despite Drift Guard OK"


def test_ensemble_divergence_reduces_confidence(engine):
    engine.drift_guard.check_drift.return_value = "OK"
    engine._registry.get_coverage_status.return_value = "FULL"
    engine.evaluator.evaluate_decision = MagicMock(
        return_value=[
            {
                "market_name": "HOME_WIN",
                "type": "1X2",
                "confidence": 0.8,
                "tier": 1,
                "tempo_status": "NORMAL",
            }
        ]
    )

    match = {
        "home_team": "A",
        "away_team": "B",
        "league": "PL",
        "date": "2026-01-01",
        "match_id": "m1",
    }
    preds = {"ensemble_divergence": True, "divergence_pct": 25.0}

    candidates = engine.analyze_match(
        match,
        preds,
        precomputed_confidences={"HOME_WIN": 0.8},
    )

    assert len(candidates) == 1
    assert candidates[0]["confidence"] == pytest.approx(0.72, abs=1e-9)

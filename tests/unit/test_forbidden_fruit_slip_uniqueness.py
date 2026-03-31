import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.strategies.forbidden_fruit import ForbiddenFruitEngine

from unittest.mock import MagicMock, patch

def test_construct_slip_enforces_uniqueness():
    with patch('src.ml.registry.ModelRegistry') as MockRegistry:
        # Configure mock to pass coverage check
        mock_instance = MockRegistry.return_value
        mock_instance.get_coverage_status.return_value = "FULL"
        
        engine = ForbiddenFruitEngine()
    
    # Mock candidates where some are from the same match
    candidates = [
        {
            "match": "Borussia Dortmund vs Borussia Monchengladbach",
            "market": "CORNERS_U11.5",
            "confidence": 0.730,
            "league": "BL1",
            "id": "match_1",
            "date": "2025-12-19T20:00:00Z"
        },
        {
            "match": "Valencia vs Mallorca",
            "market": "CORNERS_U11.5",
            "confidence": 0.715,
            "league": "PD",
            "id": "match_2",
            "date": "2025-12-19T18:00:00Z"
        },
        {
            "match": "Valencia vs Mallorca",
            "market": "AWAY_TG_U1.5",
            "confidence": 0.690,
            "league": "PD",
            "id": "match_2",
            "date": "2025-12-19T18:00:00Z"
        },
        {
            "match": "Borussia Dortmund vs Borussia Monchengladbach",
            "market": "CARDS_U4.5",
            "confidence": 0.652,
            "league": "BL1",
            "id": "match_1",
            "date": "2025-12-19T20:00:00Z"
        }
    ]
    
    # Run engine logic
    slip = engine.construct_slip(candidates)
    
    # Verify results
    assert len(slip) == 2, f"Expected 2 unique matches, got {len(slip)}"
    
    # Verify high confidence ones are picked
    matches_in_slip = [leg['match'] for leg in slip]
    assert "Borussia Dortmund vs Borussia Monchengladbach" in matches_in_slip
    assert "Valencia vs Mallorca" in matches_in_slip
    
    # Verify specific markets and confidences
    dortmund_leg = next(l for l in slip if "Borussia Dortmund" in l['match'])
    valencia_leg = next(l for l in slip if "Valencia" in l['match'])
    
    assert dortmund_leg['confidence'] == 0.730
    assert dortmund_leg['market'] == "CORNERS_U11.5"
    
    assert valencia_leg['confidence'] == 0.715
    assert valencia_leg['market'] == "CORNERS_U11.5"
    
    print("✅ test_construct_slip_enforces_uniqueness PASSED")

if __name__ == "__main__":
    test_construct_slip_enforces_uniqueness()

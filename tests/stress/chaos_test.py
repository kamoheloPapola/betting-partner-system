
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).parents[2]))

# Import Exceptions
from src.core.exceptions import ModelNotFoundError, DataValidationError

# Import Components
from src.ml.registry import ModelRegistry
from src.ml.guards import PredictionGuard
from src.cli.commands.prediction import _predict_scalar
from src.strategies.forbidden_fruit import ForbiddenFruitEvaluator

class TestChaosFailureModes(unittest.TestCase):
    
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [1.5]
        self.mock_model.features = ['f1', 'f2']
        self.mock_model.meta = {'metrics': {'calibration_score': 0.05}, 'version': 'chaos_v1'}
        
        self.valid_df = pd.DataFrame([{'f1': 1, 'f2': 2}])
        self.context = "ChaosTest"

    # --- Case A: Model Registry Hard Fail ---
    def test_missing_model_hard_fail(self):
        """Case A: Rename/Missing Model File -> Assert ModelNotFoundError"""
        registry = ModelRegistry()
        
        # Mock get_production_model returning None
        with patch.object(registry, 'get_production_model', return_value=None):
            with patch.object(registry, 'get_production_model_for_league', return_value=None):
                 with self.assertRaises(ModelNotFoundError):
                     registry.load_model("ghost_model", league="PL")
                     
    # --- Case B: Corrupt Feature Column ---
    def test_corrupt_feature_validation(self):
        """Case B: Corrupt Feature Column (String in Float col) -> Assert DataValidationError"""
        # Testing _predict_scalar wrapper logic
        # We simulate a row that has a string where a number is expected
        row = pd.Series({'f1': 'NOT_A_NUMBER', 'f2': 2})
        
        # Wrapped call should raise DataValidationError
        with self.assertRaises(DataValidationError):
            _predict_scalar(self.mock_model, row, ['f1', 'f2'], self.context)

    # --- Case C: Schema Shift ---
    def test_schema_shift_guard_rejection(self):
        """Case C: Dropped Column -> Assert PredictionGuard Rejection"""
        # Input missing 'f2'
        bad_df = pd.DataFrame([{'f1': 1}]) 
        
        with self.assertRaises(RuntimeError) as cm:
            PredictionGuard.validate_prediction_integrity(self.mock_model, bad_df, self.context)
        
        self.assertIn("Schema Mismatch", str(cm.exception))

    # --- Case D: Missing Standings ---
    def test_force_missing_standings(self):
        """Case D: Force Missing Standings -> Verify Handling"""
        evaluator = ForbiddenFruitEvaluator()
        preds = {'home_win': 0.8, 'away_win': 0.1, 'draw': 0.1}
        
        # 1. With Standings (Normal)
        # We need a market... 1X2 always generated logic check
        # Hidden dependency: _parse_inputs calls _validate_underdog_standings only if underdog logic triggered
        
        # Let's test specific method handling
        # If standings is None, it should return default gate (True/1.0)
        gate = evaluator._get_adjusted_gate({'type': 'CORNERS'}, None)
        self.assertEqual(gate, evaluator.HARD_GATES['CORNERS'], "Should fallback to hard gate without standings")
        
        
        # Ensure it doesn't crash
        try:
             res = evaluator.evaluate(preds, standings_context=None)
             self.assertIsInstance(res, list)
        except Exception as e:
            self.fail(f"Missing standings caused crash: {e}")

if __name__ == '__main__':
    unittest.main()


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

if __name__ == '__main__':
    unittest.main()

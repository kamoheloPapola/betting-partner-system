"""
Tests for Corner Calibration Evaluator.

Unit tests for Brier score, ECE calculation, status classification,
data validation, and label calculation.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config.evaluation import CalibrationConfig
from src.core.exceptions import DataValidationError
from src.evaluation.corners_calibration import (
    CalibrationResult,
    CornerCalibrationEvaluator,
)


class TestCalibrationMetrics:
    """Test core calibration metrics."""
    
    def test_brier_score_perfect(self):
        """Perfect predictions should have Brier score = 0."""
        evaluator = CornerCalibrationEvaluator()
        
        probs = pd.Series([0.9, 0.1, 0.8, 0.2])
        actuals = pd.Series([1, 0, 1, 0])
        
        # Not perfect, but close
        brier = evaluator.brier_score(probs, actuals)
        assert 0 <= brier < 0.05
    
    def test_brier_score_worst(self):
        """Worst predictions should have Brier score = 1."""
        evaluator = CornerCalibrationEvaluator()
        
        probs = pd.Series([0.0, 1.0, 0.0, 1.0])
        actuals = pd.Series([1, 0, 1, 0])  # Opposite
        
        brier = evaluator.brier_score(probs, actuals)
        assert brier == 1.0
    
    def test_ece_perfect_calibration(self):
        """Perfectly calibrated model should have ECE ≈ 0."""
        evaluator = CornerCalibrationEvaluator()
        
        # Simulate perfect calibration
        n = 1000
        np.random.seed(42)
        probs = pd.Series(np.random.uniform(0, 1, n))
        actuals = pd.Series((np.random.uniform(0, 1, n) < probs).astype(float))
        
        ece = evaluator.ece_score(probs, actuals)
        # With random data, ECE should be low
        assert ece < 0.15
    
    def test_ece_poor_calibration(self):
        """Overconfident model should have high ECE."""
        evaluator = CornerCalibrationEvaluator()
        
        # Model predicts 90% but actual is 50%
        probs = pd.Series([0.9] * 100)
        actuals = pd.Series([1, 0] * 50)  # 50% actual
        
        ece = evaluator.ece_score(probs, actuals)
        assert ece > 0.30  # Should be high


class TestStatusClassification:
    """Test ECE-based status assignment."""
    
    def test_status_boost(self):
        """ECE < 0.02 should get BOOST status."""
        evaluator = CornerCalibrationEvaluator()
        status = evaluator.get_status(0.015)
        assert status == "BOOST"
    
    def test_status_eligible(self):
        """ECE in [0.03, 0.05) should get ELIGIBLE."""
        evaluator = CornerCalibrationEvaluator()
        status = evaluator.get_status(0.035)
        assert status == "ELIGIBLE"
    
    def test_status_exclude(self):
        """ECE > 0.10 should get EXCLUDE."""
        evaluator = CornerCalibrationEvaluator()
        status = evaluator.get_status(0.15)
        assert status == "EXCLUDE"


class TestDataLoading:
    """Test data loading and validation."""
    
    def test_validate_logs_valid(self):
        """Valid logs should pass validation."""
        evaluator = CornerCalibrationEvaluator()
        
        logs = pd.DataFrame({
            'match_id': ['1', '2', '3'],
            'league': ['PL', 'PL', 'PL'],
            'P_under_11_5': [0.7, 0.8, 0.6]
        })
        
        # Should not raise
        evaluator._validate_logs(logs)
    
    def test_validate_logs_missing_columns(self):
        """Logs missing required columns should raise error."""
        evaluator = CornerCalibrationEvaluator()
        
        logs = pd.DataFrame({
            'match_id': ['1', '2'],
            # Missing 'league' and 'P_under_11_5'
        })
        
        with pytest.raises(DataValidationError, match="missing required columns"):
            evaluator._validate_logs(logs)
    
    def test_validate_logs_invalid_probabilities(self):
        """Probabilities outside [0,1] should raise error."""
        evaluator = CornerCalibrationEvaluator()
        
        logs = pd.DataFrame({
            'match_id': ['1', '2'],
            'league': ['PL', 'PL'],
            'P_under_11_5': [0.7, 1.5]  # Invalid!
        })
        
        with pytest.raises(DataValidationError, match="Invalid probabilities"):
            evaluator._validate_logs(logs)


class TestLabelCalculation:
    """Test target label generation."""
    
    def test_calculate_target_labels_valid(self):
        """Correct target calculation from corners."""
        evaluator = CornerCalibrationEvaluator()
        
        matches = pd.DataFrame({
            'match_id': ['1', '2', '3'],
            'home_corners': [6, 5, 7],
            'away_corners': [4, 8, 3]
        })
        
        result = evaluator._calculate_target_labels(matches)
        
        # match 1: 6+4=10 <= 11.5 -> 1.0
        assert result.loc[0, 'total_corners'] == 10
        assert result.loc[0, 'target_u11_5'] == 1.0
        
        # match 2: 5+8=13 > 11.5 -> 0.0
        assert result.loc[1, 'total_corners'] == 13
        assert result.loc[1, 'target_u11_5'] == 0.0
    
    def test_calculate_target_labels_nan_handling(self):
        """NaN corners should produce NaN targets."""
        evaluator = CornerCalibrationEvaluator()
        
        matches = pd.DataFrame({
            'match_id': ['1', '2'],
            'home_corners': [6, np.nan],
            'away_corners': [4, 5]
        })
        
        result = evaluator._calculate_target_labels(matches)
        
        assert result.loc[0, 'target_u11_5'] == 1.0
        assert pd.isna(result.loc[1, 'target_u11_5'])


class TestCalibrationResult:
    """Test CalibrationResult dataclass."""
    
    def test_is_empty_true(self):
        """Empty metrics_df should return True."""
        result = CalibrationResult(
            metrics_df=pd.DataFrame(),
            total_matches=0,
            leagues_evaluated=0,
            output_file=None,
            timestamp=datetime.now()
        )
        assert result.is_empty
    
    def test_is_empty_false(self):
        """Non-empty metrics_df should return False."""
        result = CalibrationResult(
            metrics_df=pd.DataFrame({'a': [1]}),
            total_matches=100,
            leagues_evaluated=5,
            output_file=Path("test.csv"),
            timestamp=datetime.now()
        )
        assert not result.is_empty
    
    def test_summary_empty(self):
        """Summary for empty result."""
        result = CalibrationResult(
            metrics_df=pd.DataFrame(),
            total_matches=0,
            leagues_evaluated=0,
            output_file=None,
            timestamp=datetime.now()
        )
        assert "No data" in result.summary
    
    def test_summary_with_data(self):
        """Summary with data."""
        result = CalibrationResult(
            metrics_df=pd.DataFrame({'a': [1]}),
            total_matches=250,
            leagues_evaluated=5,
            output_file=Path("test.csv"),
            timestamp=datetime.now()
        )
        assert "5 leagues" in result.summary
        assert "250 matches" in result.summary

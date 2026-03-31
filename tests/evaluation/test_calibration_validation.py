"""
Calibration Validation Tests.

Tests that verify the calibration logic and ECE calculations
are mathematically correct and properly integrated.
"""
import pytest
import numpy as np
from typing import List

# Test imports
from src.strategies.drift_guard import DriftGuardrail
from src.strategies.selection_gate import SelectionGate


class TestDriftGuardrailCalibration:
    """Tests for DriftGuardrail ECE-based calibration checks."""
    
    def test_drift_guardrail_loads_baseline(self):
        """Verify that the drift guardrail has defined baselines."""
        guard = DriftGuardrail()
        
        assert 'hit_rate' in guard.BASELINES
        assert 'ece' in guard.BASELINES
        assert 'mean_conf' in guard.BASELINES
        
        # ECE baseline should be reasonable (< 0.10)
        assert guard.BASELINES['ece'] < 0.10
    
    def test_drift_guardrail_thresholds_defined(self):
        """Verify that stop thresholds are properly defined."""
        guard = DriftGuardrail()
        
        assert 'ece_stop' in guard.THRESHOLDS
        # ECE stop should trigger before ECE gets too high
        assert guard.THRESHOLDS['ece_stop'] <= 0.05
    
    def test_drift_check_returns_valid_status(self):
        """Verify check_drift returns a valid status string."""
        guard = DriftGuardrail()
        status = guard.check_drift()
        
        # Valid statuses: OK, WARN (allow proceed), STOP/FAIL/CRITICAL (block)
        assert status in ('OK', 'WARN', 'STOP', 'FAIL', 'CRITICAL')
    
    def test_drift_guard_fail_closed(self):
        """Verify that the guardrail fails closed (requires enforcement)."""
        guard = DriftGuardrail()
        
        # Default should require enforcement
        assert guard.requires_enforcement is True


class TestSelectionGateDriftIntegration:
    """Tests for Gate 4 (Market Drift) in SelectionGate."""
    
    def test_gate_4_exists_in_stats(self):
        """Verify Gate 4 is tracked in rejection stats."""
        gate = SelectionGate()
        
        # Empty predictions should have initialized stats
        _, stats = gate.process([])
        
        assert 'GATE_4_DRIFT_BLOCKED' in stats
    
    def test_gate_processes_with_drift_check(self):
        """Verify the gate doesn't crash when processing with drift check."""
        gate = SelectionGate()
        
        # Create a valid prediction
        predictions = [{
            'market': 'home_win',
            'probability': 0.80,
            'match_id': 'test_1'
        }]
        
        # Should not raise
        passed, stats = gate.process(predictions)
        
        # Either passes or gets rejected with proper reason
        assert stats['TOTAL_INPUT'] == 1


class TestCalibrationMathematics:
    """Tests for calibration formula correctness."""
    
    def test_ece_perfect_calibration(self):
        """ECE should be 0 for perfectly calibrated predictions."""
        # Perfect: predicted 0.7, actual 70% hit rate
        predictions = [0.7] * 100
        actuals = [1] * 70 + [0] * 30
        
        ece = self._calculate_ece(predictions, actuals, n_bins=10)
        
        assert ece < 0.02  # Some tolerance for binning
    
    def test_ece_poor_calibration(self):
        """ECE should be high for poorly calibrated predictions."""
        # Poor: predicted 0.9, actual 10% hit rate
        predictions = [0.9] * 100
        actuals = [1] * 10 + [0] * 90
        
        ece = self._calculate_ece(predictions, actuals, n_bins=10)
        
        assert ece > 0.70  # Should be high (0.80 gap)
    
    def _calculate_ece(
        self, 
        predictions: List[float], 
        actuals: List[int], 
        n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error."""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            in_bin = (predictions > bin_boundaries[i]) & (predictions <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = predictions[in_bin].mean()
                avg_accuracy = actuals[in_bin].mean()
                ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
        
        return ece


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

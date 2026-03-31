import pytest
from unittest.mock import Mock, patch
from src.guards.integrity_guard import MPIG, IntegrityError

class TestIntegrityGuard:
    @pytest.fixture
    def guard(self):
        with patch('src.guards.integrity_guard.ModelRegistry'), \
             patch('src.guards.integrity_guard.DriftGuardrail'):
            return MPIG()

    def test_missing_league_error(self, guard):
        matches = [{'match_id': '1', 'home_team': 'A', 'away_team': 'B'}] # No league
        with pytest.raises(IntegrityError, match="missing league identifier"):
            guard.verify_system(matches, ['test_market'], 'test_strategy')

    def test_missing_model_error(self, guard):
        matches = [{'league': 'PL', 'match_id': '1'}]
        # Mock registry to return None (no model)
        guard.registry.get_production_model_for_league.return_value = None
        
        with pytest.raises(IntegrityError, match=r"No production model for \[PL"):
            guard.verify_system(matches, ['test_market'], 'test_strategy')

    def test_state_reset_between_calls(self, guard):
        matches = [{'league': 'PL', 'match_id': '1'}]
        # Mock successful first call (but we'll force fail drift check)
        guard.registry.get_production_model_for_league.return_value = {
            'name': 'test',
            'registered_at': '2026-01-01',
            'train_size': 5000,
            'test_size': 1000,
            'metrics': {'calibration_score': 0.05}
        }
        guard.drift_guard.check_drift.return_value = "STOP"
        
        with pytest.raises(IntegrityError):
            guard.verify_system(matches, ['test_market'], 'test_strategy')
        
        # results["checks_passed"] should have ['model_exists', 'calibrated']
        assert guard.results["checks_passed"] == ['model_exists', 'calibrated']
        
        # Modify state manually to simulate leakage
        guard.results["checks_passed"].append("leaked_state")
        
        # Second call should reset it
        with pytest.raises(IntegrityError):
             guard.verify_system(matches, ['test_market'], 'test_strategy')
             
        assert "leaked_state" not in guard.results["checks_passed"]
        assert guard.results["checks_passed"] == ['model_exists', 'calibrated']

    def test_verified_pairs_tracking(self, guard):
        matches = [{'league': 'PL', 'match_id': '1'}]
        guard.registry.get_production_model_for_league.return_value = {
            'name': 'test',
            'registered_at': '2026-01-01',
            'train_size': 5000,
            'test_size': 1000,
            'metrics': {'calibration_score': 0.05}
        }
        guard.drift_guard.check_drift.return_value = "OK"
        
        results = guard.verify_system(matches, ['market_a', 'market_b'], 'test_strategy')
        
        assert results["verified"] is True
        assert len(results["verified_pairs"]) == 2
        assert ('PL', 'market_a') in results["verified_pairs"]
        assert ('PL', 'market_b') in results["verified_pairs"]
        assert results["phases"]["phase_1_models"] == "passed"

    def test_stale_model_date_object(self, guard):
        matches = [{'league': 'PL', 'match_id': '1'}]
        guard.registry.get_production_model_for_league.return_value = {
            'name': 'stale_model',
            'registered_at': '2025-11-20', # Before DEC 1st
            'train_size': 5000,
            'test_size': 1000,
            'metrics': {'calibration_score': 0.05}
        }
        with pytest.raises(IntegrityError, match="is stale"):
            guard.verify_system(matches, ['m'], 'strat')

    def test_forbidden_fruit_requirements(self, guard):
        # 1. Missing tier
        candidates = [{'match': 'A vs B'}] # No tier
        with pytest.raises(IntegrityError, match="requires tiered candidates"):
            guard.verify_strategy(candidates, "forbidden-fruit")
            
        # 2. Valid
        candidates = [{'match': 'A vs B', 'tier': 1}]
        result = guard.verify_strategy(candidates, "forbidden-fruit")
        assert result["verified"] is True
        assert "tier_presence" in result["checks_performed"]

    def test_accumulator_requirements(self, guard):
        # 1. Not enough candidates
        candidates = [{'match': 'A vs B', 'confidence': 0.7, 'market': 'BTTS'}]
        with pytest.raises(IntegrityError, match="requires at least 2 candidates"):
            guard.verify_strategy(candidates, "accumulator")
            
        # 2. Missing fields
        candidates = [
            {'match': 'A vs B', 'confidence': 0.7, 'market': 'BTTS'},
            {'match': 'C vs D'} # Missing fields
        ]
        with pytest.raises(IntegrityError, match="missing required fields"):
            guard.verify_strategy(candidates, "accumulator")
            
        # 3. Valid
        candidates = [
            {'match': 'A vs B', 'confidence': 0.7, 'market': 'BTTS'},
            {'match': 'C vs D', 'confidence': 0.6, 'market': 'O25'}
        ]
        result = guard.verify_strategy(candidates, "accumulator")
        assert result["verified"] is True
        assert "min_count" in result["checks_performed"]

    def test_empty_candidates_warning(self, guard):
        result = guard.verify_strategy([], "any-strategy")
        assert result["verified"] is True
        assert result["note"] == "No candidates to verify"

    def test_drift_caution(self, guard):
        matches = [{'league': 'PL', 'match_id': '1'}]
        guard.registry.get_production_model_for_league.return_value = {
            'name': 'test',
            'registered_at': '2026-01-01',
            'train_size': 5000,
            'test_size': 1000,
            'metrics': {'calibration_score': 0.05}
        }
        guard.drift_guard.check_drift.return_value = "CAUTION"
        guard.drift_guard.alerts = ["Mean Goals Shift: 12%"]
        
        result = guard.verify_system(matches, ['m'], 'strat')
        assert result["verified"] is True
        assert "drift_warnings" in result
        assert result["drift_warnings"] == ["Mean Goals Shift: 12%"]

    def test_dry_run_mode(self, guard):
        guard.dry_run = True
        matches = [{'league': 'PL', 'match_id': '1'}]
        # Mock failure (no model)
        guard.registry.get_production_model_for_league.return_value = None
        
        # Should NOT raise IntegrityError in dry-run
        result = guard.verify_system(matches, ['m'], 'strat')
        
        assert result["verified"] is False
        assert "would_block" in result
        assert "No production model for [PL" in result["would_block"]

    def test_audit_and_performance(self, guard):
        matches = [{'league': 'PL', 'match_id': '1'}]
        guard.registry.get_production_model_for_league.return_value = {
            'name': 'test',
            'registered_at': '2026-01-01',
            'train_size': 5000,
            'test_size': 1000,
            'metrics': {'calibration_score': 0.05}
        }
        guard.drift_guard.check_drift.return_value = "OK"
        
        result = guard.verify_system(matches, ['m'], 'strat')
        
        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0
        assert "audit_trail" in result
        assert len(result["audit_trail"]) >= 3
        assert result["audit_trail"][0]["phase"] == "phase_1_models"
        assert result["audit_trail"][0]["status"] == "passed"

    def test_miscalibrated_model(self, guard):
        matches = [{'league': 'PL', 'match_id': '1'}]
        guard.registry.get_production_model_for_league.return_value = {
            'name': 'bad_calib',
            'registered_at': '2026-01-01',
            'train_size': 5000,
            'test_size': 1000,
            'metrics': {'calibration_score': 0.15} # Above MAX_ECE (0.08)
        }
        with pytest.raises(IntegrityError, match="failed calibration gate"):
            guard.verify_system(matches, ['m'], 'strat')

    def test_insufficient_samples(self, guard):
        matches = [{'league': 'PL', 'match_id': '1'}]
        guard.registry.get_production_model_for_league.return_value = {
            'name': 'small_model',
            'registered_at': '2026-01-01',
            'train_size': 100,
            'test_size': 100,
            'metrics': {'calibration_score': 0.05}
        } # Total 200 < MIN_SAMPLES (1000)
        with pytest.raises(IntegrityError, match="failed sample size gate"):
            guard.verify_system(matches, ['m'], 'strat')

    def test_input_validation(self, guard):
        # 1. Non-list matches
        with pytest.raises(TypeError, match="matches must be list"):
            guard.verify_system("not-a-list", ['m'], 'strat')
        
        # 2. Empty matches
        with pytest.raises(ValueError, match="matches list cannot be empty"):
            guard.verify_system([], ['m'], 'strat')
            
        # 3. Invalid strategy
        with pytest.raises(ValueError, match="strategy_name must be non-empty string"):
            guard.verify_system([{'league': 'PL'}], ['m'], None)

    def test_audit_trail_reset(self, guard):
        matches = [{'league': 'PL', 'match_id': '1'}]
        guard.registry.get_production_model_for_league.return_value = {
            'name': 'test',
            'registered_at': '2026-01-01',
            'train_size': 5000,
            'test_size': 1000,
            'metrics': {'calibration_score': 0.05}
        }
        guard.drift_guard.check_drift.return_value = "OK"
        
        # First run
        guard.verify_system(matches, ['m'], 'strat1')
        audit_len_1 = len(guard.audit_trail)
        assert audit_len_1 > 0
        
        # Second run should reset audit trail
        guard.verify_system(matches, ['m'], 'strat2')
        assert len(guard.audit_trail) == audit_len_1 # Should be same length, not accumulated

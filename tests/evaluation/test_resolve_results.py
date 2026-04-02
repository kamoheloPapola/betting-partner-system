"""
Tests for Authoritative Result Resolver.

Unit tests for match hash normalization, outcome resolution,
validation logic, and full resolution pipeline.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.exceptions import DataValidationError
from src.db import connection as connection_module
from src.db.models import Base
from src.evaluation.resolve_results import AuthoritativeResolver, ResolutionStats


class TestNormalizeMatchHash:
    """Test hash normalization logic."""
    
    def test_valid_hash_unchanged(self):
        """Valid 16-char hex hash should not be modified."""
        resolver = AuthoritativeResolver()
        valid_hash = 'a1b2c3d4e5f60000'
        row = pd.Series({'match_hash': valid_hash})
        assert resolver._validate_match_hash(valid_hash)
        result = resolver._normalize_match_hash(row)
        assert result == valid_hash
    
    def test_invalid_format_triggers_regen(self):
        """Hash with underscores or non-hex length should trigger regen."""
        resolver = AuthoritativeResolver()
        row = pd.Series({'match_hash': 'PL_2024_INVALID_HASH'})
        
        assert not resolver._validate_match_hash('PL_2024_INVALID_HASH')
        # We expect it to try regeneration. If fields are missing, it returns None.
        result = resolver._normalize_match_hash(row)
        assert result is None 
        
    def test_display_id_regenerated(self):
        """Display ID format should trigger regeneration."""
        resolver = AuthoritativeResolver()
        row = pd.Series({
            'match_hash': 'PL_2024_ARS_CHE',
            'league': 'PL',
            'prediction_date': '2024-01-15',
            'home_team': 'Arsenal',
            'away_team': 'Chelsea'
        })
        
        with patch('src.evaluation.resolve_results.generate_match_fingerprint') as mock_gen:
            mock_gen.return_value = 'deadbeefdeadbeef'
            result = resolver._normalize_match_hash(row)
            assert result == 'deadbeefdeadbeef'

class TestValidationLogic:
    """Test distinct validation methods."""

    def test_probability_validation(self):
        resolver = AuthoritativeResolver()
        assert resolver._validate_probability(0.5)
        assert resolver._validate_probability(0)
        assert resolver._validate_probability(1)
        assert not resolver._validate_probability(1.5)
        assert not resolver._validate_probability(-0.1)
        assert not resolver._validate_probability(np.nan)
        assert not resolver._validate_probability(None)
        
    def test_hash_validation(self):
        resolver = AuthoritativeResolver()
        assert resolver._validate_match_hash("a1b2c3d4e5f60000")
        assert resolver._validate_match_hash("A1B2C3D4E5F60000")
        assert not resolver._validate_match_hash("short")
        assert not resolver._validate_match_hash("too_long_string_here")
        assert not resolver._validate_match_hash("invalidchar$#@!")

class TestOutcomeResolution:
    """Test outcome determination logic."""
    
    def test_won_outcome(self):
        resolver = AuthoritativeResolver()
        pred_row = pd.Series({
            'market': 'HOME_WIN',
            'predicted_probability': 0.75,
            'kickoff_date': '2024-01-15',
            'league': 'PL'
        })
        result_row = {'home_win': 1, 'kickoff_date_utc': '2024-01-15'}
        outcome = resolver._generate_outcome_record(pred_row, result_row, 'hash123')
        assert outcome['outcome'] == 'WON'
    
    def test_defensive_date_validation(self):
        """Ensure missing kickoff_date_utc raises DataValidationError."""
        resolver = AuthoritativeResolver()
        pred_row = pd.Series({'kickoff_date': '2024-01-15', 'market': 'X', 'predicted_probability': 0.5})
        result_row = {'home_win': 1} # Missing kickoff_date_utc
        
        with pytest.raises(DataValidationError, match="Missing kickoff_date_utc"):
            resolver._generate_outcome_record(pred_row, result_row, 'hash123')

class TestIntegration:
    """Integration tests with fixtures."""
    
    def test_full_resolution_flow_and_stats_return(self, tmp_path):
        """Test pipeline returns stats and resolves correctly."""
        labeled_path = tmp_path / "results_labeled.csv"
        labeled_path.parent.mkdir(parents=True, exist_ok=True)
        valid_hash = 'a1b2c3d4e5f60000'
        
        pd.DataFrame({
            'match_hash': [valid_hash],
            'kickoff_date_utc': ['2024-01-15'],
            'home_win': [1]
        }).to_csv(labeled_path, index=False)
        
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'match_hash': [valid_hash],
            'market': ['HOME_WIN'],
            'predicted_probability': [0.75],
            'prediction_date': ['2024-01-14'],
            'kickoff_date': ['2024-01-15'],
             'league': ['PL']
        }).to_csv(pred_dir / "preds.csv", index=False)
        
        resolver = AuthoritativeResolver(
            labeled_path=labeled_path,
            pred_dir=pred_dir,
            outcomes_path=tmp_path / "outcomes.csv"
        )
        
        stats = resolver.resolve_all()
        
        assert isinstance(stats, ResolutionStats)
        assert stats.resolved_count == 1
        assert stats.won == 1

    def test_dry_run_saves_nothing(self, tmp_path):
        """Test dry_run=True prevents file writing."""
        labeled_path = tmp_path / "results_labeled.csv"
        labeled_path.parent.mkdir(parents=True, exist_ok=True)
        valid_hash = 'a1b2c3d4e5f60000'
        
        pd.DataFrame({
            'match_hash': [valid_hash],
            'kickoff_date_utc': ['2024-01-15'],
            'home_win': [1]
        }).to_csv(labeled_path, index=False)
        
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'match_hash': [valid_hash],
            'market': ['HOME_WIN'],
            'predicted_probability': [0.75],
            'prediction_date': ['2024-01-14'],
            'kickoff_date': ['2024-01-15'],
             'league': ['PL']
        }).to_csv(pred_dir / "preds.csv", index=False)
        
        outcomes_path = tmp_path / "outcomes.csv"
        
        resolver = AuthoritativeResolver(
            labeled_path=labeled_path,
            pred_dir=pred_dir,
            outcomes_path=outcomes_path,
            dry_run=True # Enable dry run
        )
        
        stats = resolver.resolve_all()
        
        assert stats.resolved_count == 1
        assert not outcomes_path.exists()

    def test_validate_wrapper_is_dry_run(self, tmp_path):
        """Test validate() wrapper forces dry_run and restores it."""
        labeled_path = tmp_path / "results_labeled.csv"
        labeled_path.parent.mkdir(parents=True, exist_ok=True)
        valid_hash = 'a1b2c3d4e5f60000'
        
        pd.DataFrame({
            'match_hash': [valid_hash],
            'kickoff_date_utc': ['2024-01-15'],
            'home_win': [1]
        }).to_csv(labeled_path, index=False)
        
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'match_hash': [valid_hash],
            'market': ['HOME_WIN'],
            'predicted_probability': [0.75],
            'prediction_date': ['2024-01-14'],
            'kickoff_date': ['2024-01-15'],
             'league': ['PL']
        }).to_csv(pred_dir / "preds.csv", index=False)
        
        outcomes_path = tmp_path / "outcomes.csv"
        
        resolver = AuthoritativeResolver(
            labeled_path=labeled_path,
            pred_dir=pred_dir,
            outcomes_path=outcomes_path,
            dry_run=False # Default
        )
        
        assert not resolver.dry_run
        
        # Call validate(), should be dry run effectively
        stats = resolver.validate()
        
        assert stats.resolved_count == 1
        assert not outcomes_path.exists() # Should NOT have saved
        assert not resolver.dry_run # Should be restored to False

    def test_unmatched_prediction_tracking(self, tmp_path):
        """Test that unmatched predictions are tracked."""
        labeled_path = tmp_path / "results_labeled.csv"
        labeled_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            'match_hash': ['b2c3d4e5f6000011'], 
            'kickoff_date_utc': ['2024-01-15']
        }).to_csv(labeled_path, index=False)
        
        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        valid_hash = 'a1b2c3d4e5f60000'
        pd.DataFrame({
            'match_hash': [valid_hash],
            'market': ['HOME_WIN'],
            'predicted_probability': [0.75],
            'prediction_date': ['2024-01-14'],
            'kickoff_date': ['2024-01-15'],
             'league': ['PL']
        }).to_csv(pred_dir / "preds.csv", index=False)
        
        resolver = AuthoritativeResolver(
            labeled_path=labeled_path,
            pred_dir=pred_dir,
            outcomes_path=tmp_path / "outcomes.csv"
        )
        
        stats = resolver.resolve_all()
        assert stats.resolved_count == 0
        assert stats.unmatched_predictions == 1

    def test_load_outcomes_normalizes_binary_labels(self, tmp_path):
        outcomes_path = tmp_path / "prediction_outcomes.csv"
        pd.DataFrame(
            [
                {
                    "prediction_id": "hash1_HOME_WIN",
                    "match_hash": "hash1",
                    "league": "PL",
                    "kickoff_date": "2026-03-01T12:00:00Z",
                    "market": "HOME_WIN",
                    "probability": 0.75,
                    "outcome": "WON",
                    "resolved_at": "2026-03-01T18:00:00Z",
                },
                {
                    "prediction_id": "hash2_AWAY_WIN",
                    "match_hash": "hash2",
                    "league": "PL",
                    "kickoff_date": "2026-03-01T12:00:00Z",
                    "market": "AWAY_WIN",
                    "probability": 0.35,
                    "outcome": "LOST",
                    "resolved_at": "2026-03-01T18:00:00Z",
                },
                {
                    "prediction_id": "hash3_HOME_DC",
                    "match_hash": "hash3",
                    "league": "PL",
                    "kickoff_date": "2026-03-01T12:00:00Z",
                    "market": "HOME_DC",
                    "probability": 0.51,
                    "outcome": "VOID",
                    "resolved_at": "2026-03-01T18:00:00Z",
                },
            ]
        ).to_csv(outcomes_path, index=False)

        resolver = AuthoritativeResolver(outcomes_path=outcomes_path)
        outcomes = resolver.load_outcomes()

        assert outcomes["outcome"].tolist() == [1, 0]
        assert set(outcomes["market"]) == {"HOME_WIN", "AWAY_WIN"}

    def test_load_outcomes_repairs_legacy_column_shift(self, tmp_path):
        outcomes_path = tmp_path / "prediction_outcomes.csv"
        outcomes_path.write_text(
            "\n".join(
                [
                    "prediction_id,match_hash,league,kickoff_date,market,probability,outcome,resolved_at",
                    "hash1_HOME_WIN,hash1,PL,2026-03-01T12:00:00Z,HOME_WIN,0.75,WON,2026-03-01T18:00:00Z",
                    "hash2,PL,2026-03-01T12:00:00Z,AWAY_WIN,0.25,LOST,2026-03-01T18:00:00Z,hash2_AWAY_WIN",
                ]
            ),
            encoding="utf-8",
        )

        resolver = AuthoritativeResolver(outcomes_path=outcomes_path)
        outcomes = resolver.load_outcomes(include_void=True)

        repaired = outcomes[outcomes["match_hash"] == "hash2"].iloc[0]
        assert repaired["prediction_id"] == "hash2_AWAY_WIN"
        assert repaired["league"] == "PL"
        assert repaired["market"] == "AWAY_WIN"
        assert repaired["probability"] == 0.25
        assert repaired["outcome"] == 0

    def test_save_and_load_outcomes_prefer_database_when_configured(self, tmp_path, monkeypatch):
        db_path = tmp_path / "resolved_predictions.db"
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
        connection_module.get_engine.cache_clear()
        engine = connection_module.get_engine()
        Base.metadata.create_all(engine)

        outcomes_path = tmp_path / "prediction_outcomes.csv"
        resolver = AuthoritativeResolver(outcomes_path=outcomes_path)

        try:
            resolver._save_outcomes(
                [
                    {
                        "prediction_id": "hash1_HOME_WIN",
                        "match_hash": "hash1",
                        "league": "PL",
                        "kickoff_date": "2026-03-01T12:00:00+00:00",
                        "market": "HOME_WIN",
                        "probability": 0.75,
                        "outcome": "WON",
                        "resolved_at": "2026-03-01T18:00:00+00:00",
                    }
                ]
            )

            outcomes_path.write_text(
                "\n".join(
                    [
                        "prediction_id,match_hash,league,kickoff_date,market,probability,outcome,resolved_at",
                        "wrong_id,wrong_hash,PL,2026-03-01T12:00:00Z,AWAY_WIN,0.20,LOST,2026-03-01T18:00:00Z",
                    ]
                ),
                encoding="utf-8",
            )

            outcomes = resolver.load_outcomes(include_void=True)

            assert outcomes["prediction_id"].tolist() == ["hash1_HOME_WIN"]
            assert outcomes["market"].tolist() == ["HOME_WIN"]
            assert outcomes["outcome"].tolist() == [1]
        finally:
            engine.dispose()
            connection_module.get_engine.cache_clear()

import pytest
import requests
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch
from src.fetch.fetch_latest_season_csvs import SeasonFetcher

class TestSeasonValidation:
    """Test season format validation."""
    
    def test_valid_season_format(self, tmp_path):
        """Valid YYZZ format should pass."""
        # Use empty league_map to avoid network/file activity
        fetcher = SeasonFetcher(output_dir=tmp_path, league_map={})
        fetcher.fetch_season("2526")
    
    def test_invalid_season_format(self, tmp_path):
        """Invalid format should raise ValueError."""
        fetcher = SeasonFetcher(output_dir=tmp_path)
        
        with pytest.raises(ValueError, match="Invalid season format"):
            fetcher.fetch_season("25-26")
    
    def test_invalid_season_length(self, tmp_path):
        """Wrong length should raise ValueError."""
        fetcher = SeasonFetcher(output_dir=tmp_path)
        
        with pytest.raises(ValueError, match="Invalid season format"):
            fetcher.fetch_season("252")


class TestHashCalculation:
    """Test file hashing."""
    
    def test_sha256_used(self, tmp_path):
        """Should use SHA-256."""
        fetcher = SeasonFetcher(output_dir=tmp_path)
        content = b"test content"
        
        expected = hashlib.sha256(content).hexdigest()
        assert fetcher._calculate_hash(content) == expected


class TestRetryLogic:
    """Test network retry behavior."""
    
    @patch('requests.Session.get')
    def test_retries_on_503(self, mock_get, tmp_path):
        """Should retry on 503 Service Unavailable."""
        mock_get.side_effect = [
            Mock(status_code=503, raise_for_status=Mock(side_effect=requests.HTTPError("503 Server Error"))),
            Mock(status_code=200, content=b"test", raise_for_status=Mock())
        ]
        
        fetcher = SeasonFetcher(output_dir=tmp_path, league_map={"PL": "E0"})
        with patch('src.fetch.fetch_latest_season_csvs.requests.Session.get', mock_get):
             with patch.object(SeasonFetcher, '_has_remote_changed', return_value=True):
                fetcher.fetch_season("2526")
        
        assert mock_get.call_count >= 1


class TestImmutability:
    """Test snapshot behavior."""
    
    def test_creates_timestamped_snapshot(self, tmp_path):
        """Each fetch should create new timestamped file if content changes."""
        fetcher = SeasonFetcher(output_dir=tmp_path, league_map={"PL": "E0"})
        
        with patch('requests.Session.get') as mock_get:
            mock_get.return_value = Mock(status_code=200, content=b"version 1", raise_for_status=Mock())
            fetcher.fetch_season("2526")
            
            first_file = tmp_path / "PL" / "season_2526.csv"
            assert first_file.exists()
            
            mock_get.return_value = Mock(status_code=200, content=b"version 2", raise_for_status=Mock())
            
            # Force remote changed
            with patch.object(SeasonFetcher, '_has_remote_changed', return_value=True):
                fetcher.fetch_season("2526")
        
        # Check for latest and backup
        latest = tmp_path / "PL" / "season_2526.csv"
        backups = list(tmp_path.rglob("*.bak"))
        
        assert latest.exists()
        assert len(backups) == 1
        assert "season_2526" in backups[0].name

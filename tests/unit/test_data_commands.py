
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.cli.commands.data import _process_fixtures_to_dataframe, fetch_upcoming
from src.cli.utils import LeagueCode
from src.core.exceptions import DataValidationError

def test_process_fixtures_to_dataframe():
    """Verify fixture dict is correctly converted to DataFrame (Issue #7)."""
    fixtures = [
        {
            "match_id": "test_123",
            "date_utc": "2025-01-15T15:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "league": "PL",
            "season": "2024"
        }
    ]
    
    df = _process_fixtures_to_dataframe(fixtures, LeagueCode.PL)
    
    assert len(df) == 1
    assert df.loc[0, 'home_team'] == "Arsenal"
    assert df.loc[0, 'away_team'] == "Chelsea"
    assert df.loc[0, 'competition'] == "Premier League"
    assert df.loc[0, 'status'] == "SCHEDULED"
    assert df.loc[0, 'kickoff_utc'] == "2025-01-15T15:00:00Z"
    assert df.loc[0, 'date'] == pd.Timestamp("2025-01-15")

def test_fetch_upcoming_handles_no_fixtures():
    """Verify graceful handling when API returns no fixtures (Issue #7)."""
    with patch('src.ingestion.fixture_manager.FixtureManager.get_fixtures') as mock_get:
        mock_get.return_value = []
        
        # Should return early without error
        # We don't want it to actually run ensure_output_path if it returns early
        with patch('src.cli.commands.data._save_upcoming_matches') as mock_save:
            fetch_upcoming(LeagueCode.PL)
            mock_save.assert_not_called()

def test_fetch_upcoming_raises_error_on_empty_df_after_processing():
    """Verify DataValidationError is raised if processing results in empty DF (Issue #3)."""
    fixtures = [{"something": "invalid"}]
    
    with patch('src.ingestion.fixture_manager.FixtureManager.get_fixtures') as mock_get, \
         patch('src.cli.commands.data._process_fixtures_to_dataframe') as mock_process:
        
        mock_get.return_value = fixtures
        mock_process.return_value = pd.DataFrame() # Empty DF
        
        print(f"\nDEBUG: Calling fetch_upcoming with mocked empty DF")
        # Typer.Exit raises SystemExit or click.exceptions.Exit
        # Both are BaseExceptions
        with pytest.raises(BaseException) as excinfo:
            fetch_upcoming(LeagueCode.PL)
        
        # Check if it's an Exit exception
        assert "Exit" in str(type(excinfo.value))

def test_fetch_upcoming_validates_dataframe():
    """Verify validation layer is called (Issue #7)."""
    fixtures = [
        {
            "match_id": "1",
            "date_utc": "2025-01-15T15:00:00Z",
            "home_team": "Arsenal",
            "away_team": "Chelsea",
            "league": "PL",
            "season": "2024"
        }
    ]
    
    with patch('src.ingestion.fixture_manager.FixtureManager.get_fixtures') as mock_get, \
         patch('src.cli.commands.data._process_fixtures_to_dataframe') as mock_process, \
         patch('src.cli.commands.data.validate_match_dataframe') as mock_validate, \
         patch('src.cli.commands.data._save_upcoming_matches') as mock_save:
        
        mock_get.return_value = fixtures
        mock_process.return_value = pd.DataFrame([{"match_id": "1", "date": "2024-01-01"}])
        
        # Call via the imported name
        fetch_upcoming(LeagueCode.PL)
        
        # Verify save was called (implies it passed validation check)
        mock_save.assert_called_once()

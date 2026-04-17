
import pytest
import pandas as pd
from click.exceptions import Exit
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
    """
    When the API returns no fixtures, fetch_upcoming must return without error
    and must not attempt to save anything.

    The meaningful behaviour is the absence of a save side effect - but we also
    assert the function completes (does not raise), which the original test did
    not explicitly cover.
    """
    with patch("src.ingestion.fixture_manager.FixtureManager.get_fixtures", return_value=[]):
        with patch("src.cli.commands.data._save_upcoming_matches") as mock_save:
            # Must not raise
            fetch_upcoming(LeagueCode.PL)

    # Nothing should have been persisted
    mock_save.assert_not_called()

def test_fetch_upcoming_raises_error_on_empty_df_after_processing():
    """
    When fixture data processes to an empty DataFrame, fetch_upcoming must
    exit with a non-zero code rather than silently succeeding.
    """
    with patch("src.ingestion.fixture_manager.FixtureManager.get_fixtures", return_value=[{"something": "invalid"}]), \
         patch("src.cli.commands.data._process_fixtures_to_dataframe", return_value=pd.DataFrame()):
        with pytest.raises(Exit) as excinfo:
            fetch_upcoming(LeagueCode.PL)

    assert excinfo.value.exit_code == 1

def test_fetch_upcoming_validates_dataframe():
    """
    fetch_upcoming must pass the processed DataFrame through validate_match_dataframe
    before saving, and must save exactly the DataFrame that was produced by processing.

    Asserts:
    - validate_match_dataframe is called with the processed DataFrame
    - _save_upcoming_matches is called with that same DataFrame (not a copy or subset)
    """
    processed_df = pd.DataFrame([{"match_id": "1", "date": "2024-01-01"}])

    with patch("src.ingestion.fixture_manager.FixtureManager.get_fixtures") as mock_get, \
         patch("src.cli.commands.data._process_fixtures_to_dataframe", return_value=processed_df), \
         patch("src.cli.commands.data.validate_match_dataframe") as mock_validate, \
         patch("src.cli.commands.data._save_upcoming_matches") as mock_save:

        mock_get.return_value = [
            {
                "match_id": "1",
                "date_utc": "2025-01-15T15:00:00Z",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "league": "PL",
                "season": "2024",
            }
        ]

        fetch_upcoming(LeagueCode.PL, skip_validation=False)

    # Validation must have been called with the processed DataFrame
    mock_validate.assert_called_once()
    validated_df = mock_validate.call_args[0][0]
    assert validated_df is processed_df, (
        "validate_match_dataframe must receive the processed DataFrame"
    )
    assert mock_validate.call_args.kwargs == {"context": "fetch_upcoming"}

    # Save must have been called with the same DataFrame
    mock_save.assert_called_once()
    saved_df = mock_save.call_args[0][0]
    saved_league = mock_save.call_args[0][1]
    assert saved_df is processed_df, (
        "_save_upcoming_matches must receive the validated DataFrame, not a different object"
    )
    assert saved_league == LeagueCode.PL

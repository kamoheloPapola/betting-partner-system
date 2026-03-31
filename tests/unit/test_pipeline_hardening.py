
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.features.pipeline import FeaturePipeline
from src.core.exceptions import DataValidationError

def test_pipeline_invalid_league_validation():
    pipeline = FeaturePipeline()
    # Should allow None
    # pipeline.run(league=None) # skipping execution, just checking validation logic if possible without files
    
    # Should raise ValueError for invalid league
    with pytest.raises(ValueError, match="Invalid league"):
        pipeline.run(league="INVALID_LEAGUE_CODE")

def test_static_priors_existence():
    assert FeaturePipeline.StaticPriors.ROLLING_GOALS == 1.35
    assert FeaturePipeline.StaticPriors.ROLLING_CORNERS == 4.5
    
def test_valid_leagues_set():
    assert 'PL' in FeaturePipeline.VALID_LEAGUES
    assert 'INVALID' not in FeaturePipeline.VALID_LEAGUES

@patch('src.features.pipeline.pd.read_csv')
@patch('src.features.pipeline.Path.glob')
def test_quarantine_logic(mock_glob, mock_read_csv):
    # Setup
    pipeline = FeaturePipeline()
    mock_file = MagicMock()
    mock_file.name = "PL_2024.csv"
    mock_file.stem = "PL_2024"
    mock_glob.return_value = [mock_file]
    
    # Create DF with invalid date
    df = pd.DataFrame({
        'match_id': ['1', '2'],
        'date': ['2024-01-01', None], # One valid, one invalid
        'home_team': ['A', 'B'], 
        'away_team': ['C', 'D'],
        'score': ['1-0', '1-1'],
        'status': ['FT', 'FT'] # Required for dedup logic
    })
    mock_read_csv.return_value = df
    
    # Run load_stored_matches
    # We need to mock PROCESSED_DATA_DIR existence checks too
    with patch('src.features.pipeline.PROCESSED_DATA_DIR') as mock_dir:
        mock_dir.__truediv__.return_value.exists.return_value = True
        mock_dir.__truediv__.return_value.glob.return_value = [mock_file]
        
        # We also need to mock quarantine dir creation logic or it might fail on disk permission in test env
        # But let's see if simple execution works.
        # Actually, writing to disk 'to_csv' should be mocked to avoid clutter.
        
        result_df = pipeline.load_stored_matches(league='PL')
        
    # Check that invalid row was dropped
    assert len(result_df) == 1
    assert result_df.iloc[0]['match_id'] == '1'

def test_feature_columns_whitelist():
    # Test Basic
    cols = FeaturePipeline.FeatureColumns.get_home_columns()
    assert 'rolling_goals_scored_3' in cols
    assert 'rolling_xg_scored_3' not in cols
    
    # Test Toggles
    cols_xg = FeaturePipeline.FeatureColumns.get_home_columns(include_xg=True)
    assert 'rolling_xg_scored_3' in cols_xg
    
    cols_all = FeaturePipeline.FeatureColumns.get_home_columns(include_xg=True, include_corners=True, include_cards=True)
    assert 'rolling_corners_scored_5' in cols_all
    assert 'rolling_cards_conceded_10' in cols_all

def test_pipeline_dependency_injection():
    # Define a mock loader
    mock_data = pd.DataFrame({
        'match_id': ['100', '101'],
        'date': ['2024-01-01', '2024-01-08'],
        'home_team': ['A', 'C'],
        'away_team': ['B', 'D'],
        'home_score': [1, 2],
        'away_score': [1, 0],
        'status': ['FT', 'FT']
    })
    
    def mock_loader(league=None):
        return mock_data
        
    # Inject loader
    pipeline = FeaturePipeline(data_loader=mock_loader)
    
    # Run pipeline (should use mock_loader instead of disk)
    # run() calls _load_and_process() -> load_stored_matches() -> data_loader()
    result = pipeline.run(league='PL')
    
    assert len(result) == 2
    assert result.iloc[0]['match_id'] == '100'
    # Verify processing happened (cols added and renamed)
    # Note: features are prefixed with home_/away_ in final output
    assert 'home_form_rating' in result.columns
    assert 'referee_card_rate_10' in result.columns
    assert 'precipitation_mm' in result.columns
    assert 'wind_speed_kmh' in result.columns
    assert 'home_team_availability_pct' in result.columns
    assert 'away_team_availability_pct' in result.columns


def test_master_feature_rotation_keeps_latest_three(tmp_path):
    pipeline = FeaturePipeline()
    names = [
        "master_features_20250101.csv",
        "master_features_20250102.csv",
        "master_features_20250103.csv",
        "master_features_20250104.csv",
        "master_features_20250105.csv",
    ]
    for name in names:
        (tmp_path / name).write_text("match_id\n1\n", encoding="utf-8")

    with patch("src.features.pipeline.DATA_DIR", tmp_path):
        pipeline._rotate_master_feature_exports()

    remaining = sorted(p.name for p in tmp_path.glob("master_features_*.csv"))
    assert remaining == [
        "master_features_20250103.csv",
        "master_features_20250104.csv",
        "master_features_20250105.csv",
    ]


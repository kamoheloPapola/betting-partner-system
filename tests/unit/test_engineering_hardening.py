import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer

def test_calculate_rolling_stats_strict_validation():
    engineer = FeatureEngineer()
    
    # Valid data
    df_valid = pd.DataFrame({
        'team_id': [1, 1, 1],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'goals_scored': [1, 2, 3],
        'goals_conceded': [0, 1, 0],
        'cards_scored': [1, 0, 2],
        'cards_conceded': [0, 1, 1]
    })
    
    # Should pass
    result = engineer.calculate_rolling_stats(df_valid, metrics=['goals', 'cards'])
    assert 'rolling_goals_scored_5' in result.columns
    assert 'rolling_cards_scored_5' in result.columns
    
    # Invalid data (missing goals_conceded)
    df_invalid = pd.DataFrame({
        'team_id': [1, 1, 1],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'goals_scored': [1, 2, 3]
    })
    
    with pytest.raises(ValueError) as exc:
        engineer.calculate_rolling_stats(df_invalid, metrics=['goals'])
    
    assert "Missing required columns" in str(exc.value)
    assert "goals_conceded" in str(exc.value)

def test_calculate_form_upcoming_and_new_teams():
    engineer = FeatureEngineer()
    
    # Mix of historical and upcoming
    df = pd.DataFrame({
        'team_id': [1, 1, 1, 2], # Team 1 has history, Team 2 is new
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-03']),
        'goals_scored': [3, 0, np.nan, np.nan], # Last match for 1 is upcoming, 2 is entirely upcoming
        'goals_conceded': [0, 2, np.nan, np.nan]
    })
    
    result = engineer.calculate_form(df)
    
    # Team 1: 
    # Index 0: No history -> fillna(1.5)
    # Index 1: History is [3] -> shift(1) gives [NaN, 3]. ewm mean is [NaN, 3].
    # Index 2 (Upcoming): Shifting gives [NaN, 3, 0]. ewm mean gives [NaN, 3, (3*0.9 + 0*0.1)].
    # Actually shift(1) gives:
    # row 0: points=3, shift=NaN
    # row 1: points=0, shift=3
    # row 2: points=NaN, shift=0
    # EWM on [NaN, 3, 0] -> [NaN, 3.0, 3.0*0.9 + 0.0*0.1 = 2.7]
    # Then ffill for the NaN at the end.
    
    # Team 2:
    # Only one row, points=NaN, shift=NaN -> EWM yields NaN -> fillna(1.5)
    
    assert result.loc[0, 'form_rating'] == 1.5
    assert result.loc[1, 'form_rating'] == 3.0
    assert result.loc[2, 'form_rating'] == 2.7
    assert result.loc[3, 'form_rating'] == 1.5

def test_calculate_rest_days_clipping():
    engineer = FeatureEngineer()
    
    df = pd.DataFrame({
        'team_id': [1, 1, 1],
        'date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-02-01']) # 7 days, then 24 days
    })
    
    result = engineer.calculate_rest_days(df)
    
    # 0.7.3: Default for first match
    assert result.loc[0, 'days_rest'] == 7
    # 0.7.3: Normal interval
    assert result.loc[1, 'days_rest'] == 7
    # 0.7.3: Long interval capped at 21
    assert result.loc[2, 'days_rest'] == 21

def test_create_double_chance_targets():
    engineer = FeatureEngineer()
    
    df = pd.DataFrame({
        'home_score': [2, 0, 1, np.nan],
        'away_score': [1, 2, 1, np.nan]
    })
    
    result = engineer.create_double_chance_targets(df)
    
    # row 0: 2-1 (Home Win)
    assert result.loc[0, 'target_1x'] == 1 # Win/Draw
    assert result.loc[0, 'target_x2'] == 0 # Away Win/Draw
    assert result.loc[0, 'target_12'] == 1 # Either win
    
    # row 1: 0-2 (Away Win)
    assert result.loc[1, 'target_1x'] == 0
    assert result.loc[1, 'target_x2'] == 1
    assert result.loc[1, 'target_12'] == 1
    
    # row 2: 1-1 (Draw)
    assert result.loc[2, 'target_1x'] == 1
    assert result.loc[2, 'target_x2'] == 1
    assert result.loc[2, 'target_12'] == 0 # No winner
    
    # row 3: Upcoming (NaN)
    assert pd.isna(result.loc[3, 'target_1x'])
    assert pd.isna(result.loc[3, 'target_x2'])
    assert pd.isna(result.loc[3, 'target_12'])

def test_normalize_column_names():
    engineer = FeatureEngineer()
    
    df = pd.DataFrame({
        'home_goals': [2, 0],
        'away_goals': [1, 2],
        'match_hash': ['h1', 'h2'],
        'match_date': pd.to_datetime(['2024-01-01', '2024-01-02'])
    })
    
    result = engineer.normalize_column_names(df)
    
    assert 'home_score' in result.columns
    assert 'away_score' in result.columns
    assert 'match_id' in result.columns
    assert 'date' in result.columns
    assert 'home_goals' not in result.columns
    assert result.loc[0, 'home_score'] == 2

def test_pure_functions():
    """Verify that transformation methods do not mutate the input DataFrame."""
    engineer = FeatureEngineer()
    
    df = pd.DataFrame({
        'team_id': [1, 1],
        'date': pd.to_datetime(['2024-01-01', '2024-01-08']),
        'goals_scored': [2, 1],
        'goals_conceded': [0, 1]
    })
    
    df_orig = df.copy()
    
    # Test calculate_rolling_stats
    _ = engineer.calculate_rolling_stats(df, metrics=['goals'])
    pd.testing.assert_frame_equal(df, df_orig)
    
    # Test calculate_form
    _ = engineer.calculate_form(df)
    pd.testing.assert_frame_equal(df, df_orig)
    
    # Test calculate_rest_days
    _ = engineer.calculate_rest_days(df)
    pd.testing.assert_frame_equal(df, df_orig)

def test_calculate_rolling_stats_xg_validation():
    engineer = FeatureEngineer()
    
    df = pd.DataFrame({
        'team_id': [1, 1, 1],
        'date': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'goals_scored': [1, 2, 3],
        'goals_conceded': [0, 1, 0]
    })
    
    # xG is missing, should raise error if requested
    with pytest.raises(ValueError) as exc:
        engineer.calculate_rolling_stats(df, metrics=['goals', 'xg'])
    
    assert "xg_scored" in str(exc.value)
    assert "xg_conceded" in str(exc.value)

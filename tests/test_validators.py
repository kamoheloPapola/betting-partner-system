
import pytest
import pandas as pd
import numpy as np
from src.core.validators import validate_match_dataframe
from src.core.exceptions import DataValidationError

def test_validate_valid_dataframe():
    data = {
        'match_id': ['1', '2'],
        'date': pd.to_datetime(['2023-01-01', '2023-01-02']),
        'home_team': ['A', 'B'],
        'away_team': ['C', 'D'],
        'league': ['PL', 'PL']
    }
    df = pd.DataFrame(data)
    # Should not raise
    validate_match_dataframe(df)

def test_validate_missing_columns():
    data = {'match_id': ['1']}
    df = pd.DataFrame(data)
    with pytest.raises(DataValidationError) as exc:
        validate_match_dataframe(df)
    assert "Missing required columns" in str(exc.value)

def test_validate_invalid_date_type():
    data = {
        'match_id': ['1'],
        'date': ['2023-01-01'], # String, not datetime
        'home_team': ['A'],
        'away_team': ['B'],
        'league': ['PL']
    }
    df = pd.DataFrame(data)
    with pytest.raises(DataValidationError) as exc:
        validate_match_dataframe(df)
    assert "'date' column must be datetime" in str(exc.value)

def test_validate_settled_missing_score():
    data = {
        'match_id': ['1'],
        'date': pd.to_datetime(['2023-01-01']),
        'home_team': ['A'],
        'away_team': ['B'],
        'league': ['PL']
        # No home_score
    }
    df = pd.DataFrame(data)
    with pytest.raises(DataValidationError) as exc:
        validate_match_dataframe(df, require_settled=True)
    assert "required column" in str(exc.value) or "missing scores" in str(exc.value) or "require 'home_score'" in str(exc.value)

def test_validate_settled_nan_score():
    data = {
        'match_id': ['1'],
        'date': pd.to_datetime(['2023-01-01']),
        'home_team': ['A'],
        'away_team': ['B'],
        'league': ['PL'],
        'home_score': [np.nan]
    }
    df = pd.DataFrame(data)
    with pytest.raises(DataValidationError) as exc:
        validate_match_dataframe(df, require_settled=True)
    assert "missing scores" in str(exc.value)

def test_validate_empty_strings():
    data = {
        'match_id': ['1'],
        'date': pd.to_datetime(['2023-01-01']),
        'home_team': [''], # Empty
        'away_team': ['B'],
        'league': ['PL']
    }
    df = pd.DataFrame(data)
    with pytest.raises(DataValidationError) as exc:
        validate_match_dataframe(df)
    assert "empty strings" in str(exc.value)

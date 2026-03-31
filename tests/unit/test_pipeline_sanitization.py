import pytest
import pandas as pd
import numpy as np
from src.features.pipeline import FeaturePipeline
from src.core.exceptions import DataValidationError

class TestPipelineSanitization:
    @pytest.fixture
    def pipeline(self):
        return FeaturePipeline()

    def test_whitespace_trimming(self, pipeline):
        df = pd.DataFrame({
            'match_id': [' 1 '],
            'date': ['2023-01-01'],
            'home_team': [' Team A '],
            'away_team': ['Team B'],
            'status': ['FT']
        })
        sanitized = pipeline.process_matches(df)
        assert sanitized.loc[0, 'match_id'] == '1'
        assert sanitized.loc[0, 'home_team'] == 'TEAM A'

    def test_control_character_removal(self, pipeline):
        # \x07 is 'bell' control char
        df = pd.DataFrame({
            'match_id': ['1\x07'],
            'date': ['2023-01-01'],
            'home_team': ['Team A'],
            'away_team': ['Team B'],
            'status': ['FT']
        })
        sanitized = pipeline.process_matches(df)
        assert sanitized.loc[0, 'match_id'] == '1'

    def test_invalid_identifier_raises_error(self, pipeline):
        df = pd.DataFrame({
            'match_id': ['1; DROP TABLE matches;'],
            'date': ['2023-01-01'],
            'home_team': ['Team A'],
            'away_team': ['Team B'],
            'status': ['FT']
        })
        with pytest.raises(DataValidationError, match="Invalid characters found"):
            pipeline.process_matches(df)

    def test_mixed_types_bypass(self, pipeline):
        # Ensure it doesn't break on numeric cols
        df = pd.DataFrame({
            'match_id': ['1'],
            'date': ['2023-01-01'],
            'home_team': ['Team A'],
            'away_team': ['Team B'],
            'home_score': [2],
            'away_score': [1]
        })
        sanitized = pipeline.process_matches(df)
        assert sanitized.loc[0, 'home_score'] == 2

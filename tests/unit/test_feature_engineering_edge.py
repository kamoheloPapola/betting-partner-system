import pytest
import pandas as pd
import numpy as np
import logging
from dataclasses import FrozenInstanceError
from unittest.mock import Mock
from src.features.engineering import FeatureEngineer, FeatureDefaults

class TestFeatureEngineeringEdgeCases:
    
    @pytest.fixture
    def engineer(self):
        return FeatureEngineer()
    
    @pytest.fixture
    def leakage_df(self):
        """History: Match 1 (Played), Match 2 (Upcoming), Match 3 (Upcoming)"""
        return pd.DataFrame({
            'team_id': ['A', 'A', 'A'],
            'date': pd.to_datetime(['2023-01-01', '2023-01-08', '2023-01-15']),
            'match_id': ['1', '2', '3'],
            'goals_scored': [2.0, np.nan, np.nan], # Match 1 result known. 2 & 3 upcoming.
            'goals_conceded': [1.0, np.nan, np.nan]
        })

    def test_feature_defaults_immutability(self):
        """Ensure specific config values cannot be changed at runtime."""
        with pytest.raises(FrozenInstanceError):
            FeatureDefaults.DAYS_REST_DEFAULT = 10
            
    def test_input_sanitization_error(self, engineer):
        """Verify ValueError on suspicious home_team."""
        df = pd.DataFrame({
            'match_id': ['1'],
            'date': pd.to_datetime(['2023-01-01']),
            'home_team': ['Team<script>'],
            'away_team': ['TeamNormal'],
            'home_score': [1], 'away_score': [0]
        })
        
        with pytest.raises(ValueError, match="Invalid characters in home_team"):
            engineer.transform_match_to_team_rows(df)

    def test_negative_rest_days_validation(self, engineer):
        """Verify validate_features raises ValueError for negative rest."""
        df = pd.DataFrame({
            'team_id': ['A'],
            'days_rest': [-1]
        })
        
        with pytest.raises(ValueError, match="negative days_rest"):
            engineer.validate_features(df)
            
    def test_no_future_leak_rolling_stats(self, engineer, leakage_df):
        """
        Verify that rolling stats do NOT propagate indefinitely (ffill removal).
        Match 2 (1st upcoming): Should have stats from Match 1.
        Match 8 (Far future): Should be NaN (Window=5 expired, History all NaN).
        """
        # Extend dataframe to exceed window size (5)
        # Match 1 (Index 0) is valid. Matches 2..10 are NaN.
        long_df = pd.DataFrame({
            'team_id': ['A'] * 10,
            'match_id': [str(i) for i in range(10)],
            'date': pd.date_range('2023-01-01', periods=10),
            'goals_scored': [2.0] + [np.nan]*9,
            'goals_conceded': [1.0] + [np.nan]*9
        })
        
        # Calculate Rolling Goals (Window 5)
        df = engineer.calculate_rolling_stats(long_df, window=5, metrics=['goals'])
        
        # Match 2 (Index 1): Valid (Window covers Index 0)
        assert df.loc[1, 'rolling_goals_scored_5'] == 2.0
        
        # Match 3 (Index 2): Valid (Window covers Index 0)
        assert df.loc[2, 'rolling_goals_scored_5'] == 2.0
        
        # Match 8 (Index 7):
        # Shift(1) -> Index 6. Window [2,3,4,5,6]. All NaNs.
        # Rolling mean should be NaN. 
        # If ffill was present, this would be 2.0.
        assert pd.isna(df.loc[7, 'rolling_goals_scored_5'])

    def test_validate_lookahead_bias(self, engineer):
        """Verify validate_features detects rolling stats on debut match."""
        df = pd.DataFrame({
            'team_id': ['A'],
            'date': pd.to_datetime(['2023-01-01']),
            'rolling_goals_5': [1.5] # IMPOSSIBLE for debut match
        })
        
        with pytest.raises(ValueError, match="Look-ahead bias detected"):
            engineer.validate_features(df)

    def test_no_look_ahead_bias(self):
        """Ensure rolling stats don't leak into future matches (Match 4)."""
        # Create dataset with 1 upcoming match (Match 3) and 1 far future (Match 4)
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=4),
            'team_id': ['A', 'A', 'A', 'A'],
            'goals_scored': [2.0, 1.0, np.nan, np.nan],  
            'goals_conceded': [1.0, 1.0, np.nan, np.nan]
        })
        
        engineer = FeatureEngineer()
        result = engineer.calculate_rolling_stats(df, window=2, metrics=['goals'])
        
    def test_no_look_ahead_bias(self):
        """
        Verify that rolling stats stop propagating after window expires (ffill removal).
        Scenario:
        - Window = 2
        - M1, M2: Played (values 2.0, 1.0)
        - M3, M4, M5: Upcoming (NaN)
        
        Expectations:
        - M3: (2.0, 1.0) -> 1.5 (Valid Prediction Feature)
        - M4: (1.0, NaN) -> 1.0 (Valid - partial history in window)
        - M5: (NaN, NaN) -> NaN (Correct - no history in window. ffill would have leaked 1.0 here)
        """
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'team_id': ['A'] * 5,
            'goals_scored': [2.0, 1.0, np.nan, np.nan, np.nan],
            'goals_conceded': [1.0, 1.0, np.nan, np.nan, np.nan]
        })
        
        engineer = FeatureEngineer()
        result = engineer.calculate_rolling_stats(df, window=2, metrics=['goals'])
        
        # Match 3 (Index 2): Valid
        assert result.iloc[2]['rolling_goals_scored_2'] == 1.5
        
        # Match 5 (Index 4): Should be NaN
        assert pd.isna(result.iloc[4]['rolling_goals_scored_2']), \
            (f"Match 5 should be NaN (Expired Window). "
             f"Got {result.iloc[4]['rolling_goals_scored_2']} - check ffill removal.")

    def test_no_form_leakage(self, engineer):
        """
        Verify form rating does not leak via ffill.
        Match 1: Played (W). Form = Neutral (1.5) [Before match].
        Match 2: Upcoming. Form = derived from Match 1.
        Match 3: Upcoming. Form should be Neutral (1.5), NOT leaked from Match 2.
        """
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=3),
            'team_id': ['A', 'A', 'A'],
            'goals_scored': [3.0, np.nan, np.nan],
            'goals_conceded': [0.0, np.nan, np.nan]
        })
        
        df = engineer.calculate_form(df)
        
        # Match 1 (Index 0): Neutral (First match)
        assert df.loc[0, 'form_rating'] == FeatureDefaults.FORM_NEUTRAL
        
        # Match 2 (Index 1): Valid Form. Match 1 was Win (3 pts).
        # Previous Form=1.5. Target=3.0. EWM updates towards 3.0.
        # It won't be 1.5.
        assert df.loc[1, 'form_rating'] != FeatureDefaults.FORM_NEUTRAL
        
        # Match 3 (Index 2): Should be Neutral (1.5).
        # Shift(1) gives Match 2 points (NaN). 
        # EWM with NaN input -> NaN.
        # fillna(Neutral) -> 1.5.
        # If ffill existed, it would equal Match 2's form.
        assert df.loc[2, 'form_rating'] == FeatureDefaults.FORM_NEUTRAL
        assert df.loc[2, 'form_rating'] != df.loc[1, 'form_rating']

    def test_h2h_sparse_history_uses_defaults(self, engineer):
        """Sparse H2H history should still return the expected derived columns."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01']),
            'home_team': ['Alpha FC'],
            'away_team': ['Beta FC'],
            'home_score': [1.0],
            'away_score': [0.0],
            'home_total_cards': [2.0],
            'away_total_cards': [1.0],
            'home_corners': [5.0],
            'away_corners': [3.0],
        })

        result = engineer.calculate_h2h_features(df)

        assert 'h2h_goals_o25_rate' in result.columns
        assert 'h2h_btts_rate' in result.columns
        assert result.loc[0, 'h2h_goals_o25_rate'] == pytest.approx(0.53)
        assert result.loc[0, 'h2h_btts_rate'] == pytest.approx(0.54)

    def test_referee_rolling_card_rate_feature(self, engineer):
        """Referee rolling card rate uses shifted history (no current-match leakage)."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-08', '2024-01-15']),
            'referee': ['REF_A', 'REF_A', 'REF_A'],
            'home_yellow_cards': [1, 0, 2],
            'away_yellow_cards': [2, 1, 1],
            'home_red_cards': [0, 0, 1],
            'away_red_cards': [1, 1, 0],
        })

        res = engineer.add_referee_features(df, window=10)

        assert 'referee_id' in res.columns
        assert 'referee_card_rate_10' in res.columns
        assert res.loc[1, 'referee_card_rate_10'] == pytest.approx(4.0)
        assert res.loc[2, 'referee_card_rate_10'] == pytest.approx(3.0)

    def test_referee_feature_handles_missing_referee_column(self, engineer):
        """Missing referee metadata should degrade gracefully to UNKNOWN ids."""
        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01', '2024-01-08']),
            'home_cards': [2, 3],
            'away_cards': [1, 2],
        })

        res = engineer.add_referee_features(df, window=10)

        assert (res['referee_id'] == 'UNKNOWN').all()
        assert (res['referee_card_rate_10'] >= 0).all()

    def test_weather_feature_fails_safe_without_api_key(self, engineer, monkeypatch):
        """Weather features should default to 0 when API key is unavailable."""
        monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)

        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01']),
            'stadium_latitude': [51.5074],
            'stadium_longitude': [-0.1278],
        })

        res = engineer.add_weather_features(df)
        assert res.loc[0, 'precipitation_mm'] == pytest.approx(0.0)
        assert res.loc[0, 'wind_speed_kmh'] == pytest.approx(0.0)

    def test_weather_feature_parses_and_stores_values(self, engineer, monkeypatch):
        """Weather API values should be written into feature rows when available."""
        monkeypatch.setattr(
            engineer,
            "_fetch_openweather_snapshot",
            Mock(return_value=(1.2, 14.4)),
        )

        df = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-01']),
            'stadium_latitude': [48.8566],
            'stadium_longitude': [2.3522],
        })

        res = engineer.add_weather_features(df, api_key="test-key")
        assert res.loc[0, 'precipitation_mm'] == pytest.approx(1.2)
        assert res.loc[0, 'wind_speed_kmh'] == pytest.approx(14.4)

    def test_team_availability_stub_defaults_to_full_squad(self, engineer):
        """Team availability stub should expose a bounded completeness feature."""
        df = pd.DataFrame({
            'team_id': ['A', 'B'],
            'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        })

        res = engineer.add_team_availability_feature(df)
        assert 'team_availability_pct' in res.columns
        assert (res['team_availability_pct'] == 1.0).all()
        assert res['team_availability_pct'].between(0.0, 1.0).all()

    def test_adversarial_leak_test(self, engineer):
        """
        Match 5 contains extreme values (100 goals, 50 shots, 10 cards).
        A missing match is inserted.
        Assert that ONLY future matches (Match 6+) reflect this anomaly.
        Match 5's own features MUST NOT reflect its own anomaly.
        """
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=7),
            'team_id': ['A'] * 7,
            'match_id': [str(i) for i in range(1, 8)],
            'goals_scored': [1.0, 2.0, 1.0, 0.0, 100.0, 1.0, 2.0],  # Match 5 has 100
            'goals_conceded': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            'shots_for': [10.0, 12.0, 9.0, 11.0, 50.0, 10.0, 12.0], # Match 5 has 50
            'shots_against': [5.0, 6.0, 4.0, 5.0, 0.0, 5.0, 4.0],
            'cards_scored': [1.0, 2.0, 1.0, 0.0, 10.0, 1.0, 0.0],   # Match 5 has 10
            'cards_conceded': [1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0],
            'season': ['2023/2024'] * 7
        })
        
        # Insert a missing match at index 2 (Match 3)
        cols_to_nan = ['goals_scored', 'goals_conceded', 'shots_for', 'shots_against', 'cards_scored', 'cards_conceded']
        df.loc[2, cols_to_nan] = np.nan
        
        # Calculate Rolling
        res = engineer.calculate_rolling_stats(df, window=3, metrics=['goals', 'cards'])
        
        # Manual calculation for rolling shots
        date_col = engineer._get_date_column(res)
        res = res.sort_values(['team_id', date_col])
        res['rolling_shots_for_3'] = res.groupby('team_id')['shots_for'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        
        # Match 1-4 should not be affected by extreme values
        for idx in range(0, 5): # index 0, 1, 2, 3, 4 (Match 1 to 5)
            # Goals
            assert res.iloc[idx]['rolling_goals_scored_3'] < 3.0 or pd.isna(res.iloc[idx]['rolling_goals_scored_3']), f"Leak at index {idx} in goals"
            # Cards
            assert res.iloc[idx]['rolling_cards_scored_3'] < 3.0 or pd.isna(res.iloc[idx]['rolling_cards_scored_3']), f"Leak at index {idx} in cards"
            # Shots
            if idx > 0:
                assert res.iloc[idx]['rolling_shots_for_3'] < 15.0 or pd.isna(res.iloc[idx]['rolling_shots_for_3']), f"Leak at index {idx} in shots"
        
        # Match 6 (Index 5) SHOULD be affected by Match 5
        assert res.iloc[5]['rolling_goals_scored_3'] > 30.0, "Anomaly didn't propagate to future goals"
        assert res.iloc[5]['rolling_cards_scored_3'] > 3.0, "Anomaly didn't propagate to future cards"
        assert res.iloc[5]['rolling_shots_for_3'] > 15.0, "Anomaly didn't propagate to future shots"

    def test_chronological_shuffle_test(self, engineer):
        """
        Pass a randomly shuffled history of matches.
        Output must be identical to the chronological output, proving
        that everything explicitly enforces chronological order.
        """
        chrono_df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'team_id': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'goals_scored': [1.0, 2.0, 3.0, 1.0, 0.0, 2.0, 4.0, 1.0, 2.0, 0.0],
            'goals_conceded': [0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0],
            'season': ['2023/2024'] * 10
        })
        
        shuffled_df = chrono_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate multiple features to test various functions
        res_chrono = engineer.calculate_rolling_stats(chrono_df, window=5, metrics=['goals'])
        res_chrono = engineer.calculate_std_stats(res_chrono, metrics=['goals'], smoothing=5)
        res_chrono = engineer.calculate_form(res_chrono)
        
        res_shuffle = engineer.calculate_rolling_stats(shuffled_df, window=5, metrics=['goals'])
        res_shuffle = engineer.calculate_std_stats(res_shuffle, metrics=['goals'], smoothing=5)
        res_shuffle = engineer.calculate_form(res_shuffle)
        
        res_chrono = res_chrono.sort_values(['team_id', 'date']).reset_index(drop=True)
        res_shuffle = res_shuffle.sort_values(['team_id', 'date']).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(res_chrono, res_shuffle)

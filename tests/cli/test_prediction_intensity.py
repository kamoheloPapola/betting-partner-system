import pandas as pd

from src.cli.commands.prediction import _is_high_intensity


def test_is_high_intensity_detects_cup_context():
    match = pd.Series({"home_team": "Alpha FC", "away_team": "Beta FC"})
    assert _is_high_intensity(match, "FA_CUP") is True


def test_is_high_intensity_detects_h2h_card_profile():
    match = pd.Series(
        {
            "home_team": "Alpha FC",
            "away_team": "Beta FC",
            "h2h_match_count": 4,
            "h2h_cards_o25_rate": 0.74,
            "h2h_avg_cards": 5.2,
            "days_since_last_h2h": 30,
        }
    )
    assert _is_high_intensity(match, "PL") is True


def test_is_high_intensity_ignores_unrelated_match_id():
    match = pd.Series(
        {
            "match_id": "4719e5144c1122b1",
            "home_team": "Team A",
            "away_team": "Team B",
            "h2h_match_count": 1,
            "h2h_cards_o25_rate": 0.3,
            "h2h_avg_cards": 2.5,
        }
    )
    assert _is_high_intensity(match, "PL") is False

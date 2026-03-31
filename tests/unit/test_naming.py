import pytest
import logging
from src.utils.naming import normalize_team_name

class TestTeamNormalizationImproved:
    def test_league_isolation(self):
        # "Arsenal" is mapping for PL
        assert normalize_team_name("Arsenal", "PL") == "ARSENAL"
        # "Arsenal" shouldn't fuzzy match to anything in PD (Spain) if score < 85
        # actually there is no Arsenal in Spain. 
        # Let's try matching a Spanish team name in English league
        assert normalize_team_name("Real Madrid", "PL") == "REAL MADRID" # Returns original + warning

    def test_season_aware_relegation(self):
        # 2023 (2023/24 season) had Luton
        assert normalize_team_name("Luton", "PL", 2023) == "LUTON TOWN"
        # 2024 (2024/25 season) Does NOT have Luton canonical, but will fuzzy match if not careful.
        # But since Luton is not in 2024 canonical, it should return original (or fuzzy match if high enough)
        # Actually Luton Town score with any 2024 team should be low.
        assert normalize_team_name("Luton Town", "PL", 2024) == "LUTON TOWN" # Not in 2024 set

    def test_promotion_mapping(self):
        # 2024 (2024/25) has Leicester
        assert normalize_team_name("Leicester", "PL", 2024) == "LEICESTER CITY"
        # 2023 did NOT have Leicester canonical
        assert normalize_team_name("Leicester", "PL", 2023) == "LEICESTER" # Fallback to original

    def test_alias_league_context(self):
        # "Man Utd" alias in PL
        assert normalize_team_name("Man Utd", "PL") == "MANCHESTER UNITED"

    def test_fuzzy_match_within_league(self):
        # "Manchester City FC" -> "Manchester City" in PL
        assert normalize_team_name("Manchester City FC", "PL") == "MANCHESTER CITY"
        
        # "Bayern Muenchen" -> "Bayern Munich" in BL1
        assert normalize_team_name("Bayern Muenchen", "BL1") == "BAYERN MUNICH"

    def test_fallback_to_latest_season(self, caplog):
        # Requesting a season we don't have (e.g. 2022) should fallback to latest (2024 or 2023)
        with caplog.at_level(logging.INFO):
            result = normalize_team_name("Arsenal", "PL", 2022)
            assert result == "ARSENAL"
            assert "Falling back to" in caplog.text

    def test_invalid_league(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert normalize_team_name("Arsenal", "UNKNOWN") == "ARSENAL"
            assert "No team data for league" in caplog.text

    def test_empty_string_validation(self):
        from src.ingestion.odds_api.utils import normalize_odds_team
        with pytest.raises(ValueError, match="Team name cannot be empty"):
            normalize_odds_team("", "PL")
        with pytest.raises(ValueError, match="Team name cannot be empty"):
            normalize_odds_team("   ", "PL")

    def test_league_validation_fast_fail(self):
        from src.ingestion.odds_api.utils import normalize_odds_team
        with pytest.raises(ValueError, match="Unsupported league for normalization"):
            normalize_odds_team("Arsenal", "UNKNOWN_LEAGUE")

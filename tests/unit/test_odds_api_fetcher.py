import pytest
import requests
from unittest.mock import Mock, patch
from src.ingestion.odds_api.fetch_fixtures import OddsAPIFetcher, OddsAPIConfig
from datetime import datetime, timezone

class TestOddsAPIFetcher:
    @pytest.fixture
    def fetcher(self):
        config = OddsAPIConfig(api_key="test_key")
        return OddsAPIFetcher(config=config)

    @patch('requests.Session.get')
    def test_fetch_fixtures_success(self, mock_get, fetcher):
        # Mock successful response
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.headers = {
            "x-requests-remaining": "450",
            "Content-Length": "500"
        }
        mock_resp.json.return_value = [
            {
                "id": "match_1",
                "commence_time": "2026-01-10T20:00:00Z",
                "home_team": "Team  A\x00", # Double space + control char
                "away_team": "Team B "   # Trailing space
            },
            {
                "id": "match_2",
                "commence_time": "2025-09-10T20:00:00Z",
                "home_team": "Team C",
                "away_team": "Team D"
            }
        ]
        mock_get.return_value = mock_resp
        
        with patch('src.ingestion.odds_api.fetch_fixtures.get_odds_api_sport_key', return_value='soccer_epl'):
            fixtures = fetcher.fetch_fixtures("EPL")
        
        assert len(fixtures) == 2
        # Match 1: Jan 2026 -> 2025 season
        assert fixtures[0]["odds_api_id"] == "match_1"
        assert fixtures[0]["season"] == 2025
        assert fixtures[0]["home_team"] == "Team A" # Normalized
        assert fixtures[0]["away_team"] == "Team B" # Normalized
        assert fixtures[0]["source"] == OddsAPIFetcher.SOURCE_NAME
        
        # Match 2: Sep 2025 -> 2025 season
        assert fixtures[1]["odds_api_id"] == "match_2"
        assert fixtures[1]["season"] == 2025
        
        # Verify endpoint and params
        args, kwargs = mock_get.call_args
        assert "/events" in args[0]
        assert kwargs["params"]["apiKey"] == "test_key"
        assert kwargs["timeout"] == (5, 15)

    @patch('requests.Session.get')
    def test_fetch_fixtures_deduplication(self, mock_get, fetcher):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.headers = {"x-requests-remaining": "450"}
        mock_resp.json.return_value = [
            {"id": "dup", "commence_time": "2026-01-10T20:00:00Z", "home_team": "A", "away_team": "B"},
            {"id": "dup", "commence_time": "2026-01-10T20:00:00Z", "home_team": "A", "away_team": "B"}
        ]
        mock_get.return_value = mock_resp
        
        with patch('src.ingestion.odds_api.fetch_fixtures.get_odds_api_sport_key', return_value='soccer_epl'):
            fixtures = fetcher.fetch_fixtures("EPL")
        
        assert len(fixtures) == 1

    @patch('requests.Session.get')
    def test_fetch_fixtures_size_limit(self, mock_get, fetcher):
        mock_resp = Mock()
        mock_resp.status_code = 200
        # 20MB Header
        mock_resp.headers = {"Content-Length": str(20 * 1024 * 1024)}
        mock_get.return_value = mock_resp
        
        with patch('src.ingestion.odds_api.fetch_fixtures.get_odds_api_sport_key', return_value='soccer_epl'):
            with pytest.raises(ValueError, match="Response too large"):
                fetcher.fetch_fixtures("EPL")

    @patch('requests.Session.get')
    def test_fetch_fixtures_rate_limit(self, mock_get, fetcher):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.headers = {"x-requests-remaining": "0"}
        mock_get.return_value = mock_resp
        
        with patch('src.ingestion.odds_api.fetch_fixtures.get_odds_api_sport_key', return_value='soccer_epl'):
            with pytest.raises(RuntimeError, match="API rate limit reached"):
                fetcher.fetch_fixtures("EPL")

    @patch('requests.Session.get')
    def test_fetch_fixtures_malformed_item_skipping(self, mock_get, fetcher):
        mock_resp = Mock()
        mock_resp.status_code = 200
        mock_resp.headers = {"x-requests-remaining": "450"}
        mock_resp.json.return_value = [
            {
                "id": "valid",
                "commence_time": "2026-01-10T20:00:00Z",
                "home_team": "Team A",
                "away_team": "Team B"
            },
            {
                "id": "invalid_time",
                "commence_time": "garbage", # Will fail parser.isoparse
                "home_team": "Team C",
                "away_team": "Team D"
            },
            {
                "id": "invalid_chars",
                "commence_time": "2026-01-10T20:00:00Z",
                "home_team": "Team (A)", # Invalid char '('
                "away_team": "Team B"
            }
        ]
        mock_get.return_value = mock_resp
        
        with patch('src.ingestion.odds_api.fetch_fixtures.get_odds_api_sport_key', return_value='soccer_epl'):
            fixtures = fetcher.fetch_fixtures("EPL")
        
        assert len(fixtures) == 1
        assert fixtures[0]["odds_api_id"] == "valid"

    def test_infer_season(self, fetcher):
        # Aug 2024 -> 2024
        assert fetcher._infer_season(datetime(2024, 8, 1)) == 2024
        # Dec 2024 -> 2024
        assert fetcher._infer_season(datetime(2024, 12, 31)) == 2024
        # Jan 2025 -> 2024
        assert fetcher._infer_season(datetime(2025, 1, 15)) == 2024
        # Jul 2025 -> 2024
        assert fetcher._infer_season(datetime(2025, 7, 31)) == 2024
        # Aug 2025 -> 2025
        assert fetcher._infer_season(datetime(2025, 8, 1)) == 2025

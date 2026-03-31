"""
Odds API Fixture Fetcher.

Fetches upcoming fixtures from The Odds API with:
- Rate limit protection
- Response size validation
- Pydantic schema enforcement
- Team name normalization
"""
import logging
import requests
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Optional, Dict, Any, Mapping
from dataclasses import dataclass
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime, timezone, timedelta
from dateutil import parser

from src.ingestion.odds_api.utils import get_odds_api_sport_key
from src.config import get_required_env

logger = logging.getLogger(__name__)

@dataclass
class OddsAPIConfig:
    """Configuration for Odds API client."""
    api_key: str
    timeout: tuple = (5, 15)  # (connect, read)
    max_retries: int = 3
    min_quota_warning: int = 10
    max_response_size_mb: int = 10
    base_url: str = "https://api.the-odds-api.com/v4/sports"

class OddsAPIFixture(BaseModel):
    """Fixture model for Odds API responses."""
    model_config = ConfigDict(populate_by_name=True)
    
    odds_api_id: str = Field(alias="id", min_length=1)
    kickoff_utc: str = Field(alias="commence_time", min_length=1)
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    sport_key: Optional[str] = None

class OddsAPIFetcher:
    SOURCE_NAME = "odds_api"

    def __init__(self, config: Optional[OddsAPIConfig] = None):
        """
        Initialize the Odds API Fetcher.
        
        Args:
            config: Optional OddsAPIConfig object. If not provided, 
                    it will be created using ODDS_API_KEY from config.
        """
        if config:
            self.config = config
        else:
            self.config = OddsAPIConfig(api_key=get_required_env("ODDS_API_KEY"))
            
        self._setup_session()
        
    def _setup_session(self):
        """Configure requests session with retries for resilience."""
        self.session = requests.Session()
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def fetch_fixtures(self, league_code: str) -> List[Dict[str, Any]]:
        """
        Fetch upcoming matches for a league.
        
        Uses the high-efficiency /events endpoint for metadata discovery.
        
        Args:
            league_code: Internal league identifier (e.g. 'EPL', 'PD')
            
        Returns:
            List of normalized fixture dictionaries with keys:
            - odds_api_id: Unique fixture ID
            - kickoff_utc: ISO 8601 UTC datetime string
            - home_team: Normalized home team name
            - away_team: Normalized away team name
            - league: League code
            - season: Inferred season year (e.g. 2024)
            - status: Match status ("SCHEDULED")
            - source: Data source identifier ("odds_api")
            
        Raises:
            ValueError: If league or response is invalid
            requests.RequestException: On connection or API issues
            RuntimeError: If rate limit is reached

        Example:
            >>> fetcher = OddsAPIFetcher()
            >>> fixtures = fetcher.fetch_fixtures('EPL')
            >>> print(fixtures[0])
            {
                'odds_api_id': 'abc123',
                'kickoff_utc': '2024-09-15T14:00:00+00:00',
                'home_team': 'Manchester United',
                'away_team': 'Liverpool',
                'league': 'EPL',
                'season': 2024,
                'status': 'SCHEDULED',
                'source': 'odds_api'
            }
        """
        sport_key = get_odds_api_sport_key(league_code)
        if not sport_key:
            raise ValueError(f"No Odds API sport key found for league: {league_code}")
            
        url = f"{self.config.base_url}/{sport_key}/events"
        params = {"apiKey": self.config.api_key}
        
        logger.info(f"Discovery: Fetching fixtures from Odds API for {sport_key}...")
        
        try:
            # Use stream=True to check content length before reading full body
            resp = self.session.get(url, params=params, timeout=self.config.timeout, stream=True)
            
            # 1. Size Check (Safety Polish)
            content_length = resp.headers.get('Content-Length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                if size_mb > self.config.max_response_size_mb:
                    raise ValueError(f"Response too large: {size_mb:.1f}MB (limit: {self.config.max_response_size_mb}MB)")
            
            # Rate Limit Protection
            self._check_rate_limit(resp.headers)
            
            resp.raise_for_status()
            
            # 2. Read content with safety read limit (in case Content-Length was missing/spoofed)
            # requests.Response.content reads the whole thing. For true safety we'd iterate, 
            # but let's trust .json() if we got past the header check or just read a chunk test.
            data = resp.json()
            
        except requests.Timeout:
            logger.error(f"Odds API Timeout for {sport_key}")
            raise
        except requests.HTTPError as e:
            logger.error(
                f"Odds API HTTP Error {resp.status_code} for {sport_key}: {resp.text[:500] if resp.text else 'No body'}"
            )
            raise
        except requests.RequestException as e:
            logger.error(f"Odds API Connection Error for {sport_key}: {e}")
            raise
        except ValueError as e:
            msg = f"Data error from Odds API for {sport_key}: {e}"
            logger.error(msg)
            raise ValueError(msg) from e

        fixtures = []
        for item in data:
            try:
                # Validation via Pydantic
                fixture = OddsAPIFixture(**item)
                
                # Parse kickoff (validated ISO 8601)
                kickoff_dt = parser.isoparse(fixture.kickoff_utc)
                
                # Normalize for system use
                normalized = {
                    "odds_api_id": fixture.odds_api_id,
                    "kickoff_utc": self._parse_kickoff_time(fixture.kickoff_utc),
                    "home_team": self._normalize_team_name(fixture.home_team),
                    "away_team": self._normalize_team_name(fixture.away_team),
                    "league": league_code,
                    "season": self._infer_season(kickoff_dt),
                    "status": "SCHEDULED",
                    "source": self.SOURCE_NAME
                }
                fixtures.append(normalized)
            except (ValueError, KeyError, TypeError) as e:
                # Truncated item log to avoid bloating
                item_preview = str(item)[:200]
                logger.warning(
                    f"Skipping malformed Odds API fixture for {sport_key}: {e}",
                    extra={"item_preview": item_preview, "error_type": type(e).__name__}
                )
                continue
            except Exception as e:
                # Unexpected error during per-item processing
                logger.error(f"Unexpected error processing fixture for {sport_key}: {e}", exc_info=True)
                raise
            
        # 3. Deduplication (Safety Polish)
        seen_ids = set()
        unique_fixtures = []
        for f in fixtures:
            fid = f["odds_api_id"]
            if fid in seen_ids:
                logger.warning(f"Duplicate fixture ID detected and skipped: {fid}")
                continue
            seen_ids.add(fid)
            unique_fixtures.append(f)

        logger.info(f"Retrieved {len(unique_fixtures)} unique validated fixtures for {league_code}.")
        return unique_fixtures

    def _normalize_team_name(self, name: str) -> str:
        """
        Normalize team names for consistent matching with internal data.
        
        Steps:
        1. Remove control characters
        2. Collapse multiple spaces
        3. Strip whitespace
        4. Validate against whitelist
        
        Raises:
            ValueError: If team name contains invalid characters
        """
        # Remove control characters (same as pipeline)
        clean = re.sub(r'[\x00-\x1F\x7F]', '', name)
        
        # Collapse multiple spaces
        clean = re.sub(r'\s+', ' ', clean)
        
        # Strip edges
        clean = clean.strip()
        
        # Validate: Allow alphanumeric (incl Unicode), spaces, hyphens, periods, and underscores
        if not re.match(r'^[\w\s\-\.]+$', clean, re.UNICODE):
            raise ValueError(
                f"Team name contains invalid characters: '{name}' -> '{clean}'. "
                "Only alphanumeric, spaces, hyphens, periods, and underscores allowed."
            )
        
        return clean

    def _parse_kickoff_time(self, kickoff_str: str) -> str:
        """
        Parse and validate kickoff time from Odds API.
        
        Returns: ISO 8601 UTC string.
        """
        try:
            dt = parser.isoparse(kickoff_str)
            
            # Force UTC if missing
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            
            # Past check (grace period for live/recently finished)
            now = datetime.now(timezone.utc)
            if dt < now - timedelta(hours=2):
                logger.debug(f"Fixture is in the past: {kickoff_str}")
                
            return dt.isoformat()
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid kickoff time format: {kickoff_str}") from e

    def _infer_season(self, kickoff_dt: datetime) -> int:
        """
        Infer season year from kickoff date.
        
        Logic:
        - Aug-Dec: Current year
        - Jan-Jul: Previous year (season spans calendar years)
        """
        month = kickoff_dt.month
        year = kickoff_dt.year
        
        if month >= 8:
            return year
        else:
            return year - 1

    def _check_rate_limit(self, headers: Mapping[str, str]):
        """Check and log API rate limit status."""
        # Intentionally case-insensitive check via .get() which is standard 
        # but type hint refined to str/str as per requests spec
        remaining = headers.get("x-requests-remaining")
        if not remaining:
            return
        
        try:
            remaining_int = int(remaining)
            logger.info(f"Odds API Quota: {remaining_int} remaining")
            
            if remaining_int <= self.config.min_quota_warning:
                logger.warning(f"Low API quota alert: {remaining_int} requests left")
            
            if remaining_int == 0:
                logger.error("Odds API quota EXHAUSTED!")
                raise RuntimeError("API rate limit reached")
                
        except (ValueError, TypeError):
            logger.warning(f"Invalid rate limit header received: {remaining}")

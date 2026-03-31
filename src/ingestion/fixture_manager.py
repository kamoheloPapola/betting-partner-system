"""
Fixture Manager.

Manages upcoming fixture data from the Odds API. Handles:
- Fixture fetching and caching
- Team name normalization
- Match ID generation for upcoming matches
"""

import json
import logging
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from src.config import PROCESSED_DATA_DIR
from src.core.constants import STATUS_SCHEDULED
from src.ingestion.odds_api.fetch_fixtures import OddsAPIFetcher
from src.ingestion.odds_api.utils import normalize_odds_team
from src.utils.naming import generate_match_fingerprint, calculate_season

logger = logging.getLogger(__name__)

class FixtureManager:
    """
    Manages fixture data with strict 24-hour caching and minimal schema.
    Source: Odds API (Fixtures Only).
    Storage: data/processed/fixtures/{LEAGUE}_upcoming.json
    """
    
    CACHE_DIR = PROCESSED_DATA_DIR / "fixtures"
    DEFAULT_CACHE_HOURS = 24
    
    def __init__(self, cache_hours: Optional[int] = None) -> None:
        """
        Initialize FixtureManager.
        
        Args:
            cache_hours: Duration in hours to trust cached fixtures. Defaults to 24.
        """
        self.cache_duration = cache_hours or self.DEFAULT_CACHE_HOURS
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.fetcher = OddsAPIFetcher()
        
    def get_fixtures(self, league: str, force_refresh: bool = False) -> List[Dict]:
        """
        Main entry point. Checks cache -> Returns valid cache OR fetches new.
        
        Args:
            league: League code (e.g., 'PL').
            force_refresh: If True, bypass cache and fetch fresh from API.
        
        Returns:
            List of fixture dictionaries with schema:
            {
                "match_id": "abc123...",
                "league": "PL",
                "season": 2024,
                "date_utc": "2024-01-15T15:00:00+00:00",
                "home_team": "Arsenal",
                "away_team": "Liverpool",
                "status": "SCHEDULED"
            }
        """
        cache_path = self.CACHE_DIR / f"{league}_upcoming.json"
        
        # 1. Check Cache (skip if force_refresh)
        if not force_refresh:
            cached_data = self._load_cache(cache_path)
            if cached_data:
                try:
                    fetched_at = datetime.fromisoformat(cached_data['fetched_at'])
                    # Age calculation assumes cache timing is stored in UTC
                    age = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
                    if age < self.cache_duration:
                        logger.info(f"Using cached fixtures for {league} (Age: {age:.2f}h)")
                        return cached_data['fixtures']
                    else:
                        logger.info(f"Cache expired for {league} (Age: {age:.2f}h). Fetching new...")
                except (ValueError, TypeError, KeyError) as e:
                    logger.warning(f"Failed to parse cache timing for {league}: {e}")
            else:
                logger.info(f"No cache found for {league}. Fetching new...")
        else:
            logger.info(f"Force refresh requested for {league}. Bypassing cache...")
            
        # 2. Fetch from API
        raw_fixtures = self.fetcher.fetch_fixtures(league)
        
        if not raw_fixtures:
            logger.warning(f"Odds API returned zero fixtures for {league}. Aborting save.")
            return []
            
        # 3. Minify & Normalize
        clean_fixtures = self._minify_fixtures(raw_fixtures, league)
        
        if not clean_fixtures:
            logger.warning(f"No valid fixtures extracted for {league} after minification.")
            return []
            
        # 4. Save to Cache (Atomic Write Pattern)
        payload = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "odds_api",
            "league": league,
            "fixtures": clean_fixtures
        }
        
        # Save atomically to prevent cache corruption
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=self.CACHE_DIR, suffix=".tmp", prefix=f"{league}_") as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            with open(tmp_path, 'w') as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, cache_path)
            logger.info(f"Saved {len(clean_fixtures)} fixtures to cache: {cache_path}")
        except Exception as e:
            logger.error(f"Failed to atomically save fixtures cache for {league}: {e}")
            if tmp_path.exists():
                os.unlink(tmp_path)
        
        return clean_fixtures

    def _load_cache(self, path: Path) -> Optional[Dict]:
        """Loads and performs deep validation on cache file."""
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            # Deep Validation
            if not isinstance(data.get('fixtures'), list):
                logger.warning(f"Cache has invalid fixtures format: {path}")
                return None

            try:
                datetime.fromisoformat(data['fetched_at'])
            except (KeyError, ValueError, TypeError):
                logger.warning(f"Cache has invalid fetched_at: {path}")
                return None
                
            return data
        except Exception as e:
            logger.warning(f"Corrupt cache file {path}: {e}", exc_info=True)
        return None

    def _minify_fixtures(self, raw_data: List[Dict], league: str) -> List[Dict]:
        """
        Extracts only minimal schema: match_id, league, season, date_utc, home, away, status.
        """
        minified = []
        seen_ids = set()
        now = datetime.now(timezone.utc)
        
        for item in raw_data:
            item_id = item.get('odds_api_id', 'unknown')
            try:
                # 1. Date Handling & Validation
                date_str = item.get('kickoff_utc') or item.get('date') or item.get('commence_time')
                if not date_str:
                    logger.warning(f"Missing date for fixture {item_id}")
                    continue
                
                try:
                    match_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    # Ensure timezone awareness (UTC)
                    if match_date.tzinfo is None:
                        match_date = match_date.replace(tzinfo=timezone.utc)
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid date format '{date_str}' for {item_id}: {e}")
                    continue
                
                # Guard: Skip past games
                if match_date < now:
                    continue
                
                # 2. Team Name Validation (Pre-Normalization)
                raw_home = item.get('home_team')
                raw_away = item.get('away_team')
                
                if not raw_home or not raw_away:
                    logger.warning(f"Missing team names in fixture {item_id}")
                    continue
                
                # 3. Season Logic (Safe fallback to centralized logic)
                season_yr_raw = item.get('season')
                if not season_yr_raw:
                    season_yr_raw = calculate_season(match_date, league)
                
                # Explicit type safety for season
                try:
                    season_yr = int(season_yr_raw)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid season value '{season_yr_raw}' for {item_id}. Skipping.")
                    continue
                
                # 4. Normalize Teams with context
                home_team = normalize_odds_team(raw_home, league=league, season=season_yr)
                away_team = normalize_odds_team(raw_away, league=league, season=season_yr)
                
                if not home_team or not away_team:
                    logger.warning(f"Invalid team names after normalization for {item_id}: {raw_home} / {raw_away}")
                    continue

                # 5. ID Generation (Stable Hash)
                match_id = generate_match_fingerprint(league, match_date, home_team, away_team, season=season_yr)

                if match_id in seen_ids:
                    logger.warning(f"Duplicate fixture detected (ID: {match_id}). Skipping.")
                    continue
                seen_ids.add(match_id)

                schema = {
                    "match_id": match_id,
                    "league": league,
                    "season": season_yr, # Standardized raw int
                    "date_utc": match_date.isoformat(), # Normalized ISO format
                    "home_team": home_team,
                    "away_team": away_team,
                    "status": STATUS_SCHEDULED
                }
                minified.append(schema)
                
            except Exception as e:
                logger.warning(f"Error minifying fixture {item_id}: {e}", exc_info=True)
                
        return minified

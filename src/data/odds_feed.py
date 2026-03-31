"""
Odds Feed Module.

Phase 1: Manual/static odds from JSON file.
Phase 2: The Odds API with aggressive caching.

Usage:
    from src.data.odds_feed import get_odds_for_match
    
    odds = get_odds_for_match("LIVERPOOL vs CHELSEA", "PL")
    # Returns {'goals_u25': 1.75, 'goals_o25': 2.10, ...}
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# === PATHS ===
MANUAL_ODDS_FILE = Path(__file__).parent.parent.parent / "config" / "manual_odds.json"


def load_manual_odds() -> dict:
    """Load manual odds from JSON file."""
    if not MANUAL_ODDS_FILE.exists():
        logger.warning(f"Manual odds file not found: {MANUAL_ODDS_FILE}")
        return {}
    
    try:
        with open(MANUAL_ODDS_FILE) as f:
            data = json.load(f)
        
        # Log freshness
        meta = data.get('_meta', {})
        last_updated = meta.get('last_updated', 'unknown')
        logger.debug(f"Loaded manual odds (last updated: {last_updated})")
        
        return data
    except Exception as e:
        logger.error(f"Failed to load manual odds: {e}")
        return {}


def get_odds_for_match(
    match_str: str, 
    league: str = None
) -> Dict[str, float]:
    """
    Get odds for a specific match.
    
    Falls back to default odds if match not found.
    
    Args:
        match_str: Match identifier, e.g. "LIVERPOOL vs CHELSEA"
        league: Optional league code for context
        
    Returns:
        Dictionary of market -> decimal odds
    """
    data = load_manual_odds()
    
    if not data:
        logger.warning("No odds data available - using hard defaults")
        return _get_hard_defaults()
    
    # Try exact match
    fixtures = data.get('fixtures', {})
    if match_str in fixtures:
        return fixtures[match_str].get('markets', _get_hard_defaults())
    
    # Try fuzzy match (home team)
    home_team = match_str.split(' vs ')[0] if ' vs ' in match_str else match_str
    for fixture_key, fixture_data in fixtures.items():
        if home_team in fixture_key:
            logger.debug(f"Fuzzy match: {match_str} -> {fixture_key}")
            return fixture_data.get('markets', _get_hard_defaults())
    
    # Fall back to defaults
    defaults = data.get('default_odds', _get_hard_defaults())
    logger.debug(f"Using default odds for {match_str}")
    return defaults


def _get_hard_defaults() -> Dict[str, float]:
    """Hard-coded fallback odds (used when nothing else available)."""
    return {
        'goals_u25': 1.90,
        'goals_o25': 1.90,
        'btts_yes': 1.85,
        'btts_no': 1.95,
        'home_win': 2.50,
        'draw': 3.30,
        'away_win': 2.80
    }


def get_all_fixtures() -> Dict[str, dict]:
    """Get all fixtures with odds."""
    data = load_manual_odds()
    return data.get('fixtures', {})


def is_odds_fresh(max_age_hours: int = 24) -> bool:
    """Check if odds data is fresh enough."""
    data = load_manual_odds()
    meta = data.get('_meta', {})
    
    last_updated_str = meta.get('last_updated')
    if not last_updated_str:
        return False
    
    try:
        last_updated = datetime.fromisoformat(last_updated_str.replace('Z', '+00:00'))
        age = datetime.now(last_updated.tzinfo) - last_updated
        return age.total_seconds() < max_age_hours * 3600
    except Exception:
        return False


# === UTILITY FOR UPDATING ODDS ===
def update_fixture_odds(match_str: str, markets: Dict[str, float], league: str = None):
    """
    Update odds for a fixture in the manual file.
    
    For manual entry/paper trading.
    """
    data = load_manual_odds()
    
    if 'fixtures' not in data:
        data['fixtures'] = {}
    
    data['fixtures'][match_str] = {
        'match_date': datetime.now().strftime('%Y-%m-%d'),
        'league': league or 'UNKNOWN',
        'bookmaker': 'manual',
        'markets': markets
    }
    
    data['_meta'] = {
        'last_updated': datetime.now().isoformat() + 'Z',
        'source': 'manual',
        'notes': 'Updated via update_fixture_odds()'
    }
    
    with open(MANUAL_ODDS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Updated odds for {match_str}")

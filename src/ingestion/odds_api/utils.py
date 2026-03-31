"""
Odds API Utilities for League and Team Name Normalization.

Purpose:
    Map between Odds API identifiers and internal canonical names.
    Handles variations in team naming across data sources using a centralized naming hub.

Maintenance:
    - LEAGUE_MAP: Updated when new leagues are added (rare).
    - Team Mappings: Handled centrally in src.utils.naming.
    - Last verified: 2024-08-01 (Current for 2024/25 season start).

Data Flow:
    Odds API -> normalize_odds_team() -> Canonical Name -> Database/Pipeline

Example:
    >>> normalize_odds_team("Man Utd", "PL", 2024)
    'Manchester United'
    
    >>> normalize_odds_team("Tottenham Hotspur", "PL", 2024)
    'Tottenham'
"""

from typing import Optional, Annotated
from src.utils.naming import normalize_team_name

# Internal League Code -> Odds API sport_key
# Static mapping: These keys are stable in the Odds API v4.
# If the list grows significantly, consider moving to a config/YAML file.
LEAGUE_MAP = {
    "PL": "soccer_epl",
    "BL1": "soccer_germany_bundesliga",
    "SA": "soccer_italy_serie_a",
    "PD": "soccer_spain_la_liga",
    "FL1": "soccer_france_ligue_one",
}

def get_odds_api_sport_key(league_code: str) -> str:
    """
    Retrieves the Odds API sport key for a given league code.
    
    Args:
        league_code: Internal league identifier (e.g., 'PL').
        
    Returns:
        The corresponding Odds API sport key string.
        
    Raises:
        ValueError: If the league code is not supported.
    """
    if league_code not in LEAGUE_MAP:
        raise ValueError(
            f"Unsupported league for Odds API: '{league_code}'. "
            f"Supported codes: {list(LEAGUE_MAP.keys())}"
        )
    return LEAGUE_MAP[league_code]

def normalize_odds_team(
    name: str, 
    league: str, 
    season: Annotated[Optional[int], "Season start year (e.g., 2024)"] = None
) -> str:
    """
    Normalizes Odds API team name to internal canonical names.
    Delegates to the centralized naming hub in src.utils.naming.
    
    Args:
        name: Team name as returned by Odds API.
        league: League context to isolate mapping sets (e.g., 'PL').
        season: Optional season start year (e.g., 2024).
        
    Returns:
        Canonical team name used in internal systems.
        Falls back to original name with warning if unmapped 
        (handled by src.utils.naming.normalize_team_name).

    Raises:
        ValueError: If the team name is empty/whitespace or league is unknown.
    """
    if not name or not name.strip():
        raise ValueError("Team name cannot be empty or whitespace only")
    
    if league not in LEAGUE_MAP:
        raise ValueError(
            f"Unsupported league for normalization: '{league}'. "
            f"Supported codes: {list(LEAGUE_MAP.keys())}"
        )
        
    return normalize_team_name(name, league=league, season=season)

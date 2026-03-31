"""
Centralized metadata for supported leagues and match status definitions.

This module provides canonical league identifiers, display names, aliases for
fuzzy matching, and standardized match status classifications.
"""
from typing import Dict, FrozenSet, List, TypedDict

# Define public API
__all__ = [
    "LEAGUE_METADATA",
    "ACTIVE_LEAGUES",
    "FINISHED_STATUSES",
    "UPCOMING_STATUSES",
    "ALL_STATUSES",
]


class LeagueMeta(TypedDict):
    """Type definition for league metadata entries."""
    full_name: str
    country: str
    aliases: List[str]


LEAGUE_METADATA: Dict[str, LeagueMeta] = {
    'PL': {
        'full_name': 'Premier League',
        'country': 'England',
        'aliases': ['PREMIER', 'EPL', 'PREMIER LEAGUE', 'PREMIERLEAGUE', 'ENGLAND']
    },
    'PD': {
        'full_name': 'La Liga',
        'country': 'Spain',
        'aliases': ['LA LIGA', 'LALIGA', 'SPAIN', 'PRIMERA', 'PRIMERA DIVISION']
    },
    'SA': {
        'full_name': 'Serie A',
        'country': 'Italy',
        'aliases': ['SERIE A', 'SERIEA', 'ITALY', 'CALCIO']
    },
    'BL1': {
        'full_name': 'Bundesliga',
        'country': 'Germany',
        'aliases': ['BUNDESLIGA', 'GERMANY', 'BL']
    },
    'FL1': {
        'full_name': 'Ligue 1',
        'country': 'France',
        'aliases': ['LIGUE 1', 'LIGUE1', 'FRANCE', 'UBER EATS', 'LIGUE 1 UBER EATS']
    }
}

# Derived list of active/supported league codes
ACTIVE_LEAGUES: List[str] = list(LEAGUE_METADATA.keys())

# Standardized match statuses (frozenset for O(1) lookup and immutability)
FINISHED_STATUSES: FrozenSet[str] = frozenset({
    'FINISHED', 'FT', 'AET', 'PEN', 
    'POSTPONED', 'CANCELLED', 'SUSPENDED', 'INTERRUPTED'
})

UPCOMING_STATUSES: FrozenSet[str] = frozenset({
    'SCHEDULED', 'TIMED', 'IN_PLAY', 'PAUSED'
})

# Combined set of all recognized statuses
ALL_STATUSES: FrozenSet[str] = FINISHED_STATUSES | UPCOMING_STATUSES


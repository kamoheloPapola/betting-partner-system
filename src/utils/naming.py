"""
Team Name Normalization.

Season-aware team name resolution with fuzzy matching and
deterministic match fingerprint generation for deduplication.
"""
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

from rapidfuzz import process

# Define public API
__all__ = [
    "calculate_season",
    "normalize_team_name",
    "generate_match_fingerprint",
    "get_latest_season",
    "LEAGUE_TEAMS",
]

logger = logging.getLogger(__name__)
_normalize_warned: set[tuple[str, str]] = set()

# === Constants ===
# Fuzzy matching minimum score (0-100)
FUZZY_SCORE_CUTOFF = 85

# European leagues season start month (June)
EUROPEAN_SEASON_CUTOFF_MONTH = 6

# Default season cutoff month for non-European leagues
DEFAULT_SEASON_CUTOFF_MONTH = 1

# European league codes
EUROPEAN_LEAGUES = frozenset(["PL", "BL1", "SA", "PD", "FL1"])

SEASON_MAPS_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "season_maps.json"


def calculate_season(date: datetime, league: str) -> int:
    """
    Calculate the season start year based on league-specific cutoffs.

    European leagues (PL, BL1, SA, PD, FL1) typically start in Aug/Sept (cutoff June).
    Other leagues might have different schedules.
    """
    cutoff_month = EUROPEAN_SEASON_CUTOFF_MONTH if league in EUROPEAN_LEAGUES else DEFAULT_SEASON_CUTOFF_MONTH
    return date.year if date.month >= cutoff_month else date.year - 1


def _load_season_maps() -> Dict[str, Dict[int, Set[str]]]:
    """Load generated season membership data from disk."""
    with SEASON_MAPS_PATH.open(encoding="utf-8") as handle:
        raw_maps = json.load(handle)

    return {
        league: {
            int(season): set(team_names)
            for season, team_names in seasons.items()
        }
        for league, seasons in raw_maps.items()
    }


# Season-aware League Membership Hub
# Key: Season START year (e.g. 2024 for 2024/25)
LEAGUE_TEAMS: Dict[str, Dict[int, Set[str]]] = _load_season_maps()

# Season-aware aliases remain curated in code.
LEAGUE_ALIASES: Dict[str, Dict[int, Dict[str, str]]] = {
    "PL": {
        2023: {
            "Man Utd": "Manchester United",
            "Man City": "Manchester City",
            "Tottenham Hotspur": "Tottenham",
            "Spurs": "Tottenham",
            "Wolves": "Wolverhampton",
            "Nott'm Forest": "Nottingham Forest",
            "Nottm Forest": "Nottingham Forest",
            "Sheffield United": "Sheffield Utd",
            "West Ham United": "West Ham",
            "Newcastle": "Newcastle United",
            "Brighton and Hove Albion": "Brighton",
        },
        2024: {
            "Man Utd": "Manchester United",
            "Man United": "Manchester United",
            "Man City": "Manchester City",
            "Tottenham Hotspur": "Tottenham",
            "Spurs": "Tottenham",
            "Wolves": "Wolverhampton",
            "Leicester": "Leicester City",
            "Ipswich": "Ipswich Town",
            "Southampton": "Southampton",
            "Nott'm Forest": "Nottingham Forest",
            "Nottm Forest": "Nottingham Forest",
            "Sheffield United": "Sheffield Utd",
            "West Ham United": "West Ham",
            "Newcastle": "Newcastle United",
            "Brighton and Hove Albion": "Brighton",
            "Luton": "Luton Town",
            "Leeds": "Leeds United",
            "Norwich": "Norwich City",
            "West Brom": "West Bromwich Albion",
            "Watford": "Watford",
            "Cardiff": "Cardiff",
            "Hull": "Hull",
            "Middlesbrough": "Middlesbrough",
            "QPR": "QPR",
            "Reading": "Reading",
            "Stoke": "Stoke",
            "Swansea": "Swansea",
            "Wigan": "Wigan",
        },
    },
    "PD": {
        2023: {
            "Ath Bilbao": "Athletic Club",
            "Ath Madrid": "Atletico Madrid",
            "Betis": "Real Betis",
            "Sociedad": "Real Sociedad",
            "Elche": "Elche",
            "Levante": "Levante",
            "Oviedo": "Real Oviedo",
        },
        2024: {
            "Espanyol": "Espanyol",
            "Espanol": "Espanyol",
            "Valladolid": "Valladolid",
            "Granada": "Granada CF",
            "Ath Bilbao": "Athletic Club",
            "Athletic Bilbao": "Athletic Club",
            "AlavÃ©s": "Alaves",
            "Ath Madrid": "Atletico Madrid",
            "Betis": "Real Betis",
            "Sociedad": "Real Sociedad",
            "Elche": "Elche",
            "Levante": "Levante",
            "Oviedo": "Real Oviedo",
            "Vallecano": "Rayo Vallecano",
            "Celta": "Celta Vigo",
            "Cordoba": "Cordoba",
            "La Coruna": "La Coruna",
            "Malaga": "Malaga",
            "Sp Gijon": "Sp Gijon",
            "Zaragoza": "Zaragoza",
        },
    },
    "SA": {
        2023: {
            "Milan": "AC Milan",
            "Roma": "AS Roma",
            "Spal": "SPAL",
        },
        2024: {
            "Milan": "AC Milan",
            "Roma": "AS Roma",
            "Spal": "SPAL",
        },
    },
    "BL1": {
        2023: {
            "Bayern": "Bayern Munich",
            "Dortmund": "Borussia Dortmund",
            "Leverkusen": "Bayer Leverkusen",
            "M'gladbach": "Borussia Monchengladbach",
            "Hamburg": "Hamburger SV",
            "Hannover": "Hannover 96",
            "Hertha": "Hertha BSC",
            "Schalke 04": "FC Schalke 04",
            "Ein Frankfurt": "Eintracht Frankfurt",
            "Bielefeld": "Arminia Bielefeld",
            "St Pauli": "FC St. Pauli",
            "Kiel": "Holstein Kiel",
            "St. Pauli": "FC St. Pauli",
            "Augsburg": "FC Augsburg",
            "Freiburg": "SC Freiburg",
            "Hoffenheim": "TSG Hoffenheim",
            "Mainz": "Mainz 05",
            "Stuttgart": "VfB Stuttgart",
            "Wolfsburg": "VfL Wolfsburg",
            "Bochum": "VfL Bochum",
            "Heidenheim": "FC Heidenheim",
            "Darmstadt": "Darmstadt 98",
            "1. FC KÃ¶ln": "FC Koln",
            "1. FC Koln": "FC Koln",
        },
        2024: {
            "Bayern": "Bayern Munich",
            "Dortmund": "Borussia Dortmund",
            "Leverkusen": "Bayer Leverkusen",
            "M'gladbach": "Borussia Monchengladbach",
            "Hamburg": "Hamburger SV",
            "Hannover": "Hannover 96",
            "Hertha": "Hertha BSC",
            "Schalke 04": "FC Schalke 04",
            "Ein Frankfurt": "Eintracht Frankfurt",
            "Bielefeld": "Arminia Bielefeld",
            "St Pauli": "FC St. Pauli",
            "Kiel": "Holstein Kiel",
            "St. Pauli": "FC St. Pauli",
            "Augsburg": "FC Augsburg",
            "Freiburg": "SC Freiburg",
            "Hoffenheim": "TSG Hoffenheim",
            "Mainz": "Mainz 05",
            "Stuttgart": "VfB Stuttgart",
            "Wolfsburg": "VfL Wolfsburg",
            "Bochum": "VfL Bochum",
            "Heidenheim": "FC Heidenheim",
            "Darmstadt": "Darmstadt 98",
            "1. FC KÃ¶ln": "FC Koln",
            "1. FC Koln": "FC Koln",
        },
    },
    "FL1": {
        2023: {
            "Evian": "Evian Thonon Gaillard",
            "Clermont": "Clermont Foot",
        },
        2024: {
            "PSG": "Paris Saint Germain",
            "Paris SG": "Paris Saint Germain",
            "Monaco": "AS Monaco",
            "St Etienne": "Saint-Etienne",
            "Clermont": "Clermont Foot",
        },
    },
}


def get_latest_season(league: str) -> Optional[int]:
    """Helper to find the most recent season for a league."""
    seasons = LEAGUE_TEAMS.get(league)
    if not seasons:
        return None
    return max(seasons.keys())


def normalize_team_name(name: str, league: Optional[str] = None, season: Optional[int] = None) -> str:
    """
    Normalizes team name using league and season specific context.

    Args:
        name: Raw team name from the data source.
        league: Optional league identifier (e.g., 'PL'). Improves precision.
        season: Optional season start year (e.g., 2024). Handles relegation.

    Returns:
        Canonical team name. Returns original name if no match found (with warning).

    Strategy:
    1. If league provided: Exact -> Alias -> Fuzzy (Isolated to league)
    2. If no league: Systematic search across all leagues (Exact -> Alias -> Fuzzy)
    3. Fallback: Return original + warning
    """
    if not name:
        return ""

    clean = name.strip().upper()

    # CASE 1: League Context Available (High Precision)
    if league:
        if league not in LEAGUE_TEAMS:
            logger.warning(f"No team data for league: {league}. Performing global fallback.")
        else:
            target_season = season or get_latest_season(league)
            canonical_names = LEAGUE_TEAMS[league].get(target_season)
            aliases_raw = LEAGUE_ALIASES.get(league, {}).get(target_season, {})
            if canonical_names is None:
                fallback_season = get_latest_season(league)
                logger.info(f"Season {season} not found for {league}. Falling back to {fallback_season}")
                canonical_names = LEAGUE_TEAMS[league][fallback_season]
                aliases_raw = LEAGUE_ALIASES.get(league, {}).get(fallback_season, {})

            canonical_set = {c.upper() for c in canonical_names}
            aliases = {k.upper(): v.upper() for k, v in aliases_raw.items()}

            if clean in canonical_set:
                return clean
            if clean in aliases:
                return aliases[clean]

            for other_season, other_canonical_names in LEAGUE_TEAMS[league].items():
                if other_season == target_season:
                    continue
                other_canonical_set = {c.upper() for c in other_canonical_names}
                other_aliases_raw = LEAGUE_ALIASES.get(league, {}).get(other_season, {})
                other_aliases = {k.upper(): v.upper() for k, v in other_aliases_raw.items()}

                if clean in other_canonical_set:
                    return clean
                if clean in other_aliases:
                    return other_aliases[clean]

            result = process.extractOne(clean, list(canonical_set), score_cutoff=FUZZY_SCORE_CUTOFF)
            if result:
                canonical, score, index = result
                logger.info(f"Fuzzy matched '{clean}' -> '{canonical}' (Score: {score}, League: {league})")
                return canonical.upper()

    # CASE 2: No League OR No Match in League (Global Fallback)
    # Search all leagues exactly/alias first (Case-Blind)
    for l_code, s_map in LEAGUE_TEAMS.items():
        s_target = season or max(s_map.keys())
        canonical_names = s_map.get(s_target, s_map[max(s_map.keys())])
        aliases_raw = LEAGUE_ALIASES.get(l_code, {}).get(s_target, LEAGUE_ALIASES.get(l_code, {}).get(max(s_map.keys()), {}))
        c_set = {c.upper() for c in canonical_names}
        a_map = {k.upper(): v.upper() for k, v in aliases_raw.items()}
        if clean in c_set:
            return clean
        if clean in a_map:
            return a_map[clean]

        # Cross-season search (mirrors league-specific path)
        for other_season, other_canonical_names in s_map.items():
            if other_season == s_target:
                continue
            other_c_set = {c.upper() for c in other_canonical_names}
            other_aliases_raw = LEAGUE_ALIASES.get(l_code, {}).get(other_season, {})
            other_a_map = {k.upper(): v.upper() for k, v in other_aliases_raw.items()}
            if clean in other_c_set:
                return clean
            if clean in other_a_map:
                return other_a_map[clean]

    # Fuzzy match across target season across ALL leagues as last ditch
    all_canonicals = []
    for l_code, s_map in LEAGUE_TEAMS.items():
        s_target = season or max(s_map.keys())
        canonical_names = s_map.get(s_target, s_map[max(s_map.keys())])
        all_canonicals.extend([c.upper() for c in canonical_names])

    result = process.extractOne(clean, list(set(all_canonicals)), score_cutoff=FUZZY_SCORE_CUTOFF)
    if result:
        canonical, score, index = result
        logger.info(f"Global fuzzy match '{clean}' -> '{canonical}' (Score: {score})")
        return canonical.upper()

    # 4. Fallback
    warn_key = (league or "", clean)
    if warn_key not in _normalize_warned:
        logger.warning(f"Could not normalize team name: '{clean}' (League: {league or 'Global Context'}).")
        _normalize_warned.add(warn_key)
    return clean.upper()


def generate_match_fingerprint(
    league: str,
    date_obj: datetime,
    home: str,
    away: str,
    season: Optional[int] = None,
) -> str:
    """
    Generates a deterministic hash (fingerprint) for a match.

    Args:
        league: League identifier.
        date_obj: Datetime object of the match.
        home: Raw home team name.
        away: Raw away team name.
        season: Optional season start year.

    Returns:
        16-character deterministic hex hash.
    """
    # Normalize inputs with league context
    h_norm = normalize_team_name(home, league=league, season=season)
    a_norm = normalize_team_name(away, league=league, season=season)
    d_str = date_obj.strftime("%Y%m%d")

    raw = f"{league}{d_str}{h_norm}{a_norm}"
    hash_obj = hashlib.md5(raw.encode("utf-8"))
    return hash_obj.hexdigest()[:16]

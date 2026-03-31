"""
Team Name Normalization.

Season-aware team name resolution with fuzzy matching and
deterministic match fingerprint generation for deduplication.
"""
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Set

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

# === Constants ===
# Fuzzy matching minimum score (0-100)
FUZZY_SCORE_CUTOFF = 85

# European leagues season start month (June)
EUROPEAN_SEASON_CUTOFF_MONTH = 6

# Default season cutoff month for non-European leagues
DEFAULT_SEASON_CUTOFF_MONTH = 1

# European league codes
EUROPEAN_LEAGUES = frozenset(['PL', 'BL1', 'SA', 'PD', 'FL1'])


def calculate_season(date: datetime, league: str) -> int:
    """
    Calculate the season start year based on league-specific cutoffs.
    
    European leagues (PL, BL1, SA, PD, FL1) typically start in Aug/Sept (cutoff June).
    Other leagues might have different schedules.
    """
    cutoff_month = EUROPEAN_SEASON_CUTOFF_MONTH if league in EUROPEAN_LEAGUES else DEFAULT_SEASON_CUTOFF_MONTH
    return date.year if date.month >= cutoff_month else date.year - 1


# Season-aware League Metadata Hub
# Key: Season START year (e.g. 2024 for 2024/25)
LEAGUE_TEAMS: Dict[str, Dict[int, Dict[str, Any]]] = {
    "PL": {
        2023: {
            "canonical": {
                "Manchester City", "Arsenal", "Liverpool", "Aston Villa", "Tottenham",
                "Manchester United", "Newcastle United", "West Ham", "Chelsea", "Brighton",
                "Wolverhampton", "Fulham", "Bournemouth", "Crystal Palace", "Brentford",
                "Everton", "Nottingham Forest", "Luton Town", "Burnley", "Sheffield Utd",
            },
            "aliases": {
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
        },
        2024: {
            "canonical": {
                "Manchester City", "Arsenal", "Liverpool", "Aston Villa", "Tottenham",
                "Manchester United", "Newcastle United", "West Ham", "Chelsea", "Brighton",
                "Wolverhampton", "Fulham", "Bournemouth", "Crystal Palace", "Brentford",
                "Everton", "Nottingham Forest", "Leicester City", "Ipswich Town", "Southampton",
                "Sheffield Utd", "Burnley", "Luton Town", "Leeds United", "Watford", "Norwich City", "West Bromwich Albion",
                "Sunderland",
            },
            "aliases": {
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
    },
    "PD": {
        2023: {
            "canonical": {
                "Real Madrid", "Barcelona", "Atletico Madrid", "Girona", "Real Sociedad",
                "Athletic Club", "Real Betis", "Villarreal", "Valencia", "Alaves",
                "Osasuna", "Getafe", "Celta Vigo", "Sevilla", "Mallorca",
                "Rayo Vallecano", "Cadiz", "Granada CF", "Almeria", "Las Palmas",
                "Elche", "Levante", "Real Oviedo",
            },
            "aliases": {
                "Ath Bilbao": "Athletic Club",
                "Ath Madrid": "Atletico Madrid",
                "Betis": "Real Betis",
                "Sociedad": "Real Sociedad",
                "Elche": "Elche",
                "Levante": "Levante",
                "Oviedo": "Real Oviedo",
            },
        },
        2024: {
            "canonical": {
                "Real Madrid", "Barcelona", "Atletico Madrid", "Girona", "Real Sociedad",
                "Athletic Club", "Real Betis", "Villarreal", "Valencia", "Alaves",
                "Osasuna", "Getafe", "Celta Vigo", "Sevilla", "Mallorca",
                "Rayo Vallecano", "Las Palmas", "Valladolid", "Leganes", "Espanyol",
                "Elche", "Levante", "Real Oviedo", "Cadiz", "Granada CF", "Almeria", "Huesca", "Eibar",
            },
            "aliases": {
                "Espanyol": "Espanyol",
                "Espanol": "Espanyol",
                "Valladolid": "Valladolid",
                "Granada": "Granada CF",
                "Ath Bilbao": "Athletic Club",
                "Athletic Bilbao": "Athletic Club",
                "Alavés": "Alaves",
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
    },
    "SA": {
        2023: {
            "canonical": {
                "Inter", "AC Milan", "Juventus", "Atalanta", "Bologna",
                "AS Roma", "Lazio", "Fiorentina", "Torino", "Napoli",
                "Genoa", "Monza", "Verona", "Lecce", "Udinese",
                "Cagliari", "Empoli", "Frosinone", "Sassuolo", "Salernitana",
                "Spezia", "Sampdoria", "Venezia", "Cremonese", "Como", "Parma", "Pisa", "Benevento", "Crotone", "SPAL", "Brescia",
            },
            "aliases": {
                "Milan": "AC Milan",
                "Roma": "AS Roma",
                "Spal": "SPAL",
            },
        },
        2024: {
            "canonical": {
                "Inter", "AC Milan", "Juventus", "Atalanta", "Bologna",
                "AS Roma", "Lazio", "Fiorentina", "Torino", "Napoli",
                "Genoa", "Monza", "Verona", "Lecce", "Udinese", 
                "Cagliari", "Empoli", "Parma", "Como", "Venezia",
                "Pisa", "Frosinone", "Sassuolo", "Salernitana", "Spezia", "Sampdoria",
                "Cremonese", "Benevento", "Crotone", "SPAL", "Brescia",
                "Carpi", "Catania", "Cesena", "Chievo", "Livorno", "Palermo", "Pescara", "Siena",
            },
            "aliases": {
                "Milan": "AC Milan",
                "Roma": "AS Roma",
                "Spal": "SPAL",
            },
        },
    },
    "BL1": {
        2023: {
            "canonical": {
                "Bayer Leverkusen", "Bayern Munich", "VfB Stuttgart", "RB Leipzig", "Borussia Dortmund",
                "Eintracht Frankfurt", "TSG Hoffenheim", "FC Augsburg", "SC Freiburg", "Werder Bremen",
                "FC Heidenheim", "VfL Wolfsburg", "Mainz 05", "Borussia Monchengladbach", "Union Berlin",
                "VfL Bochum", "FC Koln", "Darmstadt 98",
                "Hamburger SV", "Hannover 96", "Hertha BSC", "FC Schalke 04", "Arminia Bielefeld", "Greuther Furth", "Fortuna Dusseldorf", "Paderborn",
                "FC St. Pauli", "Holstein Kiel",
            },
            "aliases": {
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
                "1. FC Köln": "FC Koln",
                "1. FC Koln": "FC Koln",
            },
        },
        2024: {
            "canonical": {
                "Bayer Leverkusen", "Bayern Munich", "VfB Stuttgart", "RB Leipzig", "Borussia Dortmund",
                "Eintracht Frankfurt", "TSG Hoffenheim", "FC Augsburg", "SC Freiburg", "Werder Bremen",
                "FC Heidenheim", "VfL Wolfsburg", "Mainz 05", "Borussia Monchengladbach", "Union Berlin",
                "VfL Bochum", "FC St. Pauli", "Holstein Kiel",
                "FC Koln", "Darmstadt 98", "Hamburger SV", "Hannover 96", "Hertha BSC", 
                "FC Schalke 04", "Arminia Bielefeld", "Greuther Furth", "Fortuna Dusseldorf", "Paderborn",
            },
            "aliases": {
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
                "1. FC Köln": "FC Koln",
                "1. FC Koln": "FC Koln",
            },
        },
    },
    "FL1": {
        2023: {
            "canonical": {
                "Paris Saint Germain", "AS Monaco", "Brest", "Lille", "Nice",
                "Lyon", "Lens", "Marseille", "Reims", "Rennes",
                "Toulouse", "Montpellier", "Strasbourg", "Le Havre", "Nantes",
                "Lorient", "Metz", "Clermont Foot",
                "Nimes", "Dijon", "Guingamp", "Amiens", "Caen", "Angers", "Saint-Etienne", "Bordeaux", "Troyes", "Auxerre", "Ajaccio", "Nancy", "Bastia", "Evian Thonon Gaillard", "Sochaux", "Valenciennes",
                "Paris FC",
            },
            "aliases": {
                "Evian": "Evian Thonon Gaillard",
                "Clermont": "Clermont Foot",
            },
        },
        2024: {
            "canonical": {
                "Paris Saint Germain", "AS Monaco", "Brest", "Lille", "Nice",
                "Lyon", "Lens", "Marseille", "Reims", "Rennes",
                "Toulouse", "Montpellier", "Strasbourg", "Le Havre", "Nantes",
                "Auxerre", "Angers", "Saint-Etienne",
                "Lorient", "Metz", "Clermont Foot", "Ajaccio", "Troyes", "Bordeaux",
                "Nimes", "Dijon", "Guingamp", "Amiens", "Caen", "Nancy", "Bastia", "Evian Thonon Gaillard", "Sochaux", "Valenciennes",
            },
            "aliases": {
                "PSG": "Paris Saint Germain",
                "Paris SG": "Paris Saint Germain",
                "Monaco": "AS Monaco",
                "St Etienne": "Saint-Etienne",
                "Clermont": "Clermont Foot",
            },
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
            mappings = LEAGUE_TEAMS[league].get(target_season)
            if not mappings:
                fallback_season = get_latest_season(league)
                logger.info(f"Season {season} not found for {league}. Falling back to {fallback_season}")
                mappings = LEAGUE_TEAMS[league][fallback_season]
            
            # Action 8.1: Case-Blind Lookup
            canonical_set = {c.upper() for c in mappings["canonical"]}
            aliases = {k.upper(): v.upper() for k, v in mappings["aliases"].items()}
            
            if clean in canonical_set: return clean
            if clean in aliases: return aliases[clean]
            
            result = process.extractOne(clean, list(canonical_set), score_cutoff=FUZZY_SCORE_CUTOFF)
            if result:
                canonical, score, index = result
                logger.info(f"Fuzzy matched '{clean}' -> '{canonical}' (Score: {score}, League: {league})")
                return canonical.upper()

    # CASE 2: No League OR No Match in League (Global Fallback)
    # Search all leagues exactly/alias first (Case-Blind)
    for l_code, s_map in LEAGUE_TEAMS.items():
        s_target = season or max(s_map.keys())
        m = s_map.get(s_target, s_map[max(s_map.keys())])
        c_set = {c.upper() for c in m["canonical"]}
        a_map = {k.upper(): v.upper() for k, v in m["aliases"].items()}
        if clean in c_set: return clean
        if clean in a_map: return a_map[clean]
    
    # Fuzzy match across target season across ALL leagues as last ditch
    all_canonicals = []
    for l_code, s_map in LEAGUE_TEAMS.items():
        s_target = season or max(s_map.keys())
        m = s_map.get(s_target, s_map[max(s_map.keys())])
        all_canonicals.extend([c.upper() for c in m["canonical"]])
        
    result = process.extractOne(clean, list(set(all_canonicals)), score_cutoff=FUZZY_SCORE_CUTOFF)
    if result:
        canonical, score, index = result
        logger.info(f"Global fuzzy match '{clean}' -> '{canonical}' (Score: {score})")
        return canonical.upper()
        
    # 4. Fallback
    logger.warning(f"Could not normalize team name: '{clean}' (League: {league or 'Global Context'}).")
    return clean.upper()

def generate_match_fingerprint(league: str, date_obj: datetime, home: str, away: str, season: Optional[int] = None) -> str:
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
    hash_obj = hashlib.md5(raw.encode('utf-8'))
    return hash_obj.hexdigest()[:16]

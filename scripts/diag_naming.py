from src.utils.naming import normalize_team_name, LEAGUE_TEAMS, get_latest_season
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.utils.naming")
logger.setLevel(logging.WARNING)

teams_to_test = [
    ("ESPANOL", "PD"),
    ("VALLADOLID", "PD"),
    ("CORDOBA", "PD"),
    ("MALAGA", "PD"),
    ("LA CORUNA", "PD"),
    ("PALERMO", "SA"),
    ("CHIEVO", "SA"),
    ("LIVORNO", "SA"),
    ("STOKE", "PL"),
    ("SWANSEA", "PL"),
    ("QPR", "PL"),
    ("WIGAN", "PL"),
]

print("--- Running Full Normalization Tests ---")
for team, league in teams_to_test:
    print(f"\nTesting '{team}' ({league}):")
    result = normalize_team_name(team, league=league)
    print(f"  Result: '{result}'")

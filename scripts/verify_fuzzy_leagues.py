
from src.cli.utils import resolve_league_code

test_cases = [
    # PL
    "premier", "Premier League", "EPL",
    # PD
    "la liga", "Spain", "Primera",
    # SA
    "serie a", "Italy", "Calcio",
    # BL1
    "bundesliga", "Germany",
    # FL1
    "ligue 1", "France", "Uber Eats"
]

print(f"{'Query':<20} | {'Result':<10}")
print("-" * 35)

for query in test_cases:
    res = resolve_league_code(query)
    print(f"{query:<20} | {res.value if res else 'None':<10}")

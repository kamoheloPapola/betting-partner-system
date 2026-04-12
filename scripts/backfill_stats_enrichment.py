from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.fetch_stats_enrichment import DATA_DIR, LEAGUE_IDS, enrich_league


def discover_league_seasons() -> list[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    pattern = re.compile(r"^([A-Z0-9]+)_(\d{4})\.csv$")
    for path in DATA_DIR.glob("*.csv"):
        match = pattern.match(path.name)
        if match is None:
            continue
        league, season = match.group(1), int(match.group(2))
        if league not in LEAGUE_IDS:
            continue
        pairs.add((league, season))
    return sorted(pairs)


def main() -> int:
    total_enriched = 0
    total_fixtures = 0
    pairs = discover_league_seasons()
    if not pairs:
        print("No processed match CSV seasons found.")
        return 0

    for league, season in pairs:
        enriched, fixtures = enrich_league(league, season, LEAGUE_IDS[league])
        total_enriched += enriched
        total_fixtures += fixtures
        print(f"{league} {season}: {enriched}/{fixtures} fixtures enriched")

    print(f"Total: {total_enriched}/{total_fixtures} enriched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

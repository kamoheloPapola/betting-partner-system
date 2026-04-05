import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

LEAGUES = {
    "PL": "PL",
    "BL1": "BL1",
    "FL1": "FL1",
    "SA": "SA",
    "PD": "PD",
}

DATA_DIR = "data"
SEASON_ROLLOVER_MONTH = 7
UPCOMING_STATUSES = {"SCHEDULED", "TIMED", "POSTPONED"}


def resolve_season_year(now: datetime | None = None) -> int:
    """Return the football-data season start year.

    Uses FOOTBALL_DATA_SEASON when provided. Otherwise infer the current
    European football season and roll over in July.
    """
    override = os.getenv("FOOTBALL_DATA_SEASON")
    if override:
        return int(override)

    current = now or datetime.now(timezone.utc)
    return current.year if current.month >= SEASON_ROLLOVER_MONTH else current.year - 1


def fetch_matches(competition_code: str, season_year: int) -> list:
    """Fetch all matches for the target season."""
    url = f"{BASE}/competitions/{competition_code}/matches"
    params = {"season": season_year}
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("matches", [])


def matches_to_df(matches: list, league: str, season_year: int) -> pd.DataFrame:
    """Normalise football-data.org match response to our CSV schema."""
    rows = []
    for match in matches:
        score = match.get("score", {})
        full = score.get("fullTime", {})
        half = score.get("halfTime", {})
        rows.append(
            {
                "league": league,
                "season": season_year,
                "date": match.get("utcDate", "")[:10],
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "home_goals": full.get("home"),
                "away_goals": full.get("away"),
                "ht_home": half.get("home"),
                "ht_away": half.get("away"),
                "status": match.get("status", ""),
                "matchday": match.get("matchday"),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    season_year = resolve_season_year()
    season_label = f"{season_year}/{str(season_year + 1)[-2:]}"

    for league, code in LEAGUES.items():
        print(f"Fetching {league} {season_label}...", end=" ")
        try:
            matches = fetch_matches(code, season_year)
            df = matches_to_df(matches, league, season_year)
            finished = df[df["status"] == "FINISHED"]
            upcoming = df[df["status"].isin(UPCOMING_STATUSES)]

            out_path = os.path.join(DATA_DIR, f"{league}_{season_year}.csv")
            finished.to_csv(out_path, index=False)

            up_path = os.path.join(DATA_DIR, f"{league}_upcoming.csv")
            upcoming.to_csv(up_path, index=False)

            print(f"OK - {len(finished)} finished, {len(upcoming)} upcoming")
        except Exception as exc:
            print(f"FAILED - {exc}")

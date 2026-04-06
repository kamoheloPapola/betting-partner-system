import os
import sys
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

LEAGUES = {"PL": "PL", "BL1": "BL1", "FL1": "FL1", "SA": "SA", "PD": "PD"}

# Auto-detect current football season (rolls over July 1)
def current_season_year() -> int:
    override = os.getenv("FOOTBALL_DATA_SEASON")
    if override:
        return int(override)
    today = datetime.now()
    return today.year if today.month >= 7 else today.year - 1

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "matches"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_matches(competition_code: str, season: int) -> list:
    url = f"{BASE}/competitions/{competition_code}/matches"
    r = requests.get(url, headers=HEADERS, params={"season": season}, timeout=30)
    r.raise_for_status()
    return r.json().get("matches", [])

def matches_to_df(matches: list, league: str, season: int) -> pd.DataFrame:
    rows = []
    for m in matches:
        score = m.get("score", {})
        full = score.get("fullTime", {})
        half = score.get("halfTime", {})
        home_goals = full.get("home")
        away_goals = full.get("away")
        result = ""
        if home_goals is not None and away_goals is not None:
            result = "H" if home_goals > away_goals else "A" if away_goals > home_goals else "D"
        rows.append({
            "date":        m.get("utcDate", "")[:10],
            "home_team":   m["homeTeam"]["name"].upper(),
            "away_team":   m["awayTeam"]["name"].upper(),
            "home_score":  home_goals,
            "away_score":  away_goals,
            "result":      result,
            "home_corners": None,
            "away_corners": None,
            "home_yellow_cards": None,
            "away_yellow_cards": None,
            "home_red_cards": None,
            "away_red_cards": None,
            "season":      str(season),
            "league":      league,
            "status":      m.get("status", ""),
            "source":      "football_data_api",
            "home_total_cards": None,
            "away_total_cards": None,
            "match_total_cards": None,
            "total_corners": None,
            "match_id":    format(abs(hash(f"{m.get('utcDate','')}_{m['homeTeam']['name']}_{m['awayTeam']['name']}")), 'x')[:16],
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    season = current_season_year()
    for league, code in LEAGUES.items():
        print(f"Fetching {league} {season}/{str(season+1)[2:]}...", end=" ")
        try:
            matches = fetch_matches(code, season)
            df = matches_to_df(matches, league, season)
            finished = df[df["status"] == "FINISHED"]
            upcoming = df[df["status"].isin(["SCHEDULED", "TIMED", "POSTPONED"])]

            finished.to_csv(DATA_DIR / f"{league}_{season}.csv", index=False)
            upcoming.to_csv(DATA_DIR / f"{league}_upcoming.csv", index=False)

            print(f"OK - {len(finished)} finished, {len(upcoming)} upcoming")
        except Exception as e:
            print(f"FAILED - {e}")

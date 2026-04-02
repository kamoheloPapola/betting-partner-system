import os, requests, pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

LEAGUES = {
    "PL":  "PL",
    "BL1": "BL1",
    "FL1": "FL1",
    "SA":  "SA",
    "PD":  "PD",
}

DATA_DIR = "data"

def fetch_matches(competition_code: str) -> list:
    """Fetch all matches for the current season."""
    url = f"{BASE}/competitions/{competition_code}/matches"
    params = {"season": 2024}  # 2024 = 2024/25 season on football-data.org
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("matches", [])

def matches_to_df(matches: list, league: str) -> pd.DataFrame:
    """Normalise football-data.org match response to our CSV schema."""
    rows = []
    for m in matches:
        score = m.get("score", {})
        full = score.get("fullTime", {})
        half = score.get("halfTime", {})
        rows.append({
            "league":      league,
            "season":      "2025",
            "date":        m.get("utcDate", "")[:10],
            "home_team":   m["homeTeam"]["name"],
            "away_team":   m["awayTeam"]["name"],
            "home_goals":  full.get("home"),
            "away_goals":  full.get("away"),
            "ht_home":     half.get("home"),
            "ht_away":     half.get("away"),
            "status":      m.get("status", ""),
            "matchday":    m.get("matchday"),
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    for league, code in LEAGUES.items():
        print(f"Fetching {league}...", end=" ")
        try:
            matches = fetch_matches(code)
            df = matches_to_df(matches, league)
            finished = df[df["status"] == "FINISHED"]
            upcoming = df[df["status"].isin(["SCHEDULED", "TIMED", "POSTPONED"])]
            
            # Overwrite season file
            out_path = os.path.join(DATA_DIR, f"{league}_2025.csv")
            finished.to_csv(out_path, index=False)
            
            # Overwrite upcoming file
            up_path = os.path.join(DATA_DIR, f"{league}_upcoming.csv")
            upcoming.to_csv(up_path, index=False)
            
            print(f"OK — {len(finished)} finished, {len(upcoming)} upcoming")
        except Exception as e:
            print(f"FAILED — {e}")

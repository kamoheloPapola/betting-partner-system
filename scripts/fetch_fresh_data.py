import os
import re
import subprocess
import sys
import unicodedata
import requests
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
from src.utils.naming import generate_match_fingerprint
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
if API_KEY is None:
    print(
        "ERROR: FOOTBALL_DATA_API_KEY is not set. "
        "Add it to your .env file.",
        file=sys.stderr,
    )
    raise SystemExit(1)
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
SCRIPTS_DIR = Path(__file__).resolve().parent

REQUIRED_COLUMNS = [
    "date", "home_team", "away_team", "home_score", "away_score",
    "result", "season", "league", "status", "source", "match_id",
]
MIN_FINISHED_ROWS = 5
FETCH_RETRIES = 2
FETCH_RETRY_DELAY = 60  # seconds


def clean_team_name(name: str) -> str:
    """Normalize team name to match pipeline sanitizer rules."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.upper()
    name = name.replace("&", "AND")
    name = re.sub(r"\bF\.?C\.?\b", "", name)
    name = re.sub(r"\bA\.?F\.?C\.?\b", "", name)
    name = re.sub(r"[^A-Z0-9 \-_\.]", "", name)
    name = re.sub(r" +", " ", name).strip()
    return name

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
            "home_team":   clean_team_name(m["homeTeam"]["name"]),
            "away_team":   clean_team_name(m["awayTeam"]["name"]),
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
            "match_id":    generate_match_fingerprint(
                league=league,
                date_obj=datetime.fromisoformat(m.get("utcDate", "2000-01-01T00:00:00Z").replace("Z", "+00:00")),
                home=m["homeTeam"]["name"],
                away=m["awayTeam"]["name"],
                season=season,
            ),
        })
    return pd.DataFrame(rows)


def validate_df(df: pd.DataFrame, label: str) -> bool:
    """Return True if df passes schema and minimum row checks."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"  VALIDATION FAILED ({label}): missing columns {missing}")
        return False
    if len(df) < MIN_FINISHED_ROWS and label.endswith("finished"):
        print(f"  VALIDATION FAILED ({label}): only {len(df)} rows (min {MIN_FINISHED_ROWS})")
        return False
    return True


def atomic_write(df: pd.DataFrame, target: Path) -> None:
    """Write df to a temp file then rename to target atomically."""
    tmp = target.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(target)


def run_post_fetch_script(script_name: str) -> None:
    script_path = SCRIPTS_DIR / script_name
    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, str(script_path)], check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    season = current_season_year()
    fetch_failed = False
    post_fetch_ready = True
    for league, code in LEAGUES.items():
        print(f"Fetching {league} {season}/{str(season+1)[2:]}...", end=" ")
        success = False
        upcoming_ready = False
        for attempt in range(1, FETCH_RETRIES + 1):
            try:
                matches = fetch_matches(code, season)
                df = matches_to_df(matches, league, season)
                finished = df[df["status"] == "FINISHED"]
                upcoming = df[df["status"].isin(["SCHEDULED", "TIMED", "POSTPONED"])]

                finished_target = DATA_DIR / f"{league}_{season}.csv"
                upcoming_target = DATA_DIR / f"{league}_upcoming.csv"

                if not validate_df(upcoming, f"{league} upcoming"):
                    raise ValueError("Upcoming data failed validation")

                atomic_write(upcoming, upcoming_target)
                upcoming_ready = True

                if len(finished) == 0:
                    print(f"OK - 0 finished, {len(upcoming)} upcoming")
                    success = True
                    break

                if not validate_df(finished, f"{league} finished"):
                    raise ValueError("Finished data failed validation")

                atomic_write(finished, finished_target)
                print(f"OK - {len(finished)} finished, {len(upcoming)} upcoming")
                success = True
                break
            except Exception as e:
                if attempt < FETCH_RETRIES:
                    print(f"FAILED (attempt {attempt}/{FETCH_RETRIES}) - {e} - retrying in {FETCH_RETRY_DELAY}s...")
                    time.sleep(FETCH_RETRY_DELAY)
                else:
                    print(f"FAILED (attempt {attempt}/{FETCH_RETRIES}) - {e} - skipping, last-known-good preserved")

        if not success:
            fetch_failed = True
            if not upcoming_ready:
                post_fetch_ready = False

    if post_fetch_ready:
        run_post_fetch_script("sync_season_maps.py")
        run_post_fetch_script("validate_team_names.py")

    return 1 if fetch_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

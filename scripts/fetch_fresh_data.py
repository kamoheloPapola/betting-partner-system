import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv

from src.utils.naming import generate_match_fingerprint, normalize_team_name

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

LEAGUES = {
    "PL": "Premier League",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
    "SA": "Serie A",
    "PD": "La Liga",
}

DATA_DIR = Path("data")
PROCESSED_MATCHES_DIR = DATA_DIR / "processed" / "matches"
SEASON_ROLLOVER_MONTH = 7
UPCOMING_STATUSES = {"SCHEDULED", "TIMED", "POSTPONED"}
FINISHED_STATUS = "FINISHED"
FINISHED_COLUMNS = [
    "date",
    "kickoff_utc",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "result",
    "league",
    "status",
    "source",
    "season",
    "match_id",
    "competition",
]
UPCOMING_COLUMNS = [
    "match_id",
    "date",
    "kickoff_utc",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "competition",
    "league",
    "season",
    "status",
    "source",
]


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


def fetch_matches(competition_code: str, season_year: int) -> List[Dict[str, Any]]:
    """Fetch all matches for the target season."""
    url = f"{BASE}/competitions/{competition_code}/matches"
    params = {"season": season_year}
    response = requests.get(url, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    return response.json().get("matches", [])


def _canonical_team(name: str, league: str, season_year: int) -> str:
    return normalize_team_name(name, league=league, season=season_year)


def _match_id(league: str, kickoff: pd.Timestamp, home_team: str, away_team: str, season_year: int) -> str:
    kickoff_dt = kickoff.to_pydatetime()
    return generate_match_fingerprint(
        league,
        kickoff_dt,
        home_team,
        away_team,
        season=season_year,
    )


def _result_label(home_score: Any, away_score: Any) -> str | None:
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return "H"
    if away_score > home_score:
        return "A"
    return "D"


def _build_finished_rows(matches: List[Dict[str, Any]], league: str, season_year: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in matches:
        if match.get("status") != FINISHED_STATUS:
            continue

        kickoff = pd.to_datetime(match.get("utcDate"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue

        score = match.get("score", {})
        full = score.get("fullTime", {})

        home_team = _canonical_team(match["homeTeam"]["name"], league, season_year)
        away_team = _canonical_team(match["awayTeam"]["name"], league, season_year)
        home_score = full.get("home")
        away_score = full.get("away")

        rows.append(
            {
                "date": kickoff.strftime("%Y-%m-%d"),
                "kickoff_utc": kickoff.isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "result": _result_label(home_score, away_score),
                "league": league,
                "status": "finished",
                "source": "football-data.org",
                "season": season_year,
                "match_id": _match_id(league, kickoff, home_team, away_team, season_year),
                "competition": league,
            }
        )
    return rows


def _build_upcoming_rows(matches: List[Dict[str, Any]], league: str, season_year: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    competition_name = LEAGUES[league]

    for match in matches:
        if match.get("status") not in UPCOMING_STATUSES:
            continue

        kickoff = pd.to_datetime(match.get("utcDate"), utc=True, errors="coerce")
        if pd.isna(kickoff):
            continue

        home_team = _canonical_team(match["homeTeam"]["name"], league, season_year)
        away_team = _canonical_team(match["awayTeam"]["name"], league, season_year)

        rows.append(
            {
                "match_id": _match_id(league, kickoff, home_team, away_team, season_year),
                "date": kickoff.strftime("%Y-%m-%d"),
                "kickoff_utc": kickoff.isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": None,
                "away_team_id": None,
                "competition": competition_name,
                "league": league,
                "season": season_year,
                "status": match.get("status"),
                "source": "football-data.org",
            }
        )
    return rows


def _write_outputs(df: pd.DataFrame, *paths: Path) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_MATCHES_DIR.mkdir(parents=True, exist_ok=True)
    season_year = resolve_season_year()
    season_label = f"{season_year}/{str(season_year + 1)[-2:]}"

    for league in LEAGUES:
        print(f"Fetching {league} {season_label}...", end=" ")
        try:
            matches = fetch_matches(league, season_year)

            finished_df = pd.DataFrame(_build_finished_rows(matches, league, season_year), columns=FINISHED_COLUMNS)
            upcoming_df = pd.DataFrame(_build_upcoming_rows(matches, league, season_year), columns=UPCOMING_COLUMNS)

            _write_outputs(
                finished_df,
                DATA_DIR / f"{league}_{season_year}.csv",
                PROCESSED_MATCHES_DIR / f"{league}_{season_year}.csv",
            )
            _write_outputs(
                upcoming_df,
                DATA_DIR / f"{league}_upcoming.csv",
                PROCESSED_MATCHES_DIR / f"{league}_upcoming.csv",
            )

            print(f"OK - {len(finished_df)} finished, {len(upcoming_df)} upcoming")
        except Exception as exc:
            print(f"FAILED - {exc}")

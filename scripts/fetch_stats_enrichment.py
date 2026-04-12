from __future__ import annotations

import logging
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

API_BASE = "https://v3.football.api-sports.io"
LEAGUE_IDS = {"PL": 39, "BL1": 78, "FL1": 61, "SA": 135, "PD": 140}
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "matches"
MASTER_PATH = PROJECT_ROOT / "data" / "results" / "normalized" / "results_master.csv"
LOOKBACK_DAYS = 4
REQUEST_DELAY = 1.2

API_KEY = os.getenv("API_FOOTBALL_KEY")
if API_KEY is None:
    print("ERROR: API_FOOTBALL_KEY is not set in the environment.", file=sys.stderr)
    raise SystemExit(1)

HEADERS = {"x-apisports-key": API_KEY}

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def extract_stats(response_list: list[dict[str, Any]]) -> dict[str, int | None]:
    """Extract corners and cards from API-Football fixture statistics."""

    def get_value(team_stats: dict[str, Any] | None, stat_key: str, *, cards: bool) -> int | None:
        if not team_stats:
            return 0 if cards else None
        for item in team_stats.get("statistics", []):
            if item.get("type") != stat_key:
                continue
            value = item.get("value")
            if value is None:
                return 0 if cards else None
            try:
                return int(float(str(value).replace("%", "").strip()))
            except (TypeError, ValueError):
                logger.warning("Could not parse %s stat value %r", stat_key, value)
                return 0 if cards else None
        return 0 if cards else None

    home = response_list[0] if len(response_list) > 0 else None
    away = response_list[1] if len(response_list) > 1 else None

    return {
        "home_corners": get_value(home, "Corner Kicks", cards=False),
        "away_corners": get_value(away, "Corner Kicks", cards=False),
        "home_yellow": get_value(home, "Yellow Cards", cards=True),
        "away_yellow": get_value(away, "Yellow Cards", cards=True),
        "home_red": get_value(home, "Red Cards", cards=True),
        "away_red": get_value(away, "Red Cards", cards=True),
    }


def normalize_for_join(name: str) -> str:
    """Normalize team names for exact joins across API and local CSV names."""
    text = "" if name is None else str(name)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.upper().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(
        r"\b(?:FC|AFC|CF|SC|SV|TSV|FSV|AC|AS|SS|US|CD|UD|RCD|RC|SD|CA|CE)\b",
        " ",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_match_row(
    df: pd.DataFrame,
    api_date: str,
    api_home: str,
    api_away: str,
    date_col: str,
    home_col: str,
    away_col: str,
) -> int | None:
    """Return the matching dataframe row index, or None if no exact normalized match exists."""
    normalized_home = normalize_for_join(api_home)
    normalized_away = normalize_for_join(api_away)

    for idx, row in df.iterrows():
        row_date = str(row.get(date_col, ""))
        if not row_date.startswith(api_date):
            continue
        row_home = normalize_for_join(row.get(home_col, ""))
        row_away = normalize_for_join(row.get(away_col, ""))
        if row_home == normalized_home and row_away == normalized_away:
            return int(idx)
    return None


def _request_json(url: str, params: dict[str, Any]) -> dict[str, Any] | None:
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("API-Football request failed for %s: %s", url, exc)
        return None

    try:
        return response.json()
    except ValueError as exc:
        logger.warning("API-Football returned invalid JSON for %s: %s", url, exc)
        return None


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(path)


def _zero(value: int | None) -> int:
    return 0 if value is None or pd.isna(value) else int(value)


def enrich_league(league: str, season: int, api_league_id: int) -> tuple[int, int]:
    processed_path = DATA_DIR / f"{league}_{season}.csv"
    if not processed_path.exists():
        logger.warning("Processed CSV missing for %s %s: %s", league, season, processed_path)
        return (0, 0)

    processed_df = pd.read_csv(processed_path, encoding="utf-8")

    master_df: pd.DataFrame | None = None
    master_mutated = False
    if MASTER_PATH.exists():
        master_df = pd.read_csv(MASTER_PATH, encoding="utf-8")

    today = datetime.now(timezone.utc).date()
    from_date = today - timedelta(days=LOOKBACK_DAYS)
    to_date = today

    fixtures_payload = _request_json(
        f"{API_BASE}/fixtures",
        {
            "league": api_league_id,
            "season": season,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "status": "FT",
        },
    )
    if fixtures_payload is None:
        return (0, 0)
    if fixtures_payload.get("results", 0) == 0:
        logger.info("No finished fixtures for %s %s in %s to %s", league, season, from_date, to_date)
        return (0, 0)

    fixtures = fixtures_payload.get("response", [])
    enriched_count = 0

    for fixture in fixtures:
        try:
            fix_date = fixture["fixture"]["date"][:10]
            api_home = fixture["teams"]["home"]["name"]
            api_away = fixture["teams"]["away"]["name"]
            fixture_id = fixture["fixture"]["id"]
        except (KeyError, TypeError) as exc:
            logger.warning("Skipping malformed fixture for %s %s: %s", league, season, exc)
            continue

        idx = find_match_row(
            processed_df,
            fix_date,
            api_home,
            api_away,
            date_col="date",
            home_col="home_team",
            away_col="away_team",
        )
        if idx is None:
            logger.warning("No CSV match for %s vs %s %s", api_home, api_away, fix_date)
            continue

        if "total_corners" in processed_df.columns:
            existing_corners = processed_df.at[idx, "total_corners"]
            if pd.notna(existing_corners) and str(existing_corners).strip() != "":
                continue

        time.sleep(REQUEST_DELAY)
        stats_payload = _request_json(
            f"{API_BASE}/fixtures/statistics",
            {"fixture": fixture_id},
        )
        if stats_payload is None or stats_payload.get("results", 0) == 0:
            logger.warning("No statistics returned for fixture %s", fixture_id)
            continue

        stats = extract_stats(stats_payload.get("response", []))
        if stats["home_corners"] is None and stats["away_corners"] is None:
            logger.warning("No corner data for fixture %s", fixture_id)
            continue

        home_yellow = _zero(stats["home_yellow"])
        away_yellow = _zero(stats["away_yellow"])
        home_red = _zero(stats["home_red"])
        away_red = _zero(stats["away_red"])
        home_total_cards = home_yellow + home_red
        away_total_cards = away_yellow + away_red

        if stats["home_corners"] is not None and stats["away_corners"] is not None:
            total_corners = stats["home_corners"] + stats["away_corners"]
        else:
            total_corners = None

        processed_df.at[idx, "total_corners"] = total_corners
        processed_df.at[idx, "home_corners"] = stats["home_corners"]
        processed_df.at[idx, "away_corners"] = stats["away_corners"]
        processed_df.at[idx, "match_total_cards"] = home_total_cards + away_total_cards
        processed_df.at[idx, "home_yellow_cards"] = stats["home_yellow"]
        processed_df.at[idx, "away_yellow_cards"] = stats["away_yellow"]
        processed_df.at[idx, "home_red_cards"] = stats["home_red"]
        processed_df.at[idx, "away_red_cards"] = stats["away_red"]
        processed_df.at[idx, "home_total_cards"] = home_total_cards
        processed_df.at[idx, "away_total_cards"] = away_total_cards

        if master_df is not None:
            master_idx = find_match_row(
                master_df,
                fix_date,
                api_home,
                api_away,
                date_col="kickoff_date_utc",
                home_col="home_team",
                away_col="away_team",
            )
            if master_idx is not None:
                master_df.at[master_idx, "home_corners"] = stats["home_corners"]
                master_df.at[master_idx, "away_corners"] = stats["away_corners"]
                master_df.at[master_idx, "home_cards"] = home_total_cards
                master_df.at[master_idx, "away_cards"] = away_total_cards
                master_mutated = True

        enriched_count += 1

    _atomic_write_csv(processed_df, processed_path)
    if master_df is not None and master_mutated:
        _atomic_write_csv(master_df, MASTER_PATH)

    return (enriched_count, len(fixtures))


def current_season_year() -> int:
    override = os.getenv("FOOTBALL_DATA_SEASON")
    if override:
        return int(override)
    today = datetime.now()
    return today.year if today.month >= 7 else today.year - 1


def main() -> int:
    try:
        season = current_season_year()
        total_enriched = 0
        total_fixtures = 0
        for league, api_league_id in LEAGUE_IDS.items():
            enriched, fixtures = enrich_league(league, season, api_league_id)
            total_enriched += enriched
            total_fixtures += fixtures
            print(f"{league}: {enriched}/{fixtures} fixtures enriched")
        print(f"Total: {total_enriched}/{total_fixtures} enriched")
        return 0 if total_enriched >= 0 else 1
    except Exception as exc:
        logger.error("Unhandled exception in stats enrichment: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""
Legacy single-league fetch adapter over the current season fetcher stack.
"""

from __future__ import annotations

from pathlib import Path

from src.fetch.fetch_latest_season_csvs import SeasonFetcher

__all__ = ["FootballDataFetcher"]


class FootballDataFetcher:
    """Compatibility wrapper for the `fetch-data` CLI command."""

    def __init__(self) -> None:
        self.fetcher = SeasonFetcher()

    @staticmethod
    def _normalize_season_token(season: int | str) -> str:
        season_str = str(season)
        if len(season_str) == 4 and season_str.isdigit():
            # Accept either YYZZ tokens ("2526") or start years ("2024").
            if int(season_str[:2]) >= 20 and int(season_str) >= 2000:
                start = int(season_str) % 100
                return f"{start:02d}{(start + 1) % 100:02d}"
            return season_str

        raise ValueError(
            f"Invalid season value: {season}. Expected start year (2024) or YYZZ token (2526)."
        )

    def fetch_league_season(self, league: str, season: int | str) -> Path:
        season_token = self._normalize_season_token(season)
        league_code = self.fetcher.league_map.get(league)
        if not league_code:
            raise ValueError(f"Unsupported league for football-data fetch: {league}")

        url = f"{self.fetcher.BASE_URL}/{season_token}/{league_code}.csv"
        session = self.fetcher._get_session()
        response = session.get(url, timeout=self.fetcher.timeout)
        response.raise_for_status()

        league_dir = self.fetcher.output_dir / league
        league_dir.mkdir(parents=True, exist_ok=True)
        target_path = league_dir / f"season_{season_token}.csv"
        target_path.write_bytes(response.content)
        return target_path

"""Regenerate season membership from processed match CSVs.

This script merges inferred canonical names into the existing
`data/processed/season_maps.json` file instead of rebuilding it from scratch,
so tracked manual canonical additions remain preserved across sync runs.
"""

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import naming
from src.utils.naming import normalize_team_name


ROOT = Path(__file__).resolve().parent.parent
MATCHES_DIR = ROOT / "data" / "processed" / "matches"
SEASON_MAPS_PATH = ROOT / "data" / "processed" / "season_maps.json"
FILE_PATTERN = re.compile(r"^(?P<league>[A-Z0-9]+)_(?P<season>\d{4})\.csv$")
UNRESOLVED_PREFIX = "Could not normalize team name:"
SEASON_SEEDING_THRESHOLD = 10


class WarningCaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()


def _iter_match_files() -> list[Path]:
    files: list[Path] = []
    for csv_path in sorted(MATCHES_DIR.glob("*.csv")):
        if FILE_PATTERN.match(csv_path.name):
            files.append(csv_path)
    return files


def _load_existing_maps() -> dict[str, dict[str, list[str]]]:
    if not SEASON_MAPS_PATH.exists():
        return {}
    with SEASON_MAPS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_case_lookup() -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = defaultdict(dict)

    for league, seasons in naming.LEAGUE_TEAMS.items():
        for team_names in seasons.values():
            for team_name in team_names:
                lookup[league].setdefault(team_name.upper(), team_name)

    for league, seasons in getattr(naming, "LEAGUE_ALIASES", {}).items():
        for aliases in seasons.values():
            for canonical_name in aliases.values():
                lookup[league].setdefault(canonical_name.upper(), canonical_name)

    return dict(lookup)


def _unique_team_names(df: pd.DataFrame) -> list[str]:
    combined = pd.concat([df["home_team"], df["away_team"]], ignore_index=True)
    return [str(name).strip() for name in pd.unique(combined.dropna()) if str(name).strip()]


def _team_name_counts(df: pd.DataFrame) -> dict[str, int]:
    combined = pd.concat([df["home_team"], df["away_team"]], ignore_index=True)
    counts: dict[str, int] = {}
    for name in combined.dropna():
        clean_name = str(name).strip().upper()
        if clean_name:
            counts[clean_name] = counts.get(clean_name, 0) + 1
    return counts


def _warning_emitted(handler: WarningCaptureHandler, clean_name: str) -> bool:
    needle = f"'{clean_name}'"
    return any(
        UNRESOLVED_PREFIX in record.getMessage() and needle in record.getMessage()
        for record in handler.records
    )


def _reset_warning_state(league: str, clean_name: str) -> None:
    warned = getattr(naming, "_normalize_warned", None)
    if warned is not None:
        warned.discard((league, clean_name))


def _resolve_without_warning(
    handler: WarningCaptureHandler,
    raw_name: str,
    league: str,
    season: int,
) -> tuple[str, bool]:
    clean_name = raw_name.strip().upper()
    handler.clear()
    _reset_warning_state(league, clean_name)
    resolved_name = normalize_team_name(raw_name, league=league, season=season)
    warned = _warning_emitted(handler, clean_name)
    return resolved_name, warned


def _requires_season_membership(
    handler: WarningCaptureHandler,
    raw_name: str,
    league: str,
    season: int,
    resolved_name: str,
) -> bool:
    season_names = naming.LEAGUE_TEAMS.get(league, {}).get(season)
    if not season_names or resolved_name not in season_names:
        return False

    season_names.remove(resolved_name)
    try:
        retry_resolved, retry_warned = _resolve_without_warning(handler, raw_name, league, season)
    finally:
        season_names.add(resolved_name)

    if retry_warned:
        return True
    return retry_resolved.upper() != resolved_name.upper()


def _is_strict_season_match(
    handler: WarningCaptureHandler,
    raw_name: str,
    clean_name: str,
    league: str,
    season: int,
    resolved_name: str,
    canonical_name: str,
    season_specific: dict[str, str],
    team_counts: dict[str, int],
) -> bool:
    if clean_name not in season_specific:
        return False

    if resolved_name.upper() != clean_name:
        return False

    if team_counts.get(clean_name, 0) <= 1 and not _requires_season_membership(
        handler,
        raw_name,
        league,
        season,
        canonical_name,
    ):
        return False

    return True


def _sorted_maps(maps_data: dict[str, dict[str, list[str]]]) -> dict[str, dict[str, list[str]]]:
    sorted_maps: dict[str, dict[str, list[str]]] = {}
    for league in sorted(maps_data):
        seasons = maps_data[league]
        sorted_maps[league] = {}
        for season_key in sorted(seasons, key=int):
            sorted_maps[league][season_key] = sorted(set(seasons[season_key]), key=str.casefold)
    return sorted_maps


def main() -> int:
    existing_maps = _load_existing_maps()
    canonical_lookup = _canonical_case_lookup()
    summary: dict[tuple[str, int], dict[str, int]] = defaultdict(
        lambda: {"added": 0, "skipped": 0, "fallback": 0}
    )
    rule_mode: dict[tuple[str, int], str] = {}

    handler = WarningCaptureHandler()
    original_level = naming.logger.level
    original_propagate = naming.logger.propagate

    naming.logger.setLevel(logging.WARNING)
    naming.logger.propagate = False
    naming.logger.addHandler(handler)

    try:
        for csv_path in _iter_match_files():
            match = FILE_PATTERN.match(csv_path.name)
            if not match:
                continue

            league = match.group("league")
            season = int(match.group("season"))
            df = pd.read_csv(csv_path)
            if "home_team" not in df.columns or "away_team" not in df.columns:
                continue

            season_key = str(season)
            previous_names = set(existing_maps.get(league, {}).get(season_key, []))
            seeding_mode = len(previous_names) < SEASON_SEEDING_THRESHOLD
            rule_mode[(league, season)] = "seeding" if seeding_mode else "strict"
            written_names: set[str] = set()
            season_specific = {
                team_name.upper(): team_name
                for team_name in naming.LEAGUE_TEAMS.get(league, {}).get(season, set())
            }
            team_counts = _team_name_counts(df)

            for raw_name in _unique_team_names(df):
                clean_name = raw_name.strip().upper()
                resolved_name, warned = _resolve_without_warning(handler, raw_name, league, season)
                if warned:
                    summary[(league, season)]["skipped"] += 1
                    continue

                canonical_name = canonical_lookup.get(league, {}).get(resolved_name.upper())
                if not canonical_name:
                    summary[(league, season)]["skipped"] += 1
                    continue

                strict_match = _is_strict_season_match(
                    handler,
                    raw_name,
                    clean_name,
                    league,
                    season,
                    resolved_name,
                    canonical_name,
                    season_specific,
                    team_counts,
                )

                if not strict_match:
                    summary[(league, season)]["fallback"] += 1
                    if not seeding_mode:
                        continue

                written_names.add(canonical_name)

            league_map = existing_maps.setdefault(league, {})
            league_map[season_key] = sorted(written_names, key=str.casefold)
            summary[(league, season)]["added"] = len(written_names - previous_names)
    finally:
        naming.logger.removeHandler(handler)
        naming.logger.setLevel(original_level)
        naming.logger.propagate = original_propagate

    with SEASON_MAPS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(_sorted_maps(existing_maps), handle, indent=2)
        handle.write("\n")

    for league, season in sorted(summary, key=lambda item: (item[0][0], item[0][1])):
        counts = summary[(league, season)]
        mode = rule_mode[(league, season)]
        print(
            f"{league} {season} [{mode}]: added {counts['added']}, "
            f"fallback {counts['fallback']}, skipped {counts['skipped']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

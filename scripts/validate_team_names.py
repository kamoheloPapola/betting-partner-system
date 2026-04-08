import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import naming
from src.utils.naming import normalize_team_name


MATCHES_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "matches"
UNRESOLVED_PREFIX = "Could not normalize team name:"


class WarningCaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate normalized team names in processed match CSVs.")
    parser.add_argument("--league", help="Optional league code to validate, e.g. PD or PL.")
    return parser.parse_args()


def _iter_files(league: str | None) -> list[Path]:
    pattern = f"{league.upper()}*.csv" if league else "*.csv"
    return sorted(MATCHES_DIR.glob(pattern))


def _infer_league(csv_path: Path) -> str:
    return csv_path.stem.split("_", 1)[0].upper()


def _infer_season(csv_path: Path) -> int | None:
    match = re.search(r"_(\d{4})(?:_|$)", csv_path.stem)
    return int(match.group(1)) if match else None


def _unique_team_names(df: pd.DataFrame) -> list[str]:
    combined = pd.concat([df["home_team"], df["away_team"]], ignore_index=True)
    return [str(name).strip() for name in pd.unique(combined.dropna()) if str(name).strip()]


def _warning_emitted(handler: WarningCaptureHandler, clean_name: str) -> bool:
    needle = f"'{clean_name}'"
    return any(
        UNRESOLVED_PREFIX in record.getMessage() and needle in record.getMessage()
        for record in handler.records
    )


def main() -> int:
    args = _parse_args()
    files = _iter_files(args.league)
    unresolved_by_file: dict[str, list[str]] = defaultdict(list)

    handler = WarningCaptureHandler()
    original_level = naming.logger.level
    original_propagate = naming.logger.propagate

    naming.logger.setLevel(logging.WARNING)
    naming.logger.propagate = False
    naming.logger.addHandler(handler)

    try:
        for csv_path in files:
            df = pd.read_csv(csv_path)
            if "home_team" not in df.columns or "away_team" not in df.columns:
                continue

            league = _infer_league(csv_path)
            season = _infer_season(csv_path)

            for name in _unique_team_names(df):
                clean_name = name.strip().upper()
                handler.clear()

                warned = getattr(naming, "_normalize_warned", None)
                if warned is not None:
                    warned.discard((league, clean_name))

                normalize_team_name(name, league=league, season=season)

                if _warning_emitted(handler, clean_name):
                    unresolved_by_file[csv_path.name].append(clean_name)
    finally:
        naming.logger.removeHandler(handler)
        naming.logger.setLevel(original_level)
        naming.logger.propagate = original_propagate

    for file_name, unresolved in unresolved_by_file.items():
        for name in unresolved:
            print(f"UNRESOLVED in {file_name}: {name!r}")

    return 1 if unresolved_by_file else 0


if __name__ == "__main__":
    raise SystemExit(main())

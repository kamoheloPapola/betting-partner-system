"""
Unit tests for CanonicalDatasetBuilder Phase 1.

Tests the canonical dataset build pipeline: raw file discovery,
odds stripping, validation, deduplication, and coverage generation.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline.build_matches_dataset import (
    CanonicalDatasetBuilder,
    SCHEMA_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_raw_row(
    home: str = "Arsenal",
    away: str = "Chelsea",
    fthg: int = 2,
    ftag: int = 1,
    date: str = "15/09/2024",
    div: str = "E0",
    **extras,
) -> dict:
    row = {
        "Div": div,
        "Date": date,
        "HomeTeam": home,
        "AwayTeam": away,
        "FTHG": fthg,
        "FTAG": ftag,
        # Some odds columns to test dropping
        "B365H": 1.5,
        "BWH": 1.45,
        "MaxCH": 1.55,
    }
    row.update(extras)
    return row


@pytest.fixture
def source_dirs(tmp_path: Path) -> list[Path]:
    d = tmp_path / "raw"
    d.mkdir()
    # Create league subdirs
    (d / "PL").mkdir()
    (d / "SA").mkdir()
    return [d]


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "canonical"
    return d


def _write_csv(dest: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(dest, index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDiscovery:
    def test_discovers_files_in_league_dirs(self, source_dirs: list[Path], output_dir: Path):
        base = source_dirs[0]
        _write_csv(base / "PL" / "E0_2425.csv", [_make_raw_row()])
        _write_csv(base / "SA" / "I1_2425.csv", [_make_raw_row(div="I1")])

        builder = CanonicalDatasetBuilder(source_dirs=source_dirs, output_dir=output_dir)
        result = builder.build()

        assert result["status"] == "success"
        assert result["files_processed"] == 2

    def test_empty_source_dir(self, tmp_path: Path, output_dir: Path):
        builder = CanonicalDatasetBuilder(source_dirs=[tmp_path / "empty"], output_dir=output_dir)
        result = builder.build()

        assert result["status"] == "empty"


class TestParsingAndNormalisation:
    def test_strips_odds_columns(self, source_dirs: list[Path], output_dir: Path):
        _write_csv(source_dirs[0] / "PL" / "E0_2425.csv", [_make_raw_row()])

        builder = CanonicalDatasetBuilder(source_dirs=source_dirs, output_dir=output_dir)
        builder.build()

        df = pd.read_csv(output_dir / "matches.csv")
        odds_cols = [c for c in df.columns if "B365" in c or "BW" in c or "Max" in c]
        assert len(odds_cols) == 0
        assert "home_goals" in df.columns
        assert "away_goals" in df.columns

    def test_rejects_files_missing_required_raw_columns(self, source_dirs: list[Path], output_dir: Path):
        bad_row = _make_raw_row()
        del bad_row["FTHG"]
        _write_csv(source_dirs[0] / "PL" / "BAD.csv", [bad_row])
        _write_csv(source_dirs[0] / "PL" / "GOOD.csv", [_make_raw_row()])

        builder = CanonicalDatasetBuilder(source_dirs=source_dirs, output_dir=output_dir)
        result = builder.build()

        assert result["files_processed"] == 1
        assert "BAD.csv" in result["files_skipped"]

    def test_drops_future_matches(self, source_dirs: list[Path], output_dir: Path):
        # One valid past match, one future match
        rows = [
            _make_raw_row(home="A", date="01/01/2020"),
            _make_raw_row(home="B", date="01/01/2099"),
        ]
        _write_csv(source_dirs[0] / "PL" / "E0_2425.csv", rows)

        builder = CanonicalDatasetBuilder(source_dirs=source_dirs, output_dir=output_dir)
        result = builder.build()

        assert result["total_matches"] == 1
        df = pd.read_csv(output_dir / "matches.csv")
        assert df["home_team"].iloc[0] == "A"


class TestDeduplication:
    def test_removes_duplicate_match_hashes(self, source_dirs: list[Path], output_dir: Path):
        # Same match details will produce same match_hash
        rows = [
            _make_raw_row(fthg=2),
            _make_raw_row(fthg=3),  # Duplicate match, different score
        ]
        _write_csv(source_dirs[0] / "PL" / "E0.csv", rows)

        builder = CanonicalDatasetBuilder(source_dirs=source_dirs, output_dir=output_dir)
        result = builder.build()

        assert result["total_matches"] == 1
        assert result["duplicates_removed"] == 1


class TestOutput:
    def test_writes_matches_csv(self, source_dirs: list[Path], output_dir: Path):
        _write_csv(source_dirs[0] / "PL" / "E0.csv", [_make_raw_row(HS=10, AS=5)])

        builder = CanonicalDatasetBuilder(source_dirs=source_dirs, output_dir=output_dir)
        builder.build()

        assert (output_dir / "matches.csv").exists()
        df = pd.read_csv(output_dir / "matches.csv")
        assert len(df) == 1
        assert "match_hash" in df.columns
        assert "home_shots" in df.columns
        assert df["home_shots"].iloc[0] == 10

    def test_writes_coverage_json(self, source_dirs: list[Path], output_dir: Path):
        _write_csv(source_dirs[0] / "PL" / "E0.csv", [
            _make_raw_row(date="01/08/2024"),
            _make_raw_row(home="Other", date="15/08/2024"),
        ])

        builder = CanonicalDatasetBuilder(source_dirs=source_dirs, output_dir=output_dir)
        builder.build()

        cov_path = output_dir / "matches_coverage.json"
        assert cov_path.exists()

        with open(cov_path) as f:
            cov = json.load(f)

        assert cov["schema_version"] == SCHEMA_VERSION
        assert cov["total_matches"] == 2
        assert cov["total_leagues"] == 1
        assert "PL" in cov["leagues"]
        assert cov["leagues"]["PL"]["match_count"] == 2
        assert "build_timestamp" in cov

    def test_sorted_chronologically(self, source_dirs: list[Path], output_dir: Path):
        rows = [
            _make_raw_row(date="15/01/2024", home="Late"),
            _make_raw_row(date="10/08/2023", home="Early"),
        ]
        _write_csv(source_dirs[0] / "PL" / "E0_2425.csv", rows)

        builder = CanonicalDatasetBuilder(source_dirs=source_dirs, output_dir=output_dir)
        builder.build()

        df = pd.read_csv(output_dir / "matches.csv")
        assert df.iloc[0]["home_team"] == "EARLY"
        assert df.iloc[1]["home_team"] == "LATE"

    def test_multi_league_coverage(self, source_dirs: list[Path], output_dir: Path):
        base = source_dirs[0]
        (base / "BL1").mkdir()
        
        _write_csv(base / "PL" / "E0.csv", [_make_raw_row(div="E0")])
        _write_csv(base / "SA" / "I1.csv", [_make_raw_row(div="I1", home="Juve")])
        _write_csv(base / "BL1" / "D1.csv", [_make_raw_row(div="D1", home="Bayern")])

        builder = CanonicalDatasetBuilder(source_dirs=source_dirs, output_dir=output_dir)
        builder.build()

        with open(output_dir / "matches_coverage.json") as f:
            cov = json.load(f)

        assert cov["total_leagues"] == 3
        assert set(cov["leagues"].keys()) == {"PL", "SA", "BL1"}

import argparse
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts import sync_season_maps
from scripts import validate_team_names
from src.utils import naming


def _team_rows(*teams: str, season: int = 2026) -> list[dict[str, object]]:
    return [
        {
            "home_team": teams[index],
            "away_team": teams[(index + 1) % len(teams)],
            "season": season,
        }
        for index in range(len(teams))
    ]


@pytest.fixture
def sync_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    matches_dir = tmp_path / "matches"
    matches_dir.mkdir()
    maps_path = tmp_path / "season_maps.json"
    maps_path.write_text(
        json.dumps({"PL": {"2025": ["Arsenal", "Chelsea", "Tottenham"]}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(sync_season_maps, "MATCHES_DIR", matches_dir)
    monkeypatch.setattr(sync_season_maps, "SEASON_MAPS_PATH", maps_path)
    monkeypatch.setattr(
        naming,
        "LEAGUE_TEAMS",
        {"PL": {2025: {"Arsenal", "Chelsea", "Tottenham"}}},
    )
    monkeypatch.setattr(naming, "LEAGUE_ALIASES", {"PL": {}})
    monkeypatch.setattr(naming, "_normalize_warned", set())
    return matches_dir, maps_path


def test_sync_uses_upcoming_as_bootstrap_and_ignores_other_filenames(sync_state) -> None:
    matches_dir, maps_path = sync_state
    pd.DataFrame(_team_rows("Arsenal", "Chelsea")).to_csv(
        matches_dir / "PL_upcoming.csv", index=False
    )
    pd.DataFrame(_team_rows("NOT A CLUB", "ALSO NOT A CLUB")).to_csv(
        matches_dir / "PL_2026_backup.csv", index=False
    )

    assert sync_season_maps.main() == 0

    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    assert maps["PL"]["2026"] == ["Arsenal", "Chelsea"]


def test_sync_finished_file_supersedes_upcoming_without_merging(sync_state) -> None:
    matches_dir, maps_path = sync_state
    pd.DataFrame(_team_rows("Arsenal", "Tottenham")).to_csv(
        matches_dir / "PL_upcoming.csv", index=False
    )
    pd.DataFrame(_team_rows("Arsenal", "Chelsea")).to_csv(
        matches_dir / "PL_2026.csv", index=False
    )

    assert sync_season_maps.main() == 0

    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    assert maps["PL"]["2026"] == ["Arsenal", "Chelsea"]
    assert "Tottenham" not in maps["PL"]["2026"]


def test_sync_does_not_copy_the_finished_row_threshold() -> None:
    source = Path(sync_season_maps.__file__).read_text(encoding="utf-8")
    assert "MIN_FINISHED_ROWS" not in source
    assert "len(df) < 5" not in source


def test_validate_infers_upcoming_season_from_dataframe() -> None:
    df = pd.DataFrame(_team_rows("Arsenal", "Chelsea", season=2026))
    assert validate_team_names._infer_season(Path("PL_upcoming.csv"), df) == 2026


def test_validate_still_fails_for_genuinely_unresolvable_team(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    matches_dir = tmp_path / "matches"
    matches_dir.mkdir()
    pd.DataFrame(_team_rows("ZXQ NEVER UNITED", "Arsenal")).to_csv(
        matches_dir / "PL_upcoming.csv", index=False
    )

    monkeypatch.setattr(validate_team_names, "MATCHES_DIR", matches_dir)
    monkeypatch.setattr(
        validate_team_names,
        "_parse_args",
        lambda: argparse.Namespace(league=None),
    )
    monkeypatch.setattr(naming, "LEAGUE_TEAMS", {"PL": {2025: {"Arsenal"}}})
    monkeypatch.setattr(naming, "LEAGUE_ALIASES", {"PL": {}})
    monkeypatch.setattr(naming, "_normalize_warned", set())

    assert validate_team_names.main() == 1


@pytest.fixture
def fetch_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "test-key")
    return importlib.import_module("scripts.fetch_fresh_data")


def _fetch_frame(finished_count: int, upcoming_count: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(finished_count + upcoming_count):
        finished = index < finished_count
        rows.append(
            {
                "date": "2026-08-21",
                "home_team": f"HOME {index}",
                "away_team": f"AWAY {index}",
                "home_score": 1 if finished else None,
                "away_score": 0 if finished else None,
                "result": "H" if finished else "",
                "season": "2026",
                "league": "PL",
                "status": "FINISHED" if finished else "SCHEDULED",
                "source": "test",
                "match_id": f"match-{index}",
            }
        )
    return pd.DataFrame(rows)


def _configure_fetch(
    module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    frame: pd.DataFrame,
) -> list[str]:
    post_fetch_calls: list[str] = []
    monkeypatch.setattr(module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(module, "LEAGUES", {"PL": "PL"})
    monkeypatch.setattr(module, "FETCH_RETRIES", 2)
    monkeypatch.setattr(module, "FETCH_RETRY_DELAY", 0)
    monkeypatch.setattr(module, "current_season_year", lambda: 2026)
    monkeypatch.setattr(module, "fetch_matches", lambda _code, _season: [])
    monkeypatch.setattr(module, "matches_to_df", lambda _matches, _league, _season: frame.copy())
    monkeypatch.setattr(module, "run_post_fetch_script", post_fetch_calls.append)
    return post_fetch_calls


def test_zero_finished_writes_upcoming_preserves_finished_and_succeeds(
    fetch_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished_target = tmp_path / "PL_2026.csv"
    original_finished = b"last-known-good\n"
    finished_target.write_bytes(original_finished)
    calls = _configure_fetch(fetch_module, monkeypatch, tmp_path, _fetch_frame(0))

    assert fetch_module.main() == 0

    assert finished_target.read_bytes() == original_finished
    assert len(pd.read_csv(tmp_path / "PL_upcoming.csv")) == 2
    assert calls == ["sync_season_maps.py", "validate_team_names.py"]


def test_one_to_four_finished_retries_preserves_finished_runs_posts_and_fails(
    fetch_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    finished_target = tmp_path / "PL_2026.csv"
    original_finished = b"last-known-good\n"
    finished_target.write_bytes(original_finished)
    frame = _fetch_frame(3)
    fetch_calls = 0

    calls = _configure_fetch(fetch_module, monkeypatch, tmp_path, frame)

    def fetch_matches(_code, _season):
        nonlocal fetch_calls
        fetch_calls += 1
        return []

    monkeypatch.setattr(fetch_module, "fetch_matches", fetch_matches)

    assert fetch_module.main() == 1

    assert fetch_calls == 2
    assert finished_target.read_bytes() == original_finished
    assert len(pd.read_csv(tmp_path / "PL_upcoming.csv")) == 2
    assert calls == ["sync_season_maps.py", "validate_team_names.py"]


def test_sync_completion_precedes_validate_start(
    fetch_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_fetch(fetch_module, monkeypatch, tmp_path, _fetch_frame(0))
    sync_completed = False
    calls: list[str] = []

    def run_post_fetch_script(script_name: str) -> None:
        nonlocal sync_completed
        calls.append(script_name)
        if script_name == "sync_season_maps.py":
            sync_completed = True
        else:
            assert sync_completed

    monkeypatch.setattr(fetch_module, "run_post_fetch_script", run_post_fetch_script)

    assert fetch_module.main() == 0
    assert calls == ["sync_season_maps.py", "validate_team_names.py"]


@pytest.mark.parametrize("sync_error", [SystemExit(7), RuntimeError("sync crashed")])
def test_sync_failure_or_exception_blocks_validate(
    fetch_module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sync_error: BaseException,
) -> None:
    _configure_fetch(fetch_module, monkeypatch, tmp_path, _fetch_frame(0))
    calls: list[str] = []

    def run_post_fetch_script(script_name: str) -> None:
        calls.append(script_name)
        if script_name == "sync_season_maps.py":
            raise sync_error

    monkeypatch.setattr(fetch_module, "run_post_fetch_script", run_post_fetch_script)

    with pytest.raises(type(sync_error)) as exc_info:
        fetch_module.main()

    assert exc_info.value is sync_error
    assert calls == ["sync_season_maps.py"]


def test_upcoming_only_2026_bootstraps_then_validates_in_fresh_processes(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    scripts_dir = runtime_root / "scripts"
    utils_dir = runtime_root / "src" / "utils"
    matches_dir = runtime_root / "data" / "processed" / "matches"
    scripts_dir.mkdir(parents=True)
    utils_dir.mkdir(parents=True)
    matches_dir.mkdir(parents=True)

    shutil.copy2(sync_season_maps.__file__, scripts_dir / "sync_season_maps.py")
    shutil.copy2(validate_team_names.__file__, scripts_dir / "validate_team_names.py")
    shutil.copy2(naming.__file__, utils_dir / "naming.py")
    (runtime_root / "data" / "processed" / "season_maps.json").write_text(
        json.dumps({"PL": {"2025": ["Arsenal", "Chelsea"]}}),
        encoding="utf-8",
    )
    pd.DataFrame(_team_rows("Arsenal", "Chelsea")).to_csv(
        matches_dir / "PL_upcoming.csv", index=False
    )

    sync_result = subprocess.run(
        [sys.executable, str(scripts_dir / "sync_season_maps.py")],
        cwd=runtime_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert sync_result.returncode == 0, sync_result.stderr

    validate_result = subprocess.run(
        [sys.executable, str(scripts_dir / "validate_team_names.py")],
        cwd=runtime_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert validate_result.returncode == 0, validate_result.stdout + validate_result.stderr

    maps = json.loads(
        (runtime_root / "data" / "processed" / "season_maps.json").read_text(encoding="utf-8")
    )
    assert maps["PL"]["2026"] == ["Arsenal", "Chelsea"]

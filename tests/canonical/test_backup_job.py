from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess

import pytest

from scripts import run_canonical_backup as backup
from src.canonical import heartbeat
from src.canonical.store import Store


URL = "https://hc-ping.com/00000000-0000-4000-8000-000000000009"
NOW = datetime(2026, 11, 24, 16, 30, tzinfo=timezone.utc)
NAME = "canonical-db-20261124T163000000000Z.sqlite3"


@pytest.fixture(autouse=True)
def isolate_backup_side_effects(tmp_path, monkeypatch):
    monkeypatch.delenv(backup.HEARTBEAT_ENV, raising=False)
    monkeypatch.setattr(backup, "LOG", tmp_path / "logs" / "backup.jsonl")


@pytest.fixture
def job(tmp_path, monkeypatch):
    database = tmp_path / "football.sqlite3"
    with Store(database):
        pass
    destination = tmp_path / "drive"
    destination.mkdir()
    staging = tmp_path / "staging"
    monkeypatch.setenv(backup.HEARTBEAT_ENV, URL)
    monkeypatch.setattr(backup, "DATABASE", database)
    monkeypatch.setattr(backup, "DESTINATION", destination)
    calls = []

    def transport(endpoint, body, timeout):
        calls.append((endpoint.geturl(), json.loads(body) if body else None, timeout))
        return 200

    run_backup = backup.run_backup

    def run(*, source=database, target=destination, now=NOW):
        return run_backup(source, target, staging, transport=transport, now=now)

    return database, destination, calls, run


def completed(destination, timestamp):
    old = destination / f"canonical-db-{timestamp:%Y%m%dT%H%M%S%fZ}.sqlite3"
    old.write_bytes(b"old snapshot")
    (old.with_suffix(".sqlite3.json")).write_text(json.dumps({
        "snapshot": old.name, "logical_sha256": "0" * 64, "copy_logical_sha256": "0" * 64,
        "local_integrity": "ok", "copy_integrity": "ok",
    }), encoding="utf-8")
    return old


def november_pairs(destination, count):
    start = NOW.replace(day=2)
    return [completed(destination, start + timedelta(hours=hour)) for hour in range(count)]


def assert_pair(path, *, kept):
    assert path.exists() is kept
    assert path.with_suffix(".sqlite3.json").exists() is kept


def test_snapshot_copy_and_restore_use_real_sqlite(job, tmp_path):
    _, destination, calls, run = job
    result = run()
    assert result["state"] == "BACKUP_VERIFIED"
    assert result["heartbeat"] == "SENT"
    assert result["snapshot"] == NAME
    saved = destination / NAME
    assert backup.verified_sidecar(saved) == backup.inspect_snapshot(saved) == result["logical_sha256"]
    restored = tmp_path / "restored.sqlite3"
    assert backup.restore_check(saved, restored)["logical_sha256"] == result["logical_sha256"]
    assert calls == [(URL, None, 10)]


@pytest.mark.parametrize("alternate", ["database", "destination"])
def test_scratch_paths_skip_production_heartbeat(job, tmp_path, monkeypatch, alternate):
    database, destination, calls, run = job
    if alternate == "database":
        database = tmp_path / "scratch.sqlite3"
        with Store(database):
            pass
    else:
        destination = tmp_path / "scratch-destination"
        destination.mkdir()
    monkeypatch.setattr(backup, "heartbeat_endpoint", lambda: pytest.fail("read production heartbeat"))
    result = run(source=database, target=destination)
    assert result["state"] == "BACKUP_VERIFIED"
    assert result["heartbeat"] == "DISABLED"
    assert not calls


def test_prune_only_completed_pairs_matching_the_job_pattern(job):
    _, destination, calls, run = job
    old = november_pairs(destination, 33)
    invalid = completed(destination, NOW.replace(day=1))
    invalid_sidecar = invalid.with_suffix(".sqlite3.json")
    invalid_record = json.loads(invalid_sidecar.read_text(encoding="utf-8"))
    invalid_record["copy_logical_sha256"] = "1" * 64
    invalid_sidecar.write_text(json.dumps(invalid_record), encoding="utf-8")
    other_files = [destination / name for name in (
        "football.sqlite3", "canonical-db-personal.sqlite3", "canonical-db-personal.sqlite3.json", "notes.json",
        "canonical-db-20261101T163000000001Z.sqlite3", "canonical-db-20261101T163000000002Z.sqlite3.json",
    )]
    for path in other_files:
        path.write_bytes(b"keep this")
    result = run()
    assert result["pruned"] == 4
    for path in old[:4]:
        assert_pair(path, kept=False)
    for path in old[4:] + [invalid]:
        assert_pair(path, kept=True)
    assert json.loads(invalid_sidecar.read_text(encoding="utf-8")) == invalid_record
    assert all(p.read_bytes() == b"keep this" for p in other_files)
    assert calls == [(URL, None, 10)]


def test_prune_preserves_current_snapshot_after_backward_clock_jump(job):
    _, destination, _, run = job
    old = november_pairs(destination, 33)
    result = run(now=NOW.replace(month=10))
    current = destination / result["snapshot"]
    assert current.name < old[0].name
    assert_pair(current, kept=True)
    assert result["pruned"] == 4
    assert [path for path in old if path.exists()] == old[4:]
    assert len(list(destination.glob("*.sqlite3"))) == 30


def test_monthly_keepers_use_completed_utc_months_and_filename_timestamps(job):
    _, destination, _, run = job
    november_pairs(destination, 29)
    october_end = NOW.replace(month=11, day=1, hour=0, minute=0) - timedelta(microseconds=1)
    october_latest = completed(destination, october_end)
    october_earlier = completed(destination, october_end - timedelta(days=1))
    november_open = completed(destination, october_end + timedelta(microseconds=1))
    oldest_month_start = NOW.replace(year=2025, day=1, hour=0, minute=0)
    oldest_keeper = completed(destination, oldest_month_start)
    too_old = completed(destination, oldest_month_start - timedelta(microseconds=1))
    # Drive modification times favor the losing snapshot. The filename must win.
    os.utime(october_latest, (0, 0))
    os.utime(october_earlier, (NOW.timestamp(), NOW.timestamp()))

    assert run()["pruned"] == 3
    for path in (october_latest, oldest_keeper):
        assert_pair(path, kept=True)
    for path in (october_earlier, november_open, too_old):
        assert_pair(path, kept=False)


def test_empty_month_does_not_extend_the_monthly_window(job):
    _, destination, _, run = job
    november_pairs(destination, 29)
    months = [NOW.replace(year=2025, month=month) for month in (10, 11, 12)]
    months.extend(NOW.replace(month=month) for month in range(1, 11) if month != 9)
    too_old, *monthly_keepers = [completed(destination, timestamp) for timestamp in months]

    assert run()["pruned"] == 1
    assert_pair(too_old, kept=False)
    for path in monthly_keepers:
        assert_pair(path, kept=True)
    assert len(list(destination.glob("*.sqlite3"))) == 30 + 11


def test_latest_pair_in_month_survives_after_aging_out_of_daily_window(job):
    _, destination, _, run = job
    october_earlier = completed(destination, NOW.replace(month=10, day=30))
    october_latest = completed(destination, NOW.replace(month=10, day=31))
    assert run(now=NOW.replace(day=1))["pruned"] == 0
    november_pairs(destination, 29)

    assert run()["pruned"] == 2
    assert_pair(october_earlier, kept=False)
    assert_pair(october_latest, kept=True)


def test_daily_pruning_for_93_days_keeps_each_completed_months_latest_pair(tmp_path):
    start = datetime(2025, 12, 31, 16, 30, tzinfo=timezone.utc)
    daily = []
    latest_by_month = {}
    kept = set()
    for day in range(93):
        timestamp = start + timedelta(days=day)
        current = completed(tmp_path, timestamp)
        daily.append(current)
        month = (timestamp.year, timestamp.month)
        latest_by_month[month] = current
        monthly = {path for key, path in latest_by_month.items() if key < month}
        newest = set(daily[-30:])
        expected = newest | monthly
        assert backup.prune_completed(tmp_path, current) == len((kept | {current}) - expected)
        assert set(tmp_path.iterdir()) == expected | {path.with_suffix(".sqlite3.json") for path in expected}
        kept = expected

    assert len(monthly) == 4
    assert len(monthly - newest) == 3
    assert len(monthly & newest) == 1


def test_first_run_after_upgrade_keeps_all_fourteen_existing_pairs(job):
    _, destination, _, run = job
    previous = november_pairs(destination, 14)

    result = run()
    assert result["pruned"] == 0
    for path in previous + [destination / result["snapshot"]]:
        assert_pair(path, kept=True)


@pytest.mark.parametrize("mismatch", [False, True], ids=["corrupt-copy", "logical-mismatch"])
def test_failed_copy_verification_does_not_prune(job, monkeypatch, mismatch):
    _, destination, calls, run = job
    old = november_pairs(destination, 33)
    copy = shutil.copyfileobj

    def alter(reader, writer):
        if mismatch:
            copy(reader, writer)
            writer.flush()
            with closing(sqlite3.connect(writer.name)) as db:
                db.execute("CREATE TABLE changed (value INTEGER)")
        else:
            writer.write(b"corrupt copy")

    monkeypatch.setattr(backup.shutil, "copyfileobj", alter)
    assert run() == {
        "state": "FAILED", "step": "verify_copy", "heartbeat": "SENT", "snapshot": NAME,
        "error_type": "ValueError" if mismatch else "DatabaseError",
    }
    assert all(p.exists() and p.with_suffix(".sqlite3.json").exists() for p in old)
    assert not (destination / NAME).with_suffix(".sqlite3.json").exists()
    assert calls == [(URL + "/fail", "verify_copy", 10)]


def test_missing_drive_fails_before_snapshot_or_prune(job, monkeypatch):
    _, destination, calls, run = job
    destination.rmdir()
    monkeypatch.setattr(backup, "take_snapshot", lambda *_: pytest.fail("snapshot attempted"))
    monkeypatch.setattr(backup, "prune_completed", lambda *_: pytest.fail("prune attempted"))
    assert run() == {"state": "FAILED", "step": "destination", "heartbeat": "SENT",
                     "snapshot": None, "error_type": "FileNotFoundError"}
    assert not destination.exists()
    assert calls == [(URL + "/fail", "destination", 10)]


@pytest.mark.parametrize("releases", [True, False], ids=["one-retry-succeeds", "lock-persists"])
def test_database_lock_gets_one_retry(job, monkeypatch, releases):
    database, destination, calls, run = job
    attempts = []

    def invoke(args, **kwargs):
        target = Path(args[-1])
        attempts.append(target)
        if len(attempts) == 2 and releases:
            shutil.copyfile(database, target)
            return subprocess.CompletedProcess(args, 0, stdout='{"state":"BACKUP_CREATED"}')
        # The frozen CLI's masked exit for SQLite OperationalError.
        target.write_bytes(b"partial backup")
        return subprocess.CompletedProcess(args, 2, stdout='{"error_code":"OPERATOR_OPERATIONALERROR"}')

    monkeypatch.setattr(backup.subprocess, "run", invoke)
    result = run()
    assert len(attempts) == 2 and attempts[0] != attempts[1]
    if releases:
        assert result["state"] == "BACKUP_VERIFIED"
        assert calls == [(URL, None, 10)]
    else:
        assert result == {"state": "FAILED", "step": "snapshot", "heartbeat": "SENT",
                          "snapshot": None, "error_type": "OperationalError"}
        assert list(destination.iterdir()) == []
        assert calls == [(URL + "/fail", "snapshot", 10)]


def test_existing_target_never_overwrites(job):
    _, destination, calls, run = job
    target = destination / NAME
    target.write_bytes(b"existing target")
    assert run() == {"state": "FAILED", "step": "copy", "heartbeat": "SENT",
                     "snapshot": NAME, "error_type": "FileExistsError"}
    assert target.read_bytes() == b"existing target"
    assert calls == [(URL + "/fail", "copy", 10)]


def test_transport_failure_does_not_expose_url(job):
    database, destination, _, _ = job
    calls = []

    def offline(endpoint, body, timeout):
        calls.append(timeout)
        raise TimeoutError(endpoint.geturl())

    result = backup.run_backup(database, destination, destination.parent / "stage", transport=offline, now=NOW)
    assert result["state"] == "FAILED" and result["step"] == "heartbeat"
    assert result["heartbeat"] == "FAILED"
    assert result["error_type"] == "TimeoutError"
    assert URL not in json.dumps(result)
    assert calls == [10, 10]


@pytest.mark.parametrize("first", [None, ConnectionError, 429, 503])
def test_transient_heartbeat_retries_same_ping(first, monkeypatch):
    monkeypatch.setenv(backup.HEARTBEAT_ENV, URL)
    attempts = []

    def transport(endpoint, body, timeout):
        attempts.append((endpoint.geturl(), body, timeout))
        return first if len(attempts) == 1 else 200

    assert backup.notify(backup.heartbeat_endpoint(), "verify_copy", transport) == ("SENT", None)
    assert attempts == [(URL + "/fail", b'"verify_copy"', 10)] * 2


@pytest.mark.parametrize(("reply", "error_type"), [(404, "HTTPError"), (TypeError, "TypeError")])
def test_non_transient_heartbeat_failure_is_not_retried(reply, error_type, monkeypatch):
    monkeypatch.setenv(backup.HEARTBEAT_ENV, URL)
    attempts = []

    def transport(endpoint, body, timeout):
        attempts.append(timeout)
        return reply

    assert backup.notify(backup.heartbeat_endpoint(), None, transport) == ("FAILED", error_type)
    assert attempts == [10]


def test_backup_uses_the_frozen_private_transport():
    assert backup.run_backup.__kwdefaults__["transport"] is heartbeat._https_get
    assert backup.TIMEOUT == heartbeat.TIMEOUT


def test_cli_logs_one_masked_line_per_invocation(job, monkeypatch, capsys):
    _, destination, _, run = job
    monkeypatch.setattr(backup, "run_backup", lambda *args, **kwargs: run())

    def broken_registration():
        raise RuntimeError(URL)

    monkeypatch.setattr(backup, "register", broken_registration)
    assert backup.main(["run"]) == 0
    assert backup.main(["register"]) == 1
    assert backup.main(["restore-check", str(destination / NAME), str(destination / NAME)]) == 1
    stdout = capsys.readouterr().out
    logtext = backup.LOG.read_text(encoding="utf-8")
    assert URL not in stdout + logtext
    verified, register_error, restore_error = [json.loads(line) for line in stdout.splitlines()]
    assert verified["snapshot"] == NAME and verified["state"] == "BACKUP_VERIFIED" and verified["pruned"] == 0
    assert register_error == {"state": "FAILED", "step": "register", "error_type": "RuntimeError"}
    assert restore_error == {"state": "FAILED", "step": "restore-check",
                             "error_type": "FileExistsError", "snapshot": NAME}
    records = [json.loads(line) for line in logtext.splitlines()]
    assert all(datetime.fromisoformat(record.pop("timestamp_utc")).tzinfo is not None for record in records)
    assert records == [
        {"state": "BACKUP_VERIFIED", "step": "run", "error_type": None, "snapshot": NAME,
         "production": True, "heartbeat": "SENT", "pruned": 0},
        {**register_error, "snapshot": None, "production": False, "heartbeat": "DISABLED"},
        {**restore_error, "production": False, "heartbeat": "DISABLED"},
    ]


def test_scratch_success_log_is_marked_non_production(job, tmp_path, capsys):
    database, _, _, _ = job
    scratch = tmp_path / "scratch-output"
    scratch.mkdir()
    assert backup.main(["run", "--database", str(database), "--destination", str(scratch),
                        "--staging", str(tmp_path / "stage")]) == 0
    output = json.loads(capsys.readouterr().out)
    entry = json.loads(backup.LOG.read_text(encoding="utf-8"))
    assert entry["state"] == "BACKUP_VERIFIED"
    assert entry["snapshot"] == output["snapshot"]
    assert entry["production"] is False
    assert entry["heartbeat"] == output["heartbeat"] == "DISABLED"

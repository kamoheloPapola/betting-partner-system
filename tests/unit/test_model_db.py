import sqlite3
import threading
import time

import pytest

from src.ml.model_db import ModelHistoryDB
from src.ml.registry import ModelRegistry


def test_model_history_db_writes_and_reads_events(tmp_path):
    db = ModelHistoryDB(db_path=tmp_path / "model_history.db")

    db.write_event(
        model_name="poisson_home_base",
        league="PL",
        version="1.2.0",
        brier_score=0.182,
        ece=0.031,
        train_size=1200,
        event_type="train",
    )

    events = db.fetch_events(model_name="poisson_home_base", league="PL", limit=5)
    assert len(events) == 1
    assert events[0]["event_type"] == "train"
    assert events[0]["version"] == "1.2.0"
    assert events[0]["brier_score"] == 0.182


def test_model_history_db_enables_wal_and_30_second_busy_timeout(tmp_path):
    db = ModelHistoryDB(db_path=tmp_path / "model_history.db")

    with db._connect() as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        busy_timeout = connection.execute("PRAGMA busy_timeout").fetchone()[0]

    assert str(journal_mode).lower() == "wal"
    assert busy_timeout == 30_000


def test_model_history_concurrent_writes_wait_instead_of_failing_locked(tmp_path):
    db = ModelHistoryDB(db_path=tmp_path / "model_history.db")
    lock_acquired = threading.Event()
    release_lock = threading.Event()
    errors: list[BaseException] = []

    def hold_write_lock() -> None:
        try:
            with db._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO model_history (
                        model_name, league, version, timestamp, event_type
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    ("holder", "PL", "1.0.0", "2026-08-08T00:00:00+00:00", "holder"),
                )
                lock_acquired.set()
                if not release_lock.wait(timeout=5):
                    raise TimeoutError("test did not release SQLite writer lock")
                connection.commit()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def competing_write() -> None:
        try:
            db.write_event(
                model_name="contender",
                league="PL",
                version="1.0.0",
                brier_score=0.2,
                ece=0.03,
                train_size=100,
                event_type="contender",
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    holder = threading.Thread(target=hold_write_lock)
    holder.start()
    assert lock_acquired.wait(timeout=2)

    with sqlite3.connect(db.db_path, timeout=0) as no_wait_connection:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            no_wait_connection.execute(
                "INSERT INTO model_history (model_name, league, version, timestamp, event_type) "
                "VALUES ('no-wait', 'PL', '1.0.0', '2026-08-08T00:00:00+00:00', 'no-wait')"
            )

    contender = threading.Thread(target=competing_write)
    contender.start()
    time.sleep(0.1)
    assert contender.is_alive(), "configured writer should wait while the lock is held"

    release_lock.set()
    holder.join(timeout=5)
    contender.join(timeout=5)

    assert not holder.is_alive()
    assert not contender.is_alive()
    assert errors == []
    assert {event["event_type"] for event in db.fetch_events(limit=10)} == {
        "holder",
        "contender",
    }


def test_registry_logs_train_promote_and_rollback_events(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_SIGNING_KEY", "test-signing-key-with-at-least-32-bytes")
    manifest_file = tmp_path / "models" / "manifest.json"
    backup_file = tmp_path / "models" / "manifest.json.bak"
    db_file = tmp_path / "models" / "model_history.db"
    model_file = tmp_path / "models" / "dummy.pkl"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b"model")

    monkeypatch.setattr(ModelRegistry, "MANIFEST_FILE", manifest_file)
    monkeypatch.setattr(ModelRegistry, "BACKUP_FILE", backup_file)
    monkeypatch.setattr(ModelHistoryDB, "DB_FILE", db_file)

    ModelRegistry._instance = None
    ModelRegistry._manifest_cache = None
    registry = ModelRegistry()

    base_meta = {
        "type": "poisson",
        "target": "home_score",
        "features": ["f1"],
        "params": {"alpha": 0.01},
        "filename": str(model_file),
        "train_size": 1000,
        "test_size": 200,
        "mode": "production",
        "status": "productive",
        "league": "PL",
        "metrics": {"brier_score": 0.21, "calibration_score": 0.04},
    }
    registry.register_model("poisson_home_base", "1.0.0", dict(base_meta))
    registry.register_model("poisson_home_base", "1.1.0", dict(base_meta))

    key_old = next(
        key
        for key, meta in registry.manifest.items()
        if isinstance(meta, dict)
        and meta.get("name") == "poisson_home_base"
        and meta.get("league") == "PL"
        and meta.get("version") == "1.0.0"
    )
    key_new = next(
        key
        for key, meta in registry.manifest.items()
        if isinstance(meta, dict)
        and meta.get("name") == "poisson_home_base"
        and meta.get("league") == "PL"
        and meta.get("version") == "1.1.0"
    )
    registry.manifest[key_old]["registered_at"] = "2026-03-01T00:00:00+00:00"
    registry.manifest[key_new]["registered_at"] = "2026-03-20T00:00:00+00:00"
    registry._save_manifest()

    registry.set_active_model("poisson_home_base", key_new, league="PL")
    rollback_key = registry.rollback_active_model("poisson_home_base", league="PL")

    assert rollback_key == key_old
    assert registry.manifest["active_models"]["poisson_home_base_PL"] == key_old

    events = ModelHistoryDB(db_path=db_file).fetch_events(
        model_name="poisson_home_base",
        league="PL",
        limit=20,
    )
    event_types = [event["event_type"] for event in events]
    assert event_types.count("train") == 2
    assert "promote" in event_types
    assert "rollback" in event_types

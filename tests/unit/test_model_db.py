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


def test_registry_logs_train_promote_and_rollback_events(tmp_path, monkeypatch):
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

import json

from scripts.cleanup_old_models import cleanup_models
from src.ml.model_db import ModelHistoryDB


def _write_model_file(model_root, rel_path, size):
    path = model_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def _write_manifest(model_root, payload):
    manifest = model_root / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_cleanup_models_keeps_recent_versions_and_protected_scopes(tmp_path):
    model_root = tmp_path / "src" / "ml" / "models"

    files = {
        "poisson/PL/poisson_home_base_v1.0.0.pkl": 10,
        "poisson/PL/poisson_home_base_v1.1.0.pkl": 11,
        "poisson/PL/poisson_home_base_v1.2.0.pkl": 12,
        "poisson/PL/poisson_home_base_v1.3.0.pkl": 13,
        "poisson/PL/poisson_home_base_v1.4.0.pkl": 14,
        "nb/SA/nb_home_corners_base_v1.0.0.pkl": 15,
        "nb/SA/nb_home_corners_base_v1.1.0.pkl": 16,
        "calibrators/goals_o25_v1.pkl": 17,
    }
    for rel_path, size in files.items():
        _write_model_file(model_root, rel_path, size)

    manifest = {
        "poisson_home_base_v1.0.0_PL": {
            "name": "poisson_home_base",
            "league": "PL",
            "version": "1.0.0",
            "status": "productive",
            "filename": "poisson/PL/poisson_home_base_v1.0.0.pkl",
            "registered_at": "2026-01-01T00:00:00+00:00",
        },
        "poisson_home_base_v1.1.0_PL": {
            "name": "poisson_home_base",
            "league": "PL",
            "version": "1.1.0",
            "status": "provisional",
            "filename": "poisson/PL/poisson_home_base_v1.1.0.pkl",
            "registered_at": "2026-01-02T00:00:00+00:00",
        },
        "poisson_home_base_v1.2.0_PL": {
            "name": "poisson_home_base",
            "league": "PL",
            "version": "1.2.0",
            "status": "provisional",
            "filename": "poisson/PL/poisson_home_base_v1.2.0.pkl",
            "registered_at": "2026-01-03T00:00:00+00:00",
        },
        "poisson_home_base_v1.3.0_PL": {
            "name": "poisson_home_base",
            "league": "PL",
            "version": "1.3.0",
            "status": "provisional",
            "filename": "poisson/PL/poisson_home_base_v1.3.0.pkl",
            "registered_at": "2026-01-04T00:00:00+00:00",
        },
        "poisson_home_base_v1.4.0_PL": {
            "name": "poisson_home_base",
            "league": "PL",
            "version": "1.4.0",
            "status": "provisional",
            "filename": "poisson/PL/poisson_home_base_v1.4.0.pkl",
            "registered_at": "2026-01-05T00:00:00+00:00",
        },
        "nb_home_corners_base_v1.0.0_SA": {
            "name": "nb_home_corners_base",
            "league": "SA",
            "version": "1.0.0",
            "status": "provisional",
            "filename": "nb/SA/nb_home_corners_base_v1.0.0.pkl",
            "registered_at": "2026-01-01T00:00:00+00:00",
        },
        "nb_home_corners_base_v1.1.0_SA": {
            "name": "nb_home_corners_base",
            "league": "SA",
            "version": "1.1.0",
            "status": "provisional",
            "filename": "nb/SA/nb_home_corners_base_v1.1.0.pkl",
            "registered_at": "2026-01-02T00:00:00+00:00",
        },
        "active_models": {
            "poisson_home_base_PL": "poisson_home_base_v1.4.0_PL",
        },
        "shadow_models": {
            "nb_home_corners_base_SA": ["nb_home_corners_base_v1.0.0_SA"],
        },
    }
    _write_manifest(model_root, manifest)

    result = cleanup_models(model_root=model_root, keep_versions=1, confirm=False)

    assert result.protected_files == [
        "nb/SA/nb_home_corners_base_v1.0.0.pkl",
        "poisson/PL/poisson_home_base_v1.0.0.pkl",
        "poisson/PL/poisson_home_base_v1.4.0.pkl",
    ]
    assert result.retained_recent_files == [
        "nb/SA/nb_home_corners_base_v1.1.0.pkl",
        "poisson/PL/poisson_home_base_v1.4.0.pkl",
    ]
    assert result.referenced_missing_files == []
    assert result.deletion_candidates == [
        "calibrators/goals_o25_v1.pkl",
        "poisson/PL/poisson_home_base_v1.1.0.pkl",
        "poisson/PL/poisson_home_base_v1.2.0.pkl",
        "poisson/PL/poisson_home_base_v1.3.0.pkl",
    ]
    assert result.total_candidate_bytes == 53


def test_cleanup_models_confirm_deletes_and_logs_cleanup_events(tmp_path):
    model_root = tmp_path / "src" / "ml" / "models"
    db_path = tmp_path / "data" / "models" / "model_history.db"

    files = {
        "poisson/PL/poisson_home_base_v1.0.0.pkl": 10,
        "poisson/PL/poisson_home_base_v1.1.0.pkl": 11,
        "poisson/PL/poisson_home_base_v1.2.0.pkl": 12,
        "calibrators/btts_yes_v1.pkl": 13,
    }
    for rel_path, size in files.items():
        _write_model_file(model_root, rel_path, size)

    manifest = {
        "poisson_home_base_v1.0.0_PL": {
            "name": "poisson_home_base",
            "league": "PL",
            "version": "1.0.0",
            "status": "productive",
            "filename": "poisson/PL/poisson_home_base_v1.0.0.pkl",
            "registered_at": "2026-01-01T00:00:00+00:00",
        },
        "poisson_home_base_v1.1.0_PL": {
            "name": "poisson_home_base",
            "league": "PL",
            "version": "1.1.0",
            "status": "provisional",
            "filename": "poisson/PL/poisson_home_base_v1.1.0.pkl",
            "registered_at": "2026-01-02T00:00:00+00:00",
        },
        "poisson_home_base_v1.2.0_PL": {
            "name": "poisson_home_base",
            "league": "PL",
            "version": "1.2.0",
            "status": "provisional",
            "filename": "poisson/PL/poisson_home_base_v1.2.0.pkl",
            "registered_at": "2026-01-03T00:00:00+00:00",
        },
        "active_models": {},
        "shadow_models": {},
    }
    _write_manifest(model_root, manifest)

    result = cleanup_models(
        model_root=model_root,
        keep_versions=1,
        confirm=True,
        history_db_path=db_path,
    )

    assert sorted(result.deleted_files) == [
        "calibrators/btts_yes_v1.pkl",
        "poisson/PL/poisson_home_base_v1.1.0.pkl",
    ]
    assert result.deleted_bytes == 24
    assert not (model_root / "calibrators/btts_yes_v1.pkl").exists()
    assert not (model_root / "poisson/PL/poisson_home_base_v1.1.0.pkl").exists()
    assert (model_root / "poisson/PL/poisson_home_base_v1.0.0.pkl").exists()
    assert (model_root / "poisson/PL/poisson_home_base_v1.2.0.pkl").exists()

    events = ModelHistoryDB(db_path=db_path).fetch_events(limit=10)
    assert len(events) == 2
    assert {event["event_type"] for event in events} == {"cleanup"}
    assert {event["model_name"] for event in events} == {"btts_yes", "poisson_home_base"}
    assert {event["version"] for event in events} == {"1", "1.1.0"}

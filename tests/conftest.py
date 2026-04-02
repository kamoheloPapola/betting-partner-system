import shutil
import uuid
from pathlib import Path

import pytest


TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "test-fixtures"


@pytest.fixture
def tmp_path():
    path = TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def repo_state_paths(tmp_path):
    """Standard sandbox paths for tests that would otherwise write into repo state."""
    root = tmp_path / "repo-state"
    data_dir = root / "data"
    models_dir = root / "models"
    return {
        "root": root,
        "data_dir": data_dir,
        "raw_data_dir": data_dir / "raw",
        "processed_data_dir": data_dir / "processed",
        "models_dir": models_dir,
        "history_db_file": data_dir / "models" / "model_history.db",
        "drift_status_file": data_dir / "drift" / "rolling_90d_status.json",
        "drift_confidence_file": data_dir / "drift" / "confidence_drift_state.json",
        "drift_alerts_file": data_dir / "monitoring" / "drift_alerts.csv",
        "drift_baseline_file": data_dir / "models" / "drift_baselines.json",
        "model_state_file": data_dir / ".model_state",
        "model_state_audit_file": data_dir / ".model_state_audit.jsonl",
        "manifest_file": models_dir / "manifest.json",
        "manifest_backup_file": models_dir / "manifest.json.bak",
    }


@pytest.fixture
def isolated_repo_state(repo_state_paths, monkeypatch):
    """
    Project standard for tests that write to data/, models/, or config-backed state.

    Tests that exercise persistent side effects should depend on this fixture and
    then add any module-specific path overrides for already-imported constants.
    """
    import src.config as config
    import src.config.model_state as model_state
    import src.ml.model_db as model_db_module
    import src.ml.registry as registry_module
    import src.ml.trainer as trainer_module
    from src.ml.model_db import ModelHistoryDB
    from src.ml.registry import ModelRegistry
    from src.strategies.drift_guard import DriftGuardrail

    monkeypatch.setattr(config, "DATA_DIR", repo_state_paths["data_dir"])
    monkeypatch.setattr(config, "RAW_DATA_DIR", repo_state_paths["raw_data_dir"])
    monkeypatch.setattr(config, "PROCESSED_DATA_DIR", repo_state_paths["processed_data_dir"])
    monkeypatch.setattr(config, "MODELS_DIR", repo_state_paths["models_dir"])

    monkeypatch.setattr(model_state, "_STATE_FILE", repo_state_paths["model_state_file"])
    monkeypatch.setattr(registry_module, "MODELS_DIR", repo_state_paths["models_dir"])
    monkeypatch.setattr(trainer_module, "MODELS_DIR", repo_state_paths["models_dir"])
    monkeypatch.setattr(trainer_module, "DRIFT_BASELINE_FILE", repo_state_paths["drift_baseline_file"])
    monkeypatch.setattr(model_db_module, "DATA_DIR", repo_state_paths["data_dir"])
    monkeypatch.setattr(ModelHistoryDB, "DB_FILE", repo_state_paths["history_db_file"])

    monkeypatch.setattr(ModelRegistry, "MANIFEST_FILE", repo_state_paths["manifest_file"])
    monkeypatch.setattr(
        ModelRegistry,
        "BACKUP_FILE",
        repo_state_paths["manifest_backup_file"],
    )
    monkeypatch.setattr(
        DriftGuardrail,
        "STATUS_FILE",
        repo_state_paths["drift_status_file"],
    )
    monkeypatch.setattr(
        DriftGuardrail,
        "BASELINE_FILE",
        repo_state_paths["drift_baseline_file"],
    )

    return repo_state_paths


@pytest.fixture(autouse=True)
def isolate_drift_guard_files(repo_state_paths, monkeypatch):
    from src.ml.model_db import ModelHistoryDB
    from src.monitoring.drift_orchestrator import DriftOrchestrator
    from src.strategies.drift_guard import DriftGuardrail

    monkeypatch.setattr(ModelHistoryDB, "DB_FILE", repo_state_paths["history_db_file"])
    monkeypatch.setattr(DriftOrchestrator, "DEFAULT_STATUS_FILE", repo_state_paths["drift_status_file"])
    monkeypatch.setattr(
        DriftOrchestrator,
        "DEFAULT_CONFIDENCE_STATE_FILE",
        repo_state_paths["drift_confidence_file"],
    )
    monkeypatch.setattr(DriftOrchestrator, "DEFAULT_BASELINE_FILE", repo_state_paths["drift_baseline_file"])
    monkeypatch.setattr(DriftOrchestrator, "DEFAULT_ALERTS_FILE", repo_state_paths["drift_alerts_file"])
    monkeypatch.setattr(
        DriftGuardrail,
        "STATUS_FILE",
        repo_state_paths["drift_status_file"],
    )
    monkeypatch.setattr(
        DriftGuardrail,
        "BASELINE_FILE",
        repo_state_paths["drift_baseline_file"],
    )


@pytest.fixture(autouse=True)
def isolate_database_url(monkeypatch):
    from src.db import connection as connection_module

    connection_module.get_engine.cache_clear()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    try:
        yield
    finally:
        connection_module.get_engine.cache_clear()

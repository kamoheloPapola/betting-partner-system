import copy
import json
import pickle
import runpy
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

import src.config.model_state as model_state
import src.ml.registry as registry_module
from src.config.model_state import (
    InvalidModelStateError,
    ModelStateLockedError,
    require_unlocked,
)
from src.ml.registry import ModelRegistry
from src.ml.trainer import ModelTrainer
from src.models.train_probability_models import ProbabilityModelTrainer
from src.scheduler import nightly


LOCKED_STATE = "LOCKED_v14.0"
SIGNING_KEY = "test-artifact-signing-key-at-least-32-bytes"


def _write_state(path: Path, state: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state, encoding="utf-8")


def _fresh_registry(monkeypatch, manifest):
    ModelRegistry._instance = None
    ModelRegistry._manifest_cache = None
    registry = ModelRegistry()
    registry.manifest = copy.deepcopy(manifest)
    monkeypatch.setattr(registry, "_write_lifecycle_event", lambda *args, **kwargs: None)
    registry._save_manifest_file()
    return registry


@pytest.fixture(autouse=True)
def _reset_registry_cache():
    ModelRegistry._instance = None
    ModelRegistry._manifest_cache = None
    yield
    ModelRegistry._instance = None
    ModelRegistry._manifest_cache = None


def test_missing_model_state_remains_unlocked(isolated_repo_state):
    assert not isolated_repo_state["model_state_file"].exists()
    require_unlocked("test mutation")
    assert model_state.get_model_state() == "UNLOCKED"


@pytest.mark.parametrize("invalid_state", ["", "LOCKED", "LOCKED_vnext", "garbage"])
def test_invalid_model_state_fails_closed(isolated_repo_state, invalid_state):
    _write_state(isolated_repo_state["model_state_file"], invalid_state)

    with pytest.raises(InvalidModelStateError):
        require_unlocked("test mutation")


def test_unreadable_model_state_fails_closed(isolated_repo_state, monkeypatch):
    state_file = isolated_repo_state["model_state_file"]
    _write_state(state_file, "UNLOCKED")
    original_read_text = Path.read_text

    def deny_state_read(path, *args, **kwargs):
        if path == state_file:
            raise PermissionError("denied for test")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny_state_read)

    with pytest.raises(InvalidModelStateError):
        require_unlocked("test mutation")


def test_non_utf8_model_state_fails_closed(isolated_repo_state):
    state_file = isolated_repo_state["model_state_file"]
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_bytes(b"\xff\xfe\xfd")

    with pytest.raises(InvalidModelStateError):
        require_unlocked("test mutation")


def _registry_manifest():
    return {
        "demo_v1.0.0": {
            "name": "demo",
            "version": "1.0.0",
            "status": "productive",
            "league": "Global",
            "registered_at": "2026-01-01T00:00:00+00:00",
            "metrics": {"brier": 0.3},
        },
        "demo_v2.0.0": {
            "name": "demo",
            "version": "2.0.0",
            "status": "productive",
            "league": "Global",
            "registered_at": "2026-02-01T00:00:00+00:00",
            "metrics": {"brier": 0.2},
        },
        "active_models": {"demo": "demo_v2.0.0"},
        "shadow_models": {},
    }


def _invoke_registry_mutation(operation, registry, models_dir):
    if operation == "register":
        artifact = models_dir / "new.pkl"
        metadata = {"filename": artifact.name, "status": "productive"}
        registry.register_model("new", "1.0.0", metadata)
    elif operation == "set_active":
        registry.set_active_model("demo", "demo_v1.0.0")
    elif operation == "rollback":
        registry.rollback_active_model("demo")
    elif operation == "set_shadow":
        registry.set_shadow_model("demo", "demo_v1.0.0")
    elif operation == "update_metrics":
        registry.update_model_metrics("demo", "1.0.0", {"brier": 0.1})
    else:  # pragma: no cover - protects the test table itself
        raise AssertionError(operation)


def _prepare_registry_mutation(operation, models_dir):
    if operation == "register":
        artifact = models_dir / "new.pkl"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(pickle.dumps({"model": "new"}))


def _directory_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


REGISTRY_MUTATIONS = [
    "register",
    "set_active",
    "rollback",
    "set_shadow",
    "update_metrics",
]


@pytest.mark.parametrize("operation", REGISTRY_MUTATIONS)
def test_registry_mutation_is_effect_free_when_locked(
    isolated_repo_state, monkeypatch, operation
):
    monkeypatch.setenv("ARTIFACT_SIGNING_KEY", SIGNING_KEY)
    registry = _fresh_registry(monkeypatch, _registry_manifest())
    _prepare_registry_mutation(operation, isolated_repo_state["models_dir"])
    _write_state(isolated_repo_state["model_state_file"], LOCKED_STATE)
    before_memory = copy.deepcopy(registry.manifest)
    before_disk = _directory_bytes(isolated_repo_state["models_dir"])

    with pytest.raises(ModelStateLockedError):
        _invoke_registry_mutation(operation, registry, isolated_repo_state["models_dir"])

    assert registry.manifest == before_memory
    assert _directory_bytes(isolated_repo_state["models_dir"]) == before_disk


@pytest.mark.parametrize("operation", REGISTRY_MUTATIONS)
def test_registry_mutation_proceeds_when_unlocked(
    isolated_repo_state, monkeypatch, operation
):
    monkeypatch.setenv("ARTIFACT_SIGNING_KEY", SIGNING_KEY)
    registry = _fresh_registry(monkeypatch, _registry_manifest())
    _prepare_registry_mutation(operation, isolated_repo_state["models_dir"])
    _write_state(isolated_repo_state["model_state_file"], "UNLOCKED")
    before = copy.deepcopy(registry.manifest)

    _invoke_registry_mutation(operation, registry, isolated_repo_state["models_dir"])

    assert registry.manifest != before
    assert json.loads(isolated_repo_state["manifest_file"].read_text(encoding="utf-8")) == registry.manifest


def _training_frame(rows=18):
    return pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="D"),
            "feature": [float(index % 5) for index in range(rows)],
            "target": [index % 4 for index in range(rows)],
        }
    )


def test_model_trainer_creates_no_artifact_when_locked(isolated_repo_state, monkeypatch):
    registry = _fresh_registry(monkeypatch, {})
    _write_state(isolated_repo_state["model_state_file"], LOCKED_STATE)
    before_manifest = copy.deepcopy(registry.manifest)

    with pytest.raises(ModelStateLockedError):
        ModelTrainer(registry).train_model(
            _training_frame(),
            target_col="target",
            features=["feature"],
            model_name="locked_training",
        )

    assert registry.manifest == before_manifest
    assert list(isolated_repo_state["models_dir"].glob("*.pkl")) == []
    assert not isolated_repo_state["drift_baseline_file"].exists()


def test_model_trainer_registers_artifact_when_unlocked(isolated_repo_state, monkeypatch):
    monkeypatch.setenv("ARTIFACT_SIGNING_KEY", SIGNING_KEY)
    registry = _fresh_registry(monkeypatch, {})
    _write_state(isolated_repo_state["model_state_file"], "UNLOCKED")

    _, metadata = ModelTrainer(registry).train_model(
        _training_frame(),
        target_col="target",
        features=["feature"],
        model_name="unlocked_training",
    )

    assert (isolated_repo_state["models_dir"] / metadata["filename"]).exists()
    assert any(key.startswith("unlocked_training_v") for key in registry.manifest)


def test_probability_trainer_stops_before_loading_or_writing_when_locked(
    isolated_repo_state, monkeypatch
):
    trainer = ProbabilityModelTrainer(
        features_path=isolated_repo_state["data_dir"] / "features.csv",
        models_dir=isolated_repo_state["models_dir"] / "probability",
    )
    _write_state(isolated_repo_state["model_state_file"], LOCKED_STATE)
    monkeypatch.setattr(
        trainer,
        "load_and_engineer_data",
        lambda: pytest.fail("training data must not be loaded while locked"),
    )

    with pytest.raises(ModelStateLockedError):
        trainer.run(tracked_leagues=["PL"])

    assert list(trainer.models_dir.iterdir()) == []


def test_probability_trainer_writes_reports_when_unlocked(isolated_repo_state, monkeypatch):
    models_dir = isolated_repo_state["models_dir"] / "probability"
    trainer = ProbabilityModelTrainer(
        features_path=isolated_repo_state["data_dir"] / "features.csv",
        models_dir=models_dir,
    )
    _write_state(isolated_repo_state["model_state_file"], "UNLOCKED")
    monkeypatch.setattr(
        trainer,
        "load_and_engineer_data",
        lambda: pd.DataFrame({"league": ["PL"], "outcome": ["H"]}),
    )
    monkeypatch.setattr(
        trainer,
        "_train_league",
        lambda league, frame, full_retrain=False: {
            "metrics": {"trained": True},
            "metadata": {"league": league},
        },
    )

    trainer.run(tracked_leagues=["PL"])

    assert (models_dir / "training_metadata.json").exists()
    assert (models_dir / "training_report.json").exists()


def test_nightly_stale_retrain_leaves_artifacts_and_manifest_unchanged_when_locked(
    isolated_repo_state, monkeypatch
):
    manifest = {
        "poisson_home_base_v1.0.0": {
            "name": "poisson_home_base",
            "version": "1.0.0",
            "status": "productive",
            "league": "Global",
            "registered_at": "2026-01-01T00:00:00+00:00",
        },
        "active_models": {},
        "shadow_models": {},
    }
    registry = _fresh_registry(monkeypatch, manifest)
    _write_state(isolated_repo_state["model_state_file"], LOCKED_STATE)
    target = nightly.StaleModelTarget(
        model_name="poisson_home_base",
        league="Global",
        manifest_key="poisson_home_base_v1.0.0",
        trained_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    before_manifest = copy.deepcopy(registry.manifest)
    before_disk = isolated_repo_state["manifest_file"].read_bytes()
    monkeypatch.setattr(nightly, "select_features", lambda *args, **kwargs: ["feature"])

    with pytest.raises(ModelStateLockedError):
        nightly.retrain_stale_models(
            registry,
            ModelTrainer(registry),
            [target],
            _training_frame(320).assign(league="PL"),
        )

    assert registry.manifest == before_manifest
    assert isolated_repo_state["manifest_file"].read_bytes() == before_disk
    assert list(isolated_repo_state["models_dir"].glob("*.pkl")) == []


def test_nightly_main_reports_distinct_blocked_exit(monkeypatch, caplog):
    monkeypatch.setattr(
        nightly,
        "run_nightly",
        lambda **kwargs: (_ for _ in ()).throw(ModelStateLockedError("locked")),
    )

    with pytest.raises(SystemExit) as exc_info:
        nightly.main([])

    assert exc_info.value.code == 3
    assert "nightly_pipeline_blocked reason=model_state" in caplog.text


def test_auto_retrain_does_not_swallow_model_state_failure(monkeypatch):
    target = nightly.AutoRetrainTarget(
        model_name="poisson_home_base",
        league="Global",
        market="home_win",
        manifest_key="old",
        version="1.0.0",
        baseline_brier=0.2,
        rolling_brier=0.3,
        sample_size=30,
    )

    class LockedTrainer:
        def train_model(self, **kwargs):
            raise ModelStateLockedError("locked")

    class Registry:
        manifest = {}

    monkeypatch.setattr(nightly, "select_features", lambda *args, **kwargs: ["feature"])
    monkeypatch.setattr(nightly, "find_auto_retrain_targets", lambda **kwargs: [target])
    monkeypatch.setattr(nightly, "_log_auto_retrain_event", lambda *args, **kwargs: None)

    with pytest.raises(ModelStateLockedError):
        nightly.auto_retrain_underperforming_models(
            registry=Registry(),
            trainer=LockedTrainer(),
            feature_df=_training_frame(320).assign(league="PL"),
            evals_df=pd.DataFrame(),
            alerter=object(),
        )


def test_manifest_migration_is_blocked_without_writing(isolated_repo_state, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "src" / "ml" / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"demo": {"filename": "demo.pkl"}}', encoding="utf-8")
    before = manifest_path.read_bytes()
    _write_state(isolated_repo_state["model_state_file"], LOCKED_STATE)
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "migrate_manifest.py"))

    with pytest.raises(ModelStateLockedError):
        namespace["migrate"]()

    assert manifest_path.read_bytes() == before


def test_manifest_migration_proceeds_when_unlocked(isolated_repo_state, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "src" / "ml" / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"demo": {"filename": "demo.pkl"}}', encoding="utf-8")
    _write_state(isolated_repo_state["model_state_file"], "UNLOCKED")
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "migrate_manifest.py"))

    namespace["migrate"]()

    assert json.loads(manifest_path.read_text(encoding="utf-8"))["demo"]["league"] == "PL"


def test_manifest_pruning_is_blocked_without_writing(isolated_repo_state, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "src" / "ml" / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"demo": {"filename": "missing.pkl"}}', encoding="utf-8")
    before = manifest_path.read_bytes()
    _write_state(isolated_repo_state["model_state_file"], LOCKED_STATE)

    with pytest.raises(ModelStateLockedError):
        runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "prune_manifest.py"))

    assert manifest_path.read_bytes() == before


def test_manifest_pruning_proceeds_when_unlocked(isolated_repo_state, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    manifest_path = tmp_path / "src" / "ml" / "models" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text('{"demo": {"filename": "missing.pkl"}}', encoding="utf-8")
    _write_state(isolated_repo_state["model_state_file"], "UNLOCKED")

    runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "prune_manifest.py"))

    pruned = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "demo" not in pruned
    assert pruned["active_models"] == {}

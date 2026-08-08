import json
import logging
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
import src.ml.registry as registry_module
from src.ml.registry import ArtifactTransferTimeoutError, ModelRegistry, PROMOTION_THRESHOLDS
from src.core.exceptions import ConfigurationError, ModelNotFoundError

def test_model_registry_production_loading(tmp_path, monkeypatch):
    """
    Verify get_production_model_for_league resolves against a real manifest.
    Does NOT mock the method under test - exercises actual registry logic.

    Covers:
    - Correct league is returned, not a sibling league with the same model name
    - Shadow / candidate entries are ignored even when their metrics are better
    - The returned dict carries the expected fields (league, version, status)
    """
    monkeypatch.setattr(registry_module, "MODELS_DIR", tmp_path)
    for filename in (
        "pl_poisson_home_base_v2.pkl",
        "pl_poisson_home_base_v3.pkl",
        "sa_poisson_home_base_v1.pkl",
    ):
        (tmp_path / filename).write_bytes(b"model")

    registry = ModelRegistry()
    original_manifest = dict(registry.manifest)
    try:
        registry.manifest = {
            "pl_v2_production": {
                "name": "poisson_home_base",
                "league": "PL",
                "version": "2.0.0",
                "status": "productive",
                "registered_at": "2026-03-01T00:00:00",
                "metrics": {"brier_score": 0.22},
                "filename": "pl_poisson_home_base_v2.pkl",
            },
            # Newer, lower brier - but shadow; must NOT be preferred over production
            "pl_v3_shadow": {
                "name": "poisson_home_base",
                "league": "PL",
                "version": "3.0.0",
                "status": "shadow",
                "registered_at": "2026-04-01T00:00:00",
                "metrics": {"brier_score": 0.18},
                "filename": "pl_poisson_home_base_v3.pkl",
            },
            # Different league - must never bleed into PL result
            "sa_v1_production": {
                "name": "poisson_home_base",
                "league": "SA",
                "version": "1.0.0",
                "status": "productive",
                "registered_at": "2026-01-01T00:00:00",
                "metrics": {"brier_score": 0.30},
                "filename": "sa_poisson_home_base_v1.pkl",
            },
        }

        result = registry.get_production_model_for_league("PL", "poisson_home_base")

        assert result is not None, "Expected a production model for PL/poisson_home_base"
        assert result["league"] == "PL", "Returned model must belong to the requested league"
        assert result["version"] == "2.0.0", (
            "Must select the productive-tagged entry (v2), not the shadow (v3) "
            "even though the shadow has better metrics"
        )
        assert result.get("status") == "productive"
    finally:
        registry.manifest = original_manifest


def test_model_registry_production_loading_returns_none_when_no_production_model(tmp_path, monkeypatch):
    """
    get_production_model_for_league must return None (not raise) when the
    manifest contains entries for the league/model but none are production-tagged.
    """
    monkeypatch.setattr(registry_module, "MODELS_DIR", tmp_path)
    (tmp_path / "pl_poisson_home_base_v1.pkl").write_bytes(b"model")

    registry = ModelRegistry()
    original_manifest = dict(registry.manifest)
    try:
        registry.manifest = {
            "pl_candidate": {
                "name": "poisson_home_base",
                "league": "PL",
                "version": "1.0.0",
                "status": "shadow",
                "registered_at": "2026-01-01T00:00:00",
                "metrics": {"brier_score": 0.28},
                "filename": "pl_poisson_home_base_v1.pkl",
            },
        }

        result = registry.get_production_model_for_league("PL", "poisson_home_base")

        assert result is None, (
            "No productive-tagged entry exists - registry must return None, not raise"
        )
    finally:
        registry.manifest = original_manifest

def test_drift_guard_thresholds(tmp_path):
    """Test that DriftGuardrail alerts on degraded models."""
    from src.strategies.drift_guard import DriftGuardrail
    
    guard = DriftGuardrail(
        status_file=tmp_path / "drift" / "status.json",
        baseline_file=tmp_path / "models" / "drift_baselines.json",
    )
    
    # Normal performance
    healthy_data = {"hit_rate": 0.80, "ece": 0.02, "mean_conf": 0.58}
    status_h = guard.check_drift(healthy_data)
    assert status_h == "OK"
    assert len(guard.alerts) == 0
    
    # Degraded performance (High Drift)
    drifted_data = {"hit_rate": 0.50, "ece": 0.15, "mean_conf": 0.72}
    status_d = guard.check_drift(drifted_data)
    assert status_d in ["STOP", "WATCH"]
    assert len(guard.alerts) > 0


def test_enhanced_xg_feature_selector_loads_json(tmp_path: Path) -> None:
    """ENHANCED_XG should load feature list from models/feature_columns.json."""
    from src.ml.training.feature_selector import select_features
    from src.ml.training.model_configs import FeatureSet
    import src.ml.training.feature_selector as fs_module

    cols = ["home_rolling_goals_scored_3", "away_rolling_goals_scored_3", "home_form_rating"]
    json_file = tmp_path / "feature_columns.json"
    json_file.write_text(json.dumps(cols), encoding="utf-8")

    with patch.object(fs_module, "ENHANCED_XG_COLUMNS_FILE", json_file):
        df = pd.DataFrame({c: [1.0, 2.0] for c in cols} | {"outcome": [0, 1]})
        result = select_features(df, "outcome", FeatureSet.ENHANCED_XG)

    assert result == cols, "Should return exactly the columns from JSON (minus target)"


def test_enhanced_xg_missing_json_raises(tmp_path: Path) -> None:
    """ENHANCED_XG should raise DataValidationError when feature_columns.json is absent."""
    from src.ml.training.feature_selector import select_features
    from src.ml.training.model_configs import FeatureSet
    from src.core.exceptions import DataValidationError
    import src.ml.training.feature_selector as fs_module

    absent = tmp_path / "nonexistent.json"
    with patch.object(fs_module, "ENHANCED_XG_COLUMNS_FILE", absent):
        df = pd.DataFrame({"home_rolling_goals_scored_3": [1.0], "outcome": [0]})
        with pytest.raises(DataValidationError):
            select_features(df, "outcome", FeatureSet.ENHANCED_XG)


def test_get_best_model_prefers_lowest_brier_score():
    registry = ModelRegistry()
    original_manifest = dict(registry.manifest)
    try:
        registry.manifest = {
            "m1": {
                "name": "poisson_home_base",
                "version": "1.0.0",
                "registered_at": "2026-01-01T00:00:00",
                "metrics": {"brier_score": 0.31},
            },
            "m2": {
                "name": "poisson_home_base",
                "version": "1.1.0",
                "registered_at": "2026-02-01T00:00:00",
                "metrics": {"brier_score": 0.22},
            },
            "m3": {
                "name": "poisson_home_base",
                "version": "1.2.0",
                "registered_at": "2026-03-01T00:00:00",
                "brier_score": 0.28,
            },
            "other": {
                "name": "poisson_away_base",
                "version": "1.0.0",
                "registered_at": "2026-03-01T00:00:00",
                "metrics": {"brier_score": 0.10},
            },
        }

        best = registry.get_best_model("poisson_home_base")
        assert best is not None
        assert best.get("version") == "1.1.0"
    finally:
        registry.manifest = original_manifest


def test_promotion_threshold_lowers_when_local_beats_global_ece():
    registry = ModelRegistry()
    original_thresholds = dict(PROMOTION_THRESHOLDS)
    try:
        local_meta = {
            "metrics": {"calibration_score": 0.03},
            "train_size": 100,
            "test_size": 100,
        }
        global_meta = {
            "metrics": {"calibration_score": 0.06},
            "train_size": 5000,
            "test_size": 1000,
        }

        registry._select_best_model(local_meta, global_meta, "poisson_home_base", "PL")

        assert PROMOTION_THRESHOLDS["poisson_home_base::PL"] == 450
    finally:
        PROMOTION_THRESHOLDS.clear()
        PROMOTION_THRESHOLDS.update(original_thresholds)


def test_promotion_threshold_raises_when_global_wins():
    registry = ModelRegistry()
    original_thresholds = dict(PROMOTION_THRESHOLDS)
    try:
        PROMOTION_THRESHOLDS["poisson_home_base::SA"] = 300
        local_meta = {
            "metrics": {"calibration_score": 0.08},
            "train_size": 100,
            "test_size": 100,
        }
        global_meta = {
            "metrics": {"calibration_score": 0.04},
            "train_size": 5000,
            "test_size": 1000,
        }

        registry._select_best_model(local_meta, global_meta, "poisson_home_base", "SA")

        assert PROMOTION_THRESHOLDS["poisson_home_base::SA"] == 330
    finally:
        PROMOTION_THRESHOLDS.clear()
        PROMOTION_THRESHOLDS.update(original_thresholds)


def test_promotion_threshold_is_clamped_to_50_500():
    registry = ModelRegistry()
    original_thresholds = dict(PROMOTION_THRESHOLDS)
    try:
        local_meta = {
            "metrics": {"calibration_score": 0.01},
            "train_size": 100,
            "test_size": 100,
        }
        global_meta = {
            "metrics": {"calibration_score": 0.06},
            "train_size": 5000,
            "test_size": 1000,
        }

        PROMOTION_THRESHOLDS["poisson_home_base::PD"] = 52
        registry._select_best_model(local_meta, global_meta, "poisson_home_base", "PD")
        assert PROMOTION_THRESHOLDS["poisson_home_base::PD"] == 50

        PROMOTION_THRESHOLDS["poisson_home_base::PD"] = 490
        local_meta["metrics"]["calibration_score"] = 0.08
        global_meta["metrics"]["calibration_score"] = 0.04
        registry._select_best_model(local_meta, global_meta, "poisson_home_base", "PD")
        assert PROMOTION_THRESHOLDS["poisson_home_base::PD"] == 500
    finally:
        PROMOTION_THRESHOLDS.clear()
        PROMOTION_THRESHOLDS.update(original_thresholds)


def test_smart_routing_override_warning_logs_once_per_league(monkeypatch):
    registry = ModelRegistry()
    monkeypatch.setattr(registry, "_logged_smart_routing", set(), raising=False)

    local_meta = {
        "metrics": {"calibration_score": 0.09},
        "train_size": 150,
        "test_size": 150,
    }
    global_meta = {
        "metrics": {"calibration_score": 0.05},
        "train_size": 5000,
        "test_size": 1000,
    }

    with patch.object(registry, "_calibrate_promotion_threshold"), patch.object(
        registry,
        "_get_promotion_threshold",
        return_value=100,
    ), patch("src.ml.registry.logger.warning") as mock_warning:
        registry._select_best_model(local_meta, global_meta, "poisson_home_base", "PL")
        registry._select_best_model(local_meta, global_meta, "poisson_away_base", "PL")

    assert mock_warning.call_count == 1
    assert "PL" in registry._logged_smart_routing


def test_load_models_for_market_returns_all_loaded_models():
    registry = ModelRegistry()
    with patch.object(
        registry,
        "load_model",
        side_effect=lambda name, league=None: {"name": name, "league": league},
    ):
        loaded = registry.load_models_for_market(
            market="goals_lambda",
            model_names=["home_goals", "away_goals", "poisson_home_base", "poisson_away_base"],
            league="PL",
        )

    assert set(loaded.keys()) == {
        "home_goals",
        "away_goals",
        "poisson_home_base",
        "poisson_away_base",
    }


def test_load_models_for_market_raises_when_any_model_missing():
    registry = ModelRegistry()

    def _loader(name: str, league: str | None = None):
        if name == "away_goals":
            raise ModelNotFoundError("missing away_goals")
        return {"name": name, "league": league}

    with patch.object(registry, "load_model", side_effect=_loader):
        with pytest.raises(ModelNotFoundError, match="Missing required models for market"):
            registry.load_models_for_market(
                market="goals_lambda",
                model_names=["home_goals", "away_goals"],
                league="PL",
            )


class _FakeS3FileSystem:
    storage: dict[str, bytes] = {}
    created_with: list[dict[str, object]] = []

    def __init__(self, key=None, secret=None, client_kwargs=None, config_kwargs=None):
        from fsspec.asyn import get_loop

        self.loop = get_loop()
        self.__class__.created_with.append(
            {
                "key": key,
                "secret": secret,
                "client_kwargs": client_kwargs,
                "config_kwargs": config_kwargs,
            }
        )

    def put(self, local_path: str, remote_path: str) -> None:
        self.__class__.storage[remote_path] = Path(local_path).read_bytes()

    def exists(self, remote_path: str) -> bool:
        return remote_path in self.__class__.storage

    async def _exists(self, remote_path: str) -> bool:
        return self.exists(remote_path)

    def get(self, remote_path: str, local_path: str) -> None:
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.__class__.storage[remote_path])

    async def _get_file(self, remote_path: str, local_path: str) -> None:
        self.get(remote_path, local_path)


def test_push_to_s3_uploads_manifest_listed_pkl_files_only(tmp_path, monkeypatch):
    registry = ModelRegistry()
    original_manifest = dict(registry.manifest)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "alpha.pkl").write_bytes(b"alpha")
    (model_dir / "beta.joblib").write_bytes(b"beta")

    try:
        registry.manifest = {
            "alpha_model": {"filename": "alpha.pkl"},
            "beta_model": {"filename": "beta.joblib"},
            "active_models": {},
        }
        monkeypatch.setattr(registry_module, "MODELS_DIR", model_dir)
        monkeypatch.setattr(registry_module.s3fs, "S3FileSystem", _FakeS3FileSystem)
        _FakeS3FileSystem.storage = {}
        _FakeS3FileSystem.created_with = []
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv("AWS_REGION", "af-south-1")
        monkeypatch.setenv("ARTIFACT_CONNECT_TIMEOUT_SECONDS", "7")
        monkeypatch.setenv("ARTIFACT_TRANSFER_TIMEOUT_SECONDS", "19")

        uploaded = registry.push_to_s3(bucket="test-bucket", prefix="models/prod")

        assert uploaded == 1
        assert _FakeS3FileSystem.storage == {
            "test-bucket/models/prod/alpha.pkl": b"alpha"
        }
        assert _FakeS3FileSystem.created_with[0]["client_kwargs"] == {"region_name": "af-south-1"}
        assert _FakeS3FileSystem.created_with[0]["config_kwargs"] == {
            "connect_timeout": 7.0,
            "read_timeout": 19.0,
        }
    finally:
        registry.manifest = original_manifest


def test_pull_from_s3_downloads_missing_manifest_files(tmp_path, monkeypatch):
    registry = ModelRegistry()
    original_manifest = dict(registry.manifest)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    existing_path = model_dir / "existing.pkl"
    existing_path.write_bytes(b"keep-local")

    try:
        registry.manifest = {
            "alpha_model": {"filename": "nested/alpha.pkl"},
            "beta_model": {"filename": "beta.joblib"},
            "existing_model": {"filename": "existing.pkl"},
            "shadow_models": {},
        }
        monkeypatch.setattr(registry_module, "MODELS_DIR", model_dir)
        monkeypatch.setattr(registry_module.s3fs, "S3FileSystem", _FakeS3FileSystem)
        _FakeS3FileSystem.storage = {
            "test-bucket/models/prod/nested/alpha.pkl": b"alpha",
            "test-bucket/models/prod/beta.joblib": b"beta",
        }
        _FakeS3FileSystem.created_with = []
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv("AWS_REGION", "af-south-1")

        downloaded = registry.pull_from_s3(bucket="test-bucket", prefix="models/prod")

        assert downloaded == 2
        assert (model_dir / "nested" / "alpha.pkl").read_bytes() == b"alpha"
        assert (model_dir / "beta.joblib").read_bytes() == b"beta"
        assert existing_path.read_bytes() == b"keep-local"
    finally:
        registry.manifest = original_manifest


def test_pull_from_s3_timeout_is_distinct_and_removes_partial_file(
    tmp_path,
    monkeypatch,
    caplog,
):
    registry = ModelRegistry()
    original_manifest = dict(registry.manifest)
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        registry.manifest = {"alpha_model": {"filename": "alpha.pkl"}}
        monkeypatch.setattr(registry_module, "MODELS_DIR", model_dir)
        monkeypatch.setattr(registry_module.s3fs, "S3FileSystem", _FakeS3FileSystem)
        _FakeS3FileSystem.storage = {
            "test-bucket/models/prod/alpha.pkl": b"complete-remote-artifact"
        }
        _FakeS3FileSystem.created_with = []
        deadline_calls = []

        def fake_fsspec_sync(loop, func, *args, timeout=None):
            deadline_calls.append((func.__name__, timeout))
            if func.__name__ == "_exists":
                return True
            Path(args[1]).write_bytes(b"partial-download")
            raise TimeoutError("simulated total S3 transfer deadline")

        monkeypatch.setattr(registry_module, "fsspec_sync", fake_fsspec_sync)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
        monkeypatch.setenv("AWS_REGION", "af-south-1")
        monkeypatch.setenv("ARTIFACT_CONNECT_TIMEOUT_SECONDS", "3")
        monkeypatch.setenv("ARTIFACT_TRANSFER_TIMEOUT_SECONDS", "11")

        with caplog.at_level(logging.ERROR, logger="src.ml.registry"):
            with pytest.raises(ArtifactTransferTimeoutError) as exc_info:
                registry.pull_from_s3(bucket="test-bucket", prefix="models/prod")

        assert exc_info.value.context["operation"] == "download"
        assert exc_info.value.context["connect_timeout_seconds"] == 3.0
        assert exc_info.value.context["read_timeout_seconds"] == 11.0
        assert exc_info.value.context["total_timeout_seconds"] == 11.0
        assert deadline_calls == [("_exists", 11.0), ("_get_file", 11.0)]
        assert not (model_dir / "alpha.pkl").exists()
        assert list(model_dir.rglob("*.part")) == []
        messages = "\n".join(record.getMessage() for record in caplog.records)
        assert "path=s3://test-bucket/models/prod/alpha.pkl" in messages
        assert "operation=download" in messages
        assert "connect_timeout_seconds=3.0" in messages
        assert "read_timeout_seconds=11.0" in messages
        assert "total_timeout_seconds=11.0" in messages
    finally:
        registry.manifest = original_manifest


def test_s3_timeout_configuration_rejects_non_positive_values(monkeypatch):
    registry = ModelRegistry()
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_REGION", "af-south-1")
    monkeypatch.setenv("ARTIFACT_CONNECT_TIMEOUT_SECONDS", "0")

    with pytest.raises(ConfigurationError, match="positive integer number of seconds"):
        registry._build_s3_filesystem()


def test_push_to_s3_requires_aws_env(monkeypatch):
    registry = ModelRegistry()
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.delenv("AWS_REGION", raising=False)

    with pytest.raises(EnvironmentError, match="AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION"):
        registry.push_to_s3(bucket="test-bucket", prefix="models/prod")


def test_manifest_load_uses_filesystem_authority(tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    manifest_file = models_dir / "manifest.json"
    backup_file = models_dir / "manifest.json.bak"
    monkeypatch.setattr(registry_module, "MODELS_DIR", models_dir)
    monkeypatch.setattr(ModelRegistry, "MANIFEST_FILE", manifest_file)
    monkeypatch.setattr(ModelRegistry, "BACKUP_FILE", backup_file)

    try:
        manifest = {
            "alpha_model": {
                "name": "poisson_home_base",
                "version": "1.0.0",
                "league": "PL",
                "metrics": {"brier_score": 0.21},
            },
            "beta_model": {
                "name": "poisson_away_base",
                "version": "2.0.0",
                "league": "PL",
                "metrics": {"brier_score": 0.19},
            },
        }
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        ModelRegistry._instance = None
        ModelRegistry._manifest_cache = None
        reloaded = object.__new__(ModelRegistry)
        reloaded._logged_smart_routing = set()
        reloaded._load_manifest()

        assert reloaded.manifest["alpha_model"]["version"] == "1.0.0"
        assert reloaded.manifest["beta_model"]["version"] == "2.0.0"
    finally:
        ModelRegistry._instance = None
        ModelRegistry._manifest_cache = None

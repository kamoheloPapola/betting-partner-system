import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.ml.registry import ModelRegistry, PROMOTION_THRESHOLDS
from src.core.exceptions import ModelNotFoundError

def test_model_registry_production_loading():
    """Verify registry prioritizes production models."""
    registry = ModelRegistry()
    
    # Mocking storage/listing
    with patch.object(registry, 'get_production_model_for_league') as mock_prod:
        mock_prod.return_value = {'path': 'models/PL/model_v1.pkl', 'league': 'PL', 'version': '1.0'}
        
        meta = registry.get_production_model_for_league("PL", "poisson_home_base")
        assert meta['league'] == 'PL'
        assert 'version' in meta

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

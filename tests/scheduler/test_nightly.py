from datetime import datetime, timezone

import pandas as pd

from src.scheduler import nightly
from src.ml.training.model_configs import FeatureSet, ModelType


def test_find_stale_models_filters_latest_productive_by_age():
    class DummyRegistry:
        manifest = {
            "poisson_home_base_old_PL": {
                "name": "poisson_home_base",
                "league": "PL",
                "status": "productive",
                "registered_at": "2026-03-10T00:00:00+00:00",
            },
            "poisson_home_base_new_PL": {
                "name": "poisson_home_base",
                "league": "PL",
                "status": "productive",
                "registered_at": "2026-03-29T00:00:00+00:00",
            },
            "nb_home_corners_base_old_SA": {
                "name": "nb_home_corners_base",
                "league": "SA",
                "status": "productive",
                "registered_at": "2026-03-01T00:00:00+00:00",
            },
            "active_models": {},
        }

    stale = nightly.find_stale_models(
        registry=DummyRegistry(),
        stale_days=7,
        as_of=datetime(2026, 3, 31, tzinfo=timezone.utc),
    )

    assert len(stale) == 1
    assert stale[0].model_name == "nb_home_corners_base"
    assert stale[0].league == "SA"


def test_run_nightly_executes_expected_sequence(monkeypatch):
    calls = []

    target = nightly.StaleModelTarget(
        model_name="poisson_home_base",
        league="PL",
        manifest_key="old_key",
        trained_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    outcome = nightly.RetrainOutcome(
        target=target,
        before_key="old_key",
        after_key="new_key",
        improved=True,
    )
    auto_target = nightly.AutoRetrainTarget(
        model_name="poisson_home_base",
        league="PL",
        market="home_win",
        manifest_key="old_key",
        version="1.0.0",
        baseline_brier=0.2,
        rolling_brier=0.3,
        sample_size=30,
    )
    auto_outcome = nightly.AutoRetrainOutcome(
        target=auto_target,
        before_key="old_key",
        after_key="new_key",
        promoted=True,
        old_brier=0.2,
        new_brier=0.18,
        reason="promoted",
    )

    monkeypatch.setattr(nightly, "ModelRegistry", lambda: object())
    monkeypatch.setattr(nightly, "ModelTrainer", lambda registry: object())
    monkeypatch.setattr(
        nightly,
        "fetch_latest_data",
        lambda season, force=False: calls.append(("fetch", season, force)),
    )
    monkeypatch.setattr(
        nightly,
        "build_feature_matrix",
        lambda: calls.append(("build",)) or {"status": "success"},
    )
    monkeypatch.setattr(
        nightly,
        "find_stale_models",
        lambda registry, stale_days=7: calls.append(("find", stale_days)) or [target],
    )
    monkeypatch.setattr(
        nightly,
        "load_training_dataframe",
        lambda path=nightly.FEATURE_MATRIX_PATH: calls.append(("load", str(path)))
        or pd.DataFrame({"league": ["PL"]}),
    )
    monkeypatch.setattr(
        nightly,
        "retrain_stale_models",
        lambda registry, trainer, stale_targets, feature_df: calls.append(
            ("retrain", len(stale_targets), len(feature_df))
        )
        or [outcome],
    )
    monkeypatch.setattr(
        nightly,
        "promote_if_improved",
        lambda registry, outcomes: calls.append(("promote", len(outcomes))) or 1,
    )
    monkeypatch.setattr(
        nightly,
        "generate_performance_report",
        lambda: calls.append(("report",)) or {"status": "OK"},
    )
    monkeypatch.setattr(
        nightly,
        "load_recent_evaluations",
        lambda: calls.append(("evals",)) or pd.DataFrame({"x": [1]}),
    )
    monkeypatch.setattr(
        nightly,
        "find_auto_retrain_targets",
        lambda registry, evals_df: calls.append(("auto-find", len(evals_df))) or [auto_target],
    )
    monkeypatch.setattr(
        nightly,
        "auto_retrain_underperforming_models",
        lambda registry, trainer, feature_df, evals_df: calls.append(("auto-retrain", len(feature_df), len(evals_df)))
        or [auto_outcome],
    )
    monkeypatch.setattr(
        nightly,
        "run_model_cleanup",
        lambda keep_versions=3: calls.append(("cleanup", keep_versions))
        or {"deleted_files": 4, "freed_bytes": 2048},
    )
    monkeypatch.setattr(
        nightly,
        "record_nightly_success",
        lambda summary: calls.append(("heartbeat", summary)),
    )

    summary = nightly.run_nightly(season="2526", force_fetch=True, stale_days=7)

    assert [step[0] for step in calls] == [
        "fetch",
        "build",
        "find",
        "load",
        "retrain",
        "promote",
        "report",
        "evals",
        "auto-find",
        "auto-retrain",
        "cleanup",
        "heartbeat",
    ]
    assert calls[-1][1] == {"status": "ok", "season": "2526"}
    assert summary["retrained"] == 1
    assert summary["promoted"] == 1
    assert summary["auto_retrain_triggered"] == 1
    assert summary["auto_retrained"] == 1
    assert summary["auto_promoted"] == 1
    assert summary["cleanup_deleted"] == 4
    assert summary["cleanup_freed_bytes"] == 2048


def test_find_auto_retrain_targets_detects_rolling_brier_breach():
    class DummyRegistry:
        manifest = {
            "poisson_home_base_v1.0.0_PL": {
                "name": "poisson_home_base",
                "league": "PL",
                "version": "1.0.0",
                "status": "productive",
                "baseline_brier": 0.20,
                "registered_at": "2026-03-20T00:00:00+00:00",
            },
            "active_models": {},
        }

    evals_df = pd.DataFrame(
        {
            "league": ["PL"] * 30,
            "market": ["home_win"] * 30,
            "predicted_probability": [0.9] * 30,
            "actual_outcome": [0.0] * 30,
            "resolved_date": pd.date_range("2026-03-01", periods=30, freq="D"),
        }
    )

    targets = nightly.find_auto_retrain_targets(registry=DummyRegistry(), evals_df=evals_df)
    assert len(targets) == 1
    assert targets[0].model_name == "poisson_home_base"
    assert targets[0].market == "home_win"
    assert targets[0].sample_size == 30
    assert targets[0].rolling_brier > targets[0].baseline_brier * 1.15


def test_auto_retrain_underperforming_models_alerts_when_validation_fails(monkeypatch):
    class DummyHistoryDB:
        def __init__(self):
            self.events = []

        def write_event(self, **kwargs):
            self.events.append(kwargs)

    class DummyRegistry:
        def __init__(self):
            self.manifest = {
                "old_key": {"version": "1.0.0", "brier_score": 0.20},
                "new_key": {"version": "1.1.0", "brier_score": 0.30},
            }
            self.active_calls = []
            self.history = DummyHistoryDB()

        def HISTORY_DB(self):
            return self.history

        def set_active_model(self, model_type, exact_manifest_key, league=None):
            self.active_calls.append((model_type, exact_manifest_key, league))

    class DummyTrainer:
        def __init__(self):
            self.calls = []

        def train_model(self, **kwargs):
            self.calls.append(kwargs)

    class DummyAlerter:
        def __init__(self):
            self.calls = []

        def send_alert(self, message, context=None, severity="WARNING"):
            self.calls.append((message, context, severity))
            return True

    target = nightly.AutoRetrainTarget(
        model_name="poisson_home_base",
        league="PL",
        market="home_win",
        manifest_key="old_key",
        version="1.0.0",
        baseline_brier=0.20,
        rolling_brier=0.31,
        sample_size=30,
    )
    key_sequence = iter(["old_key", "new_key"])
    registry = DummyRegistry()
    trainer = DummyTrainer()
    alerter = DummyAlerter()

    monkeypatch.setattr(nightly, "find_auto_retrain_targets", lambda registry, evals_df: [target])
    monkeypatch.setattr(nightly, "_latest_productive_key", lambda registry, model_name, league: next(key_sequence))
    monkeypatch.setattr(
        nightly,
        "_config_map",
        lambda: {
            "poisson_home_base": {
                "name": "poisson_home_base",
                "target": "home_score",
                "feature_set": FeatureSet.BASE,
                "params": {},
                "type": ModelType.NB,
            }
        },
    )
    monkeypatch.setattr(nightly, "select_features", lambda df, target, feature_set: ["feat"])

    feature_df = pd.DataFrame({"league": ["PL", "PL"], "feat": [0.1, 0.2], "home_score": [1, 0], "date": pd.date_range("2026-01-01", periods=2)})
    evals_df = pd.DataFrame({"league": ["PL"], "market": ["home_win"], "predicted_probability": [0.9], "actual_outcome": [0.0]})

    outcomes = nightly.auto_retrain_underperforming_models(
        registry=registry,
        trainer=trainer,
        feature_df=feature_df,
        evals_df=evals_df,
        alerter=alerter,
    )

    assert len(outcomes) == 1
    assert outcomes[0].promoted is False
    assert outcomes[0].reason == "validation_failed"
    assert registry.active_calls == []
    assert len(alerter.calls) == 1
    assert "failed to improve performance" in alerter.calls[0][0]
    event_types = [event["event_type"] for event in registry.history.events]
    assert event_types == ["auto_retrain_triggered", "auto_retrain_rejected"]


def test_auto_retrain_underperforming_models_logs_missing_training_data_event(monkeypatch):
    class DummyHistoryDB:
        def __init__(self):
            self.events = []

        def write_event(self, **kwargs):
            self.events.append(kwargs)

    class DummyRegistry:
        def __init__(self):
            self.manifest = {"active_models": {}}
            self.history = DummyHistoryDB()

        def HISTORY_DB(self):
            return self.history

        def set_active_model(self, model_type, exact_manifest_key, league=None):
            raise AssertionError("set_active_model should not be called when training data is missing")

    class DummyTrainer:
        def train_model(self, **kwargs):
            raise AssertionError("train_model should not run when training data is missing")

    class DummyAlerter:
        def __init__(self):
            self.calls = []

        def send_alert(self, message, context=None, severity="WARNING"):
            self.calls.append((message, context, severity))
            return True

    target = nightly.AutoRetrainTarget(
        model_name="poisson_home_base",
        league="PL",
        market="home_win",
        manifest_key="old_key",
        version="1.0.0",
        baseline_brier=0.20,
        rolling_brier=0.31,
        sample_size=30,
    )
    registry = DummyRegistry()
    trainer = DummyTrainer()
    alerter = DummyAlerter()

    monkeypatch.setattr(nightly, "find_auto_retrain_targets", lambda registry, evals_df: [target])
    monkeypatch.setattr(
        nightly,
        "_config_map",
        lambda: {
            "poisson_home_base": {
                "name": "poisson_home_base",
                "target": "home_score",
                "feature_set": FeatureSet.BASE,
                "params": {},
                "type": ModelType.NB,
            }
        },
    )

    feature_df = pd.DataFrame(
        {
            "league": ["SA", "SA"],
            "feat": [0.1, 0.2],
            "home_score": [1, 0],
            "date": pd.date_range("2026-01-01", periods=2),
        }
    )
    evals_df = pd.DataFrame(
        {
            "league": ["PL"],
            "market": ["home_win"],
            "predicted_probability": [0.9],
            "actual_outcome": [0.0],
        }
    )

    outcomes = nightly.auto_retrain_underperforming_models(
        registry=registry,
        trainer=trainer,
        feature_df=feature_df,
        evals_df=evals_df,
        alerter=alerter,
    )

    assert len(outcomes) == 1
    assert outcomes[0].promoted is False
    assert outcomes[0].reason == "no_training_rows"
    assert len(alerter.calls) == 1
    assert "no training data" in alerter.calls[0][0].lower()
    event_types = [event["event_type"] for event in registry.history.events]
    assert event_types == ["auto_retrain_triggered", "auto_retrain_data_missing"]


def test_main_forwards_cli_args(monkeypatch):
    captured = {}

    def _fake_run_nightly(*, season, force_fetch, stale_days):
        captured["season"] = season
        captured["force_fetch"] = force_fetch
        captured["stale_days"] = stale_days
        return {"status": "ok"}

    monkeypatch.setattr(nightly, "run_nightly", _fake_run_nightly)
    nightly.main(["--season", "2526", "--force-fetch", "--stale-days", "9"])

    assert captured == {"season": "2526", "force_fetch": True, "stale_days": 9}

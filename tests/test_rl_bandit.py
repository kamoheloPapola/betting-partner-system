from __future__ import annotations

import copy
from datetime import datetime

import pandas as pd

import src.cli.commands.prediction as prediction_module
import src.ml.registry as registry_module
import src.monitoring.drift_monitor as drift_monitor_module
import src.simulation.rl_bandit as rl_bandit_module
from src.simulation.match_simulator import MatchSimulator
from src.simulation.rl_bandit import ContextualBandit


class _DummyProgress:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def add_task(self, *args, **kwargs) -> int:
        return 1

    def advance(self, task_id: int) -> None:
        return None


class _FakeFeatureDriftMonitor:
    def check_live_prediction(self, **kwargs):
        return {"status": "ok"}


class _FakeModelRegistry:
    def __init__(self) -> None:
        self._logged_smart_routing = set()


class _FakeModel:
    meta = {"version": "test-version"}


class _FakeOrchestrator:
    GO = "GO"
    WATCH = "WATCH"
    STOP = "STOP"

    def __init__(self) -> None:
        self.global_status = self.GO

    def load_global_state(self) -> None:
        return None


class _FixedDateTime:
    @classmethod
    def now(cls) -> datetime:
        return datetime(2026, 4, 3, 12, 0, 0)


def test_cold_start_initializes_weights_to_one(monkeypatch, tmp_path):
    monkeypatch.setattr(
        rl_bandit_module,
        "load_resolved_predictions_from_db",
        lambda container=None: pd.DataFrame(),
    )

    bandit = ContextualBandit(state_path=tmp_path / "bandit.json", auto_load=False, auto_bootstrap=True)

    assert bandit.get_weights(bandit.context_key("PL", "1x2")) == {
        "tempo_sigma_scale": 1.0,
        "lambda_scale_home": 1.0,
        "lambda_scale_away": 1.0,
    }


def test_update_moves_weights_in_correct_direction(tmp_path):
    low_ece_bandit = ContextualBandit(
        state_path=tmp_path / "low.json",
        auto_load=False,
        auto_bootstrap=False,
    )
    high_ece_bandit = ContextualBandit(
        state_path=tmp_path / "high.json",
        auto_load=False,
        auto_bootstrap=False,
    )
    context = low_ece_bandit.context_key("PL", "1x2")

    low_ece_bandit.update(context, ece=0.05)
    high_ece_bandit.update(context, ece=0.40)

    low_weights = low_ece_bandit.get_weights(context)
    high_weights = high_ece_bandit.get_weights(context)

    assert low_weights["tempo_sigma_scale"] > high_weights["tempo_sigma_scale"]
    assert low_weights["lambda_scale_home"] > high_weights["lambda_scale_home"]
    assert low_weights["lambda_scale_away"] > high_weights["lambda_scale_away"]


def test_simulate_with_rl_weights_changes_output():
    simulator = MatchSimulator(n_simulations=5_000, seed=123)
    baseline = simulator.simulate(1.7, 1.2)

    weighted = MatchSimulator(n_simulations=5_000, seed=123).simulate(
        1.7,
        1.2,
        rl_weights={
            "tempo_sigma_scale": 1.2,
            "lambda_scale_home": 1.1,
            "lambda_scale_away": 0.9,
        },
    )

    assert weighted.rl_weights_applied is True
    assert weighted.home_win_prob != baseline.home_win_prob
    assert weighted.expected_home_goals != baseline.expected_home_goals


def test_use_rl_weights_false_leaves_prediction_output_unchanged(monkeypatch):
    fake_container = type(
        "FakeContainer",
        (),
        {"registry": type("FakeRegistry", (), {"_logged_smart_routing": set()})()},
    )()

    monkeypatch.setattr(prediction_module, "Progress", _DummyProgress)
    monkeypatch.setattr(prediction_module, "DriftOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(prediction_module, "datetime", _FixedDateTime)
    monkeypatch.setattr(prediction_module, "_load_prediction_models", lambda league: {"mh_goals": _FakeModel()})
    monkeypatch.setattr(
        prediction_module,
        "_calculate_probabilities_v2",
        lambda match, suite, lg, is_intensity, warning_collector, use_simulator=True: (
            {
                "home": 0.55,
                "draw": 0.23,
                "away": 0.22,
                "goal_model_home_lambda": 1.4,
                "goal_model_away_lambda": 1.1,
            },
            {"core": {}},
        ),
    )
    monkeypatch.setattr(prediction_module, "_log_evt", lambda *args, **kwargs: None)
    monkeypatch.setattr(prediction_module, "_run_shadow_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(prediction_module, "_get_pipeline_version_hash", lambda: "test-pipeline")
    monkeypatch.setattr(
        prediction_module.ServiceContainer,
        "get_instance",
        classmethod(lambda cls: fake_container),
    )
    monkeypatch.setattr(drift_monitor_module, "FeatureDriftMonitor", _FakeFeatureDriftMonitor)
    monkeypatch.setattr(registry_module, "ModelRegistry", _FakeModelRegistry)

    class _UnexpectedBandit:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("ContextualBandit should not be instantiated when use_rl_weights=False")

    monkeypatch.setattr(prediction_module, "ContextualBandit", _UnexpectedBandit)

    df = pd.DataFrame(
        [
            {
                "league": "PL",
                "home_team": "Home FC",
                "away_team": "Away FC",
                "date": "2026-04-01",
                "match_id": "match-1",
            }
        ]
    )

    baseline = copy.deepcopy(prediction_module._run_predict_loop(df, use_simulator=True))
    explicit_false = copy.deepcopy(
        prediction_module._run_predict_loop(df, use_simulator=True, use_rl_weights=False)
    )

    assert baseline == explicit_false

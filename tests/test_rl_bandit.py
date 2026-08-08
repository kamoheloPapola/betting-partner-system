from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest
import src.cli.commands.prediction as prediction_module
import src.ml.registry as registry_module
import src.monitoring.drift_monitor as drift_monitor_module
import src.simulation.rl_bandit as rl_bandit_module
from src.core.constants import RESOLVER_LOOKBACK_DAYS
from src.simulation.match_simulator import MatchSimulator
from src.simulation.rl_bandit import ContextualBandit, load_resolved_predictions


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
        "load_resolved_predictions",
        lambda outcomes_path=None: pd.DataFrame(),
    )

    bandit = ContextualBandit(state_path=tmp_path / "bandit.json", auto_load=False, auto_bootstrap=True)

    assert bandit.get_weights(bandit.context_key("PL", "1x2")) == {
        "tempo_sigma_scale": 1.0,
        "lambda_scale_home": 1.0,
        "lambda_scale_away": 1.0,
    }


def test_update_moves_weights_in_correct_direction(tmp_path):
    bandit = ContextualBandit(
        state_path=tmp_path / "case1.json",
        auto_load=False,
        auto_bootstrap=False,
    )
    context = bandit.context_key("PL", "1x2")

    # Case 1: high ECE, no mean_error -> only tempo_sigma_scale moves, lambdas stay at 1.0
    bandit.update(context, ece=0.4, mean_error=0.0)
    weights = bandit.get_weights(context)
    assert weights["tempo_sigma_scale"] < 1.0
    assert weights["lambda_scale_home"] == pytest.approx(1.0, abs=1e-6)
    assert weights["lambda_scale_away"] == pytest.approx(1.0, abs=1e-6)

    # Case 2: overconfident (mean_error > 0) -> lambda scales suppressed below 1.0
    bandit2 = ContextualBandit(
        state_path=tmp_path / "case2.json",
        auto_load=False,
        auto_bootstrap=False,
    )
    bandit2.update(context, ece=0.1, mean_error=0.3)
    weights2 = bandit2.get_weights(context)
    assert weights2["lambda_scale_home"] < 1.0
    assert weights2["lambda_scale_away"] < 1.0

    # Case 3: underconfident (mean_error < 0) -> lambda scales boosted above 1.0
    bandit3 = ContextualBandit(
        state_path=tmp_path / "case3.json",
        auto_load=False,
        auto_bootstrap=False,
    )
    bandit3.update(context, ece=0.1, mean_error=-0.3)
    weights3 = bandit3.get_weights(context)
    assert weights3["lambda_scale_home"] > 1.0
    assert weights3["lambda_scale_away"] > 1.0


def test_refresh_skips_thin_ece_buckets(monkeypatch, tmp_path, caplog):
    resolved = pd.DataFrame(
        [
            {
                "league": "PL",
                "market": "btts_yes",
                "probability": 0.6,
                "actual_outcome": idx % 2,
                "resolved_at": None,
            }
            for idx in range(29)
        ]
        + [
            {
                "league": "BL1",
                "market": "btts_yes",
                "probability": 0.6,
                "actual_outcome": idx % 2,
                "resolved_at": None,
            }
            for idx in range(30)
        ]
    )
    monkeypatch.setattr(rl_bandit_module, "calculate_ece", lambda actuals, probs, n_bins: 0.2)
    monkeypatch.setattr(
        rl_bandit_module,
        "load_resolved_predictions",
        lambda outcomes_path=None: resolved,
    )

    bandit = ContextualBandit(
        state_path=tmp_path / "bandit.json",
        auto_load=False,
        auto_bootstrap=False,
    )

    with caplog.at_level("WARNING"):
        updated = bandit.refresh_from_resolved_predictions()

    skipped_context = bandit.context_key("PL", "btts_yes")
    updated_context = bandit.context_key("BL1", "btts_yes")

    assert f"Skipping ECE update for {skipped_context}: only 29 rows" in caplog.text
    assert skipped_context not in bandit.state
    assert updated_context in bandit.state
    assert set(updated) == {updated_context}
    assert updated[updated_context]["row_count"] == 30
    assert bandit.state[updated_context]["count"] == 1


def test_load_resolved_predictions_respects_lookback(tmp_path):
    old_kickoff = datetime.now(timezone.utc) - timedelta(days=RESOLVER_LOOKBACK_DAYS + 10)
    recent_kickoff = datetime.now(timezone.utc) - timedelta(days=5)
    resolved_at = datetime.now(timezone.utc)
    outcomes_path = tmp_path / "prediction_outcomes.csv"
    pd.DataFrame(
        [
            {
                "prediction_id": "old_pred",
                "match_hash": "hash_old",
                "league": "PL",
                "kickoff_date": old_kickoff.isoformat(),
                "market": "btts_yes",
                "probability": 0.61,
                "outcome": "WON",
                "resolved_at": resolved_at.isoformat(),
            },
            {
                "prediction_id": "recent_pred",
                "match_hash": "hash_recent",
                "league": "PL",
                "kickoff_date": recent_kickoff.isoformat(),
                "market": "btts_yes",
                "probability": 0.57,
                "outcome": "LOST",
                "resolved_at": resolved_at.isoformat(),
            },
            {
                "prediction_id": "null_kickoff_pred",
                "match_hash": "hash_null",
                "league": "PL",
                "kickoff_date": None,
                "market": "btts_yes",
                "probability": 0.52,
                "outcome": "WON",
                "resolved_at": resolved_at.isoformat(),
            },
        ]
    ).to_csv(outcomes_path, index=False)

    result = load_resolved_predictions(outcomes_path)

    assert len(result) == 1
    assert result.iloc[0]["league"] == "PL"
    assert result.iloc[0]["market"] == "btts_yes"
    assert result.iloc[0]["probability"] == pytest.approx(0.57)
    assert result.iloc[0]["actual_outcome"] == 0


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

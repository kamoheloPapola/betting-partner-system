import pandas as pd
from rich.console import Console

import src.cli.commands.prediction as prediction_module
import src.ml.registry as registry_module
import src.monitoring.drift_monitor as drift_monitor_module


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
    pass


class _FakeModel:
    meta = {"version": "test-version"}


class _FakeOrchestrator:
    GO = "GO"
    WATCH = "WATCH"
    STOP = "STOP"
    instances = []

    def __init__(self) -> None:
        self.global_status = self.GO
        self.load_calls = 0
        self.__class__.instances.append(self)

    def load_global_state(self) -> None:
        self.load_calls += 1

    def evaluate_league_drift(self, league, current_session_data=None):
        self.load_calls += 1
        return self.GO


def test_run_predict_loop_reads_persisted_global_drift_state(monkeypatch):
    monkeypatch.setattr(prediction_module, "Progress", _DummyProgress)
    monkeypatch.setattr(prediction_module, "DriftOrchestrator", _FakeOrchestrator)
    monkeypatch.setattr(prediction_module, "_load_prediction_models", lambda league: {"mh_goals": _FakeModel()})
    monkeypatch.setattr(
        prediction_module,
        "_calculate_probabilities_v2",
        lambda match, suite, lg, is_intensity, warning_collector, simulator=None, use_simulator=True, rl_bandit=None: (
            {"home": 0.55},
            {"core": {}},
        ),
    )
    monkeypatch.setattr(prediction_module, "_log_evt", lambda *args, **kwargs: None)
    monkeypatch.setattr(prediction_module, "_run_shadow_predictions", lambda *args, **kwargs: None)
    monkeypatch.setattr(prediction_module, "_get_pipeline_version_hash", lambda: "test-pipeline")
    monkeypatch.setattr(drift_monitor_module, "FeatureDriftMonitor", _FakeFeatureDriftMonitor)
    monkeypatch.setattr(registry_module, "ModelRegistry", _FakeModelRegistry)

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

    results = prediction_module._run_predict_loop(df, use_simulator=False)

    assert len(results) == 1
    assert results[0]["attribution_stamp"]["core"]["drift_state"] == "OK"
    assert len(_FakeOrchestrator.instances) == 1
    assert _FakeOrchestrator.instances[0].load_calls == 1


def test_render_output_high_conf_counts_gated_selections(monkeypatch):
    monkeypatch.setattr(prediction_module, "_get_terminal_width", lambda: 140)

    console = Console(record=True, width=160)
    match_time = pd.Timestamp("2026-04-01T15:00:00Z")
    preds = [
        {
            "league": "PL",
            "time": match_time,
            "match": "Home FC vs Away FC",
            "home": 0.52,
            "draw": 0.24,
            "away": 0.24,
            "o25": 0.58,
            "u25": 0.42,
            "btts": 0.57,
            "btts_no": 0.43,
            "dc_1x": 0.70,
            "dc_x2": 0.48,
            "dc_12": 0.76,
            "card_o25": 0.55,
            "card_u55": 0.30,
            "corn_o75": 0.45,
            "corn_u11": 0.46,
            "home_under_1_5": 0.90,
            "away_under_1_5": 0.35,
            "corn_1x2_h": 0.0,
            "corn_1x2_a": 0.0,
            "eh_plus_2": 0.0,
            "underdog": "",
            "match_id": "match-1",
        }
    ]
    gated = [
        {
            "league": "PL",
            "time": match_time,
            "match": "Home FC vs Away FC",
            "selection": "H U1.5",
            "market": "home_under_1_5",
            "probability": 0.90,
            "gate_score": 0.95,
            "match_id": "match-1",
        }
    ]

    prediction_module._render_output(preds, gated, stats={}, console=console)

    output = console.export_text()
    assert "High Conf: 1" in output
    assert "SUGGESTED SLIP" not in output
    assert "Suggested Slip" not in output
    assert "Best pick" not in output
    assert "Do not add extra legs" not in output

import json

import pandas as pd
from typer.testing import CliRunner

import src.monitoring.drift_orchestrator as drift_module
from src.cli.app import app
from src.evaluation.resolve_results import AuthoritativeResolver
from src.strategies.drift_guard import DriftGuardrail


runner = CliRunner()


def test_inspect_drift_reports_legacy_state(tmp_path, monkeypatch):
    status_file = tmp_path / "drift" / "rolling_90d_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(
        json.dumps(
            {
                "date": "2026-03-01",
                "status": "STOP",
                "alerts": ["CALIBRATION_DRIFT: 0.150 (Baseline 0.044)"],
                "metrics": {"ece": 0.15},
            }
        ),
        encoding="utf-8",
    )
    baseline_file = tmp_path / "models" / "drift_baselines.json"
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text(
        json.dumps(
            {
                "baseline_hit_rate": 0.75,
                "baseline_ece": 0.044,
                "baseline_mean_conf": 0.559,
                "baseline_selection_rate": 0.03,
                "training_date": None,
                "source": "training-artifact",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(DriftGuardrail, "STATUS_FILE", status_file)
    monkeypatch.setattr(DriftGuardrail, "BASELINE_FILE", baseline_file)

    result = runner.invoke(app, ["inspect-drift"])

    assert result.exit_code == 0
    assert "Legacy date field" in result.stdout
    assert "CALIBRATION_DRIFT" in result.stdout
    assert "Stop Boundary" in result.stdout


def test_check_drift_runs_with_loaded_outcomes(tmp_path, monkeypatch):
    outcomes_path = tmp_path / "prediction_outcomes.csv"
    recent_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
    pd.DataFrame(
        [
            {
                "prediction_id": "match1_HOME_WIN",
                "match_hash": "match1",
                "league": "PL",
                "kickoff_date": recent_date.isoformat(),
                "market": "HOME_WIN",
                "probability": 0.62,
                "outcome": "WON",
                "resolved_at": recent_date.isoformat(),
            },
            {
                "prediction_id": "match2_AWAY_WIN",
                "match_hash": "match2",
                "league": "PL",
                "kickoff_date": recent_date.isoformat(),
                "market": "AWAY_WIN",
                "probability": 0.58,
                "outcome": "LOST",
                "resolved_at": recent_date.isoformat(),
            },
        ]
    ).to_csv(outcomes_path, index=False)
    monkeypatch.setattr(AuthoritativeResolver, "DEFAULT_OUTCOMES_PATH", outcomes_path)

    result = runner.invoke(app, ["check-drift", "--league", "PL", "--lookback", "30"])

    assert result.exit_code == 0
    assert "Using 2 resolved predictions" in result.stdout
    assert ("No drift detected" in result.stdout) or ("Drift Alerts Detected" in result.stdout)


def test_check_drift_uses_persisted_state_when_no_recent_outcomes(tmp_path, monkeypatch):
    outcomes_path = tmp_path / "prediction_outcomes.csv"
    stale_date = pd.Timestamp("2020-01-01T12:00:00Z")
    pd.DataFrame(
        [
            {
                "prediction_id": "match1_HOME_WIN",
                "match_hash": "match1",
                "league": "PL",
                "kickoff_date": stale_date.isoformat(),
                "market": "HOME_WIN",
                "probability": 0.62,
                "outcome": "WON",
                "resolved_at": stale_date.isoformat(),
            }
        ]
    ).to_csv(outcomes_path, index=False)
    monkeypatch.setattr(AuthoritativeResolver, "DEFAULT_OUTCOMES_PATH", outcomes_path)

    class FakeDriftOrchestrator:
        GO = "GO"
        WATCH = "WATCH"
        STOP = "STOP"
        calls = []

        def __init__(self):
            self.global_alerts = []
            self._league_status = {}
            self._league_metrics = {}

        def evaluate_global_drift(self, current_session_data=None):
            self.__class__.calls.append(current_session_data)
            return self.GO

        def evaluate_league_drift(self, league, current_session_data=None):
            self.__class__.calls.append(current_session_data)
            return self.GO

    monkeypatch.setattr(drift_module, "DriftOrchestrator", FakeDriftOrchestrator)

    result = runner.invoke(app, ["check-drift", "--league", "PL", "--lookback", "30"])
    normalized_output = " ".join(result.stdout.split())

    assert result.exit_code == 0
    assert "No resolved predictions in the last 30 days" in result.stdout
    assert "drift status read from persisted state" in normalized_output
    assert "resolve-predictions" in result.stdout
    assert "Using" not in result.stdout
    assert FakeDriftOrchestrator.calls == [None]
    assert "No drift detected" in result.stdout

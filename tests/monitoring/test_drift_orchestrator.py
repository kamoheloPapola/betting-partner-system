import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db import connection as connection_module
from src.db.models import Base, DriftEvent
from src.monitoring.drift_orchestrator import DriftOrchestrator


def test_load_confidence_state_initializes_missing_file_to_go(tmp_path):
    status_file = tmp_path / "drift" / "rolling_90d_status.json"
    confidence_state_file = tmp_path / "drift" / "confidence_drift_state.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(
        json.dumps(
            {
                "date": "2026-04-13",
                "evaluated_at": "2026-04-13T00:00:00+00:00",
                "status": "GO",
                "alerts": [],
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )

    orchestrator = DriftOrchestrator(
        status_file=status_file,
        baseline_file=tmp_path / "models" / "drift_baselines.json",
        confidence_state_file=confidence_state_file,
    )

    expected = {
        "status": DriftOrchestrator.GO,
        "action": DriftOrchestrator.GO,
        "reason": "initialised",
    }

    assert orchestrator.confidence_state == expected
    assert confidence_state_file.exists()
    assert json.loads(confidence_state_file.read_text(encoding="utf-8")) == expected
    assert orchestrator.get_status("HOME_WIN") == DriftOrchestrator.GO


def test_league_state_file_uses_configured_status_directory(tmp_path):
    status_file = tmp_path / "isolated-drift" / "rolling_90d_status.json"
    orchestrator = DriftOrchestrator(
        status_file=status_file,
        baseline_file=tmp_path / "models" / "drift_baselines.json",
        confidence_state_file=tmp_path / "isolated-drift" / "confidence_drift_state.json",
    )

    orchestrator.evaluate_league_drift(
        "PL",
        {"hit_rate": 0.5, "ece": 0.48, "mean_conf": 0.6},
    )

    assert (status_file.parent / "PL_drift_status.json").exists()


def test_load_global_state_prefers_database_over_json_cache(tmp_path, monkeypatch):
    db_path = tmp_path / "drift_state.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    connection_module.get_engine.cache_clear()
    engine = connection_module.get_engine()
    Base.metadata.create_all(engine)

    status_file = tmp_path / "drift" / "rolling_90d_status.json"
    confidence_state_file = tmp_path / "drift" / "confidence_drift_state.json"

    try:
        orchestrator = DriftOrchestrator(
            status_file=status_file,
            baseline_file=tmp_path / "models" / "drift_baselines.json",
            confidence_state_file=confidence_state_file,
        )
        orchestrator.global_status = DriftOrchestrator.STOP
        orchestrator.global_alerts = ["CALIBRATION_DRIFT: 0.480 (Baseline 0.15)"]
        orchestrator.global_metrics = {"ece": 0.48}
        orchestrator.persist_global_state()

        status_file.write_text(
            json.dumps(
                {
                    "date": "2026-04-01",
                    "evaluated_at": "2026-04-01T00:00:00+00:00",
                    "status": "GO",
                    "alerts": [],
                    "metrics": {},
                }
            ),
            encoding="utf-8",
        )

        orchestrator.global_status = DriftOrchestrator.GO
        orchestrator.global_alerts = []
        orchestrator.global_metrics = {}
        orchestrator.load_global_state()

        assert orchestrator.global_status == DriftOrchestrator.STOP
        assert orchestrator.global_alerts == ["CALIBRATION_DRIFT: 0.480 (Baseline 0.15)"]
        assert orchestrator.global_metrics == {"ece": 0.48}
    finally:
        engine.dispose()
        connection_module.get_engine.cache_clear()


def test_append_drift_alerts_writes_csv_and_database_rows(tmp_path, monkeypatch):
    db_path = tmp_path / "drift_alerts.db"
    alerts_file = tmp_path / "monitoring" / "drift_alerts.csv"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setattr(DriftOrchestrator, "DEFAULT_ALERTS_FILE", alerts_file)
    connection_module.get_engine.cache_clear()
    engine = connection_module.get_engine()
    Base.metadata.create_all(engine)

    try:
        orchestrator = DriftOrchestrator(
            status_file=tmp_path / "drift" / "rolling_90d_status.json",
            baseline_file=tmp_path / "models" / "drift_baselines.json",
            confidence_state_file=tmp_path / "drift" / "confidence_drift_state.json",
        )
        orchestrator.append_drift_alerts(
            ["CALIBRATION_DRIFT: 0.480 (Baseline 0.15)"],
            league="PL",
            status=DriftOrchestrator.STOP,
        )

        lines = alerts_file.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "type,league,market,severity,metric,value,threshold,detected_at"
        assert "calibration_drift,PL,,CRITICAL,ece,0.48,0.15," in lines[1]

        with Session(connection_module.get_engine()) as session:
            row = session.execute(
                select(DriftEvent).where(DriftEvent.event_type == "calibration_drift")
            ).scalars().first()

        assert row is not None
        assert row.league == "PL"
        assert row.severity == "CRITICAL"
        assert row.metric == "ece"
        assert row.value == 0.48
        assert row.threshold == 0.15
    finally:
        engine.dispose()
        connection_module.get_engine.cache_clear()

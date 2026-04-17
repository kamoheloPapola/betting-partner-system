import json

from src.strategies.drift_guard import DriftGuardrail


def test_load_only_check_fails_closed_without_creating_status_file(tmp_path):
    status_file = tmp_path / "drift" / "status.json"
    guard = DriftGuardrail(status_file=status_file)

    status = guard.check_drift()

    assert status == "STOP"
    assert not status_file.exists()


def test_check_drift_with_metrics_persists_evaluation_timestamp(tmp_path):
    status_file = tmp_path / "drift" / "status.json"
    guard = DriftGuardrail(status_file=status_file)

    status = guard.check_drift({"hit_rate": 0.81, "ece": 0.02, "mean_conf": 0.56})
    payload = json.loads(status_file.read_text(encoding="utf-8"))

    assert status == "OK"
    assert payload["evaluated_at"]
    assert payload["date"] == payload["evaluated_at"][:10]


def test_guard_loads_baselines_from_metadata_file(tmp_path):
    baseline_file = tmp_path / "models" / "drift_baselines.json"
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text(
        json.dumps(
            {
                "baseline_hit_rate": 0.81,
                "baseline_ece": 0.031,
                "baseline_mean_conf": 0.541,
                "baseline_selection_rate": 0.02,
                "training_date": "2026-02-01T00:00:00Z",
                "source": "training-artifact",
            }
        ),
        encoding="utf-8",
    )

    guard = DriftGuardrail(
        status_file=tmp_path / "drift" / "status.json",
        baseline_file=baseline_file,
    )

    assert guard.BASELINES["hit_rate"] == 0.81
    assert guard.BASELINES["ece"] == 0.031
    assert guard.describe_baselines()["loaded_from_file"] is True
    assert guard.describe_baselines()["training_date"] == "2026-02-01T00:00:00Z"


def test_guard_loads_nested_baseline_metrics(tmp_path):
    baseline_file = tmp_path / "models" / "drift_baselines.json"
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text(
        json.dumps(
            {
                "training_date": "2026-03-01T00:00:00Z",
                "training_window_days": 365,
                "baseline_metrics": {
                    "hit_rate": 0.78,
                    "ece": 0.027,
                    "mean_confidence": 0.552,
                    "selection_rate": 0.025,
                },
                "source": "training-artifact",
            }
        ),
        encoding="utf-8",
    )

    guard = DriftGuardrail(
        status_file=tmp_path / "drift" / "status.json",
        baseline_file=baseline_file,
    )

    baselines = guard.describe_baselines()
    assert guard.BASELINES["hit_rate"] == 0.78
    assert guard.BASELINES["ece"] == 0.027
    assert guard.BASELINES["mean_conf"] == 0.552
    assert baselines["training_window_days"] == 365


def test_inspect_state_marks_legacy_date_as_ambiguous(tmp_path):
    status_file = tmp_path / "drift" / "status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(
        json.dumps(
            {
                "date": "2026-03-01",
                "status": "STOP",
                "alerts": ["HIT_RATE_DRIFT: 0.50 (Baseline 0.75)"],
                "metrics": {"hit_rate": 0.5},
            }
        ),
        encoding="utf-8",
    )

    state = DriftGuardrail(status_file=status_file).inspect_state()

    assert state["legacy_date"] == "2026-03-01"
    assert "last-access date" in state["note"]

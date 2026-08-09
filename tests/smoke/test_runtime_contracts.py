import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_predictor_module_imports():
    from src.predictions.predictor import Predictor

    assert Predictor.__name__ == "Predictor"


def _set_models_dir(env: dict[str, str], value: str | None) -> None:
    if value is None:
        env.pop("MODELS_DIR", None)
    else:
        env["MODELS_DIR"] = value


@pytest.mark.parametrize("models_dir_value", [None, ""])
def test_local_models_dir_falls_back_when_unset_or_blank(models_dir_value):
    env = os.environ.copy()
    _set_models_dir(env, models_dir_value)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os; "
                "from src.config import MODELS_DIR; "
                "print(repr(os.getenv('MODELS_DIR'))); "
                "print(MODELS_DIR)"
            ),
        ],
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    raw_value, resolved_path = result.stdout.splitlines()
    assert raw_value == repr(models_dir_value)
    assert Path(resolved_path) == REPO_ROOT / "src" / "ml" / "models"


@pytest.mark.parametrize("models_dir_value", [None, ""])
def test_cli_help_runs_without_models_dir_override(tmp_path, models_dir_value):
    log_file = tmp_path / "prediction_system.log"
    env = os.environ.copy()
    _set_models_dir(env, models_dir_value)
    env["PREDICTION_SYSTEM_LOG_FILE"] = str(log_file)

    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "--help"],
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=30,
    )

    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    combined_output = f"{stdout}\n{stderr}"
    assert result.returncode == 0, combined_output
    assert "show-predictions" in combined_output


def test_cli_import_does_not_create_models_directory(tmp_path):
    models_dir = tmp_path / "models"
    env = os.environ.copy()
    env["MODELS_DIR"] = str(models_dir)
    env["PREDICTION_SYSTEM_LOG_FILE"] = str(tmp_path / "prediction_system.log")

    result = subprocess.run(
        [sys.executable, "-c", "import src.cli.app"],
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert not models_dir.exists()


def test_calibrator_save_reports_directory_creation_failure(tmp_path, monkeypatch):
    from src.ml import calibration

    blocking_file = tmp_path / "models-file"
    blocking_file.write_text("not a directory", encoding="utf-8")
    calibrator_dir = blocking_file / "calibrators"
    monkeypatch.setattr(calibration, "CALIBRATOR_DIR", calibrator_dir)

    with pytest.raises(OSError) as exc_info:
        calibration.MarketCalibrator().save()

    message = str(exc_info.value)
    assert "Failed to create calibrator directory" in message
    assert str(calibrator_dir) in message
    assert exc_info.value.__cause__ is not None


def test_health_endpoint_returns_200():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "model_version" in payload

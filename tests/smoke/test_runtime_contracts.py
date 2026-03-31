import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_predictor_module_imports():
    from src.predictions.predictor import Predictor

    assert Predictor.__name__ == "Predictor"


def test_cli_help_runs(tmp_path):
    log_file = tmp_path / "prediction_system.log"
    env = os.environ.copy()
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


def test_health_endpoint_returns_200():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "model_version" in payload

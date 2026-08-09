import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from fastapi.testclient import TestClient

import src.api.main as api_main


app = api_main.app


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


def test_api_health_stays_live_with_empty_fixture_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "PROCESSED_DATA_DIR", tmp_path)

    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "data_fresh": False}


def test_api_health_reports_existing_recent_fixture_data(monkeypatch, tmp_path):
    matches_dir = tmp_path / "matches"
    matches_dir.mkdir()
    for league in api_main.DEFAULT_TRAINING_LEAGUES:
        (matches_dir / f"{league}_upcoming.csv").write_text(
            "fixture\n", encoding="utf-8"
        )
    monkeypatch.setattr(api_main, "PROCESSED_DATA_DIR", tmp_path)

    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "data_fresh": True}


def test_api_boot_ignores_fetch_failure_and_binds_health(tmp_path):
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    command_lines = [line for line in dockerfile.splitlines() if line.startswith("CMD ")]
    assert len(command_lines) == 1
    assert "fetch_fresh_data.py" not in command_lines[0]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved_socket:
        reserved_socket.bind(("127.0.0.1", 0))
        port = reserved_socket.getsockname()[1]

    env = os.environ.copy()
    env.pop("DATABASE_URL", None)
    env.pop("FOOTBALL_DATA_API_KEY", None)
    env["SKIP_MODEL_LOCK_CHECK"] = "true"
    env["PREDICTION_SYSTEM_LOG_FILE"] = str(tmp_path / "prediction_system.log")

    fetch_failure = subprocess.run(
        [sys.executable, "scripts/fetch_fresh_data.py"],
        capture_output=True,
        cwd=REPO_ROOT,
        env=env,
        timeout=15,
    )
    assert fetch_failure.returncode == 1

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    health_status = None
    payload = None
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.poll() is None:
            try:
                with urlopen(f"http://127.0.0.1:{port}/api/v1/health", timeout=1) as response:
                    health_status = response.status
                    payload = json.loads(response.read().decode("utf-8"))
                    break
            except (URLError, TimeoutError, ConnectionError):
                time.sleep(0.1)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)

    output = (
        process.stdout.read().decode("utf-8", errors="replace")
        if process.stdout
        else ""
    )
    assert payload is not None, output
    assert health_status == 200
    assert payload["status"] == "ok"
    assert "data_fresh" in payload


def test_fetch_remains_wired_to_existing_nightly_scheduler():
    workflow = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text(
        encoding="utf-8"
    )
    pipeline = (REPO_ROOT / "scripts" / "nightly_pipeline.py").read_text(
        encoding="utf-8"
    )

    assert "python scripts/nightly_pipeline.py" in workflow
    assert 'SCRIPTS_DIR / "fetch_fresh_data.py"' in pipeline
    assert '"fetch_fresh_data"' in pipeline
    assert '"fetch_fresh_data",' in pipeline.split("NON_FATAL_STEPS", 1)[1]

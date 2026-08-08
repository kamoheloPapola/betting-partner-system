from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_models.sh"


def _git_bash() -> str | None:
    candidates = [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    bash = shutil.which("bash")
    if bash and "system32" not in bash.lower():
        return bash
    return None


def test_curl_timeout_uses_configured_limits_and_discards_partial_file(tmp_path):
    bash = _git_bash()
    if bash is None:
        pytest.skip("A non-WSL bash runtime is required for the download script test")

    destination = tmp_path / "artifact.pkl"
    args_file = tmp_path / "curl-args.txt"
    env = os.environ.copy()
    env.update(
        {
            "ARTIFACT_CONNECT_TIMEOUT_SECONDS": "7",
            "ARTIFACT_TRANSFER_TIMEOUT_SECONDS": "19",
            "DOWNLOAD_SCRIPT": DOWNLOAD_SCRIPT.as_posix(),
            "DOWNLOAD_DESTINATION": destination.as_posix(),
            "FAKE_CURL_ARGS_FILE": args_file.as_posix(),
        }
    )

    result = subprocess.run(
        [
            bash,
            "-c",
            'source "$DOWNLOAD_SCRIPT"; set +e; '
            'curl() { printf "%s\\n" "$@" > "$FAKE_CURL_ARGS_FILE"; '
            'local output_path=""; while [ "$#" -gt 0 ]; do '
            'if [ "$1" = "--output" ]; then output_path="$2"; shift 2; '
            'else shift; fi; done; printf "partial-download" > "$output_path"; return 28; }; '
            'download_artifact '
            '"https://example.invalid/artifact.pkl" "$DOWNLOAD_DESTINATION"',
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
        timeout=15,
    )

    assert result.returncode == 28
    curl_args = args_file.read_text(encoding="utf-8").splitlines()
    assert curl_args[curl_args.index("--connect-timeout") + 1] == "7"
    assert curl_args[curl_args.index("--max-time") + 1] == "19"
    assert not destination.exists()
    assert list(tmp_path.glob("artifact.pkl.part.*")) == []
    assert "artifact_download_timeout" in result.stderr
    assert f"path={destination.as_posix()}" in result.stderr
    assert "connect_timeout_seconds=7" in result.stderr
    assert "max_time_seconds=19" in result.stderr


def _bash_env(**values):
    env = os.environ.copy()
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    env.update({key: str(value) for key, value in values.items()})
    return env


def _write_test_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "active_models": {"demo": "demo_v1.0.0"},
                "demo_v1.0.0": {"filename": "demo.pkl"},
            }
        ),
        encoding="utf-8",
    )


def test_manifest_readiness_requires_every_active_artifact_and_runtime_sidecar(tmp_path):
    bash = _git_bash()
    if bash is None:
        pytest.skip("A non-WSL bash runtime is required for the download script test")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    manifest = models_dir / "manifest.json"
    _write_test_manifest(manifest)
    env = _bash_env(
        DOWNLOAD_SCRIPT=DOWNLOAD_SCRIPT.as_posix(),
        MODELS_DIR=models_dir.as_posix(),
        MANIFEST_SRC=manifest.as_posix(),
    )

    incomplete = subprocess.run(
        [bash, "-c", 'source "$DOWNLOAD_SCRIPT"; models_ready'],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
        timeout=15,
    )

    assert incomplete.returncode == 1
    assert "model_readiness_failed reason=missing_files" in incomplete.stderr
    assert "demo.pkl" in incomplete.stderr

    for filename in ("demo.pkl", "feature_columns.json", "feature_baselines.json"):
        (models_dir / filename).write_text("ready", encoding="utf-8")

    complete = subprocess.run(
        [bash, "-c", 'source "$DOWNLOAD_SCRIPT"; models_ready'],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
        timeout=15,
    )

    assert complete.returncode == 0


def test_fresh_volume_downloads_manifest_derived_artifacts(tmp_path):
    bash = _git_bash()
    if bash is None:
        pytest.skip("A non-WSL bash runtime is required for the download script test")

    models_dir = tmp_path / "models"
    manifest_source = tmp_path / "release-manifest.json"
    _write_test_manifest(manifest_source)
    env = _bash_env(
        DOWNLOAD_SCRIPT=DOWNLOAD_SCRIPT.as_posix(),
        MODELS_DIR=models_dir.as_posix(),
        MANIFEST_SRC=manifest_source.as_posix(),
        MODEL_RELEASE_BASE_URL="https://example.invalid/models",
    )

    result = subprocess.run(
        [
            bash,
            "-c",
            'source "$DOWNLOAD_SCRIPT"; '
            'curl() { local output_path=""; while [ "$#" -gt 0 ]; do '
            'if [ "$1" = "--output" ]; then output_path="$2"; shift 2; '
            'else shift; fi; done; printf "downloaded" > "$output_path"; }; '
            "main",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert (models_dir / "manifest.json").exists()
    assert (models_dir / "demo.pkl").read_text(encoding="utf-8") == "downloaded"
    assert (models_dir / "feature_columns.json").exists()
    assert (models_dir / "feature_baselines.json").exists()
    assert not (models_dir / "goals_model.pkl").exists()
    assert "Manifest and all active model artifacts are ready." in result.stdout


def test_warm_restart_preserves_persistent_manifest(tmp_path):
    bash = _git_bash()
    if bash is None:
        pytest.skip("A non-WSL bash runtime is required for the download script test")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    persistent_manifest = models_dir / "manifest.json"
    _write_test_manifest(persistent_manifest)
    persistent_bytes = persistent_manifest.read_bytes()
    replacement_manifest = tmp_path / "bundled-manifest.json"
    replacement_manifest.write_text(
        json.dumps(
            {
                "active_models": {"replacement": "replacement_v1.0.0"},
                "replacement_v1.0.0": {"filename": "replacement.pkl"},
            }
        ),
        encoding="utf-8",
    )
    for filename in ("demo.pkl", "feature_columns.json", "feature_baselines.json"):
        (models_dir / filename).write_text("ready", encoding="utf-8")
    env = _bash_env(
        DOWNLOAD_SCRIPT=DOWNLOAD_SCRIPT.as_posix(),
        MODELS_DIR=models_dir.as_posix(),
        MANIFEST_SRC=replacement_manifest.as_posix(),
    )

    result = subprocess.run(
        [
            bash,
            "-c",
            'source "$DOWNLOAD_SCRIPT"; curl() { return 99; }; main',
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=PROJECT_ROOT,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert persistent_manifest.read_bytes() == persistent_bytes
    assert not (models_dir / "replacement.pkl").exists()

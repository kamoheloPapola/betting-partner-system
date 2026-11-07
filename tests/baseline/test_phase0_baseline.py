"""Regression guards for the historical FFPS Phase 0 baseline.

The mixed-market golden remains immutable pre-Phase-2 evidence. Its goal/1X2
subset is still a current regression contract; corrected corner/card behavior
is protected by the Phase 2 count baseline. These tests do not endorse quality.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = ROOT / "baselines" / "phase0"
MANIFEST_PATH = BASELINE_DIR / "baseline_manifest.json"
GOLDEN_PATH = BASELINE_DIR / "golden_predictions.json"
ACTIVE_MANIFEST_PATH = ROOT / "src" / "ml" / "models" / "manifest.json"
PHASE0_HISTORICAL_GOLDEN_SHA256 = "1491fc478932bba1205109ac3afd3a7b329d4b6b4e3f5986cac2892658f3438e"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_phase0_manifest_and_golden_exist() -> None:
    assert MANIFEST_PATH.is_file()
    assert GOLDEN_PATH.is_file()
    manifest = _load(MANIFEST_PATH)
    golden = _load(GOLDEN_PATH)
    assert manifest["schema_version"] == 1
    assert golden["schema_version"] == 1
    assert manifest["baseline_id"].startswith("ffps-phase0-")
    assert _sha256(GOLDEN_PATH) == PHASE0_HISTORICAL_GOLDEN_SHA256


def test_active_production_manifest_and_pointers_are_unchanged() -> None:
    baseline = _load(MANIFEST_PATH)
    current = _load(ACTIVE_MANIFEST_PATH)
    assert _sha256(ACTIVE_MANIFEST_PATH) == baseline["models"]["manifest"]["sha256"]
    assert current["active_models"] == baseline["models"]["active_pointers"]


def test_active_production_model_hashes_are_unchanged() -> None:
    baseline = _load(MANIFEST_PATH)
    for artifact in baseline["models"]["active_artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file(), artifact["path"]
        assert _sha256(path) == artifact["sha256"], artifact["slot"]


def test_critical_data_hashes_are_unchanged() -> None:
    baseline = _load(MANIFEST_PATH)
    for artifact in baseline["data"]["protected_artifacts"]:
        path = ROOT / artifact["path"]
        assert path.is_file(), artifact["path"]
        assert _sha256(path) == artifact["sha256"], artifact["path"]


def test_active_feature_order_and_bookmaker_boundary_are_unchanged() -> None:
    baseline = _load(MANIFEST_PATH)
    current = _load(ACTIVE_MANIFEST_PATH)
    bookmaker_markers = ("bookmaker", "odds", "implied_prob", "market_price")
    for artifact in baseline["models"]["active_artifacts"]:
        metadata = current[artifact["manifest_key"]]
        features = list(metadata.get("features") or [])
        encoded = json.dumps(
            features,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        assert hashlib.sha256(encoded).hexdigest() == artifact["feature_order_sha256"]
        assert not [
            feature for feature in features
            if any(marker in feature.lower() for marker in bookmaker_markers)
        ]


def test_seeded_offline_pl_goal_subset_reproduces(tmp_path) -> None:
    environment = os.environ.copy()
    environment["DATABASE_URL"] = ""
    bootstrap = (
        "import runpy,sys; from pathlib import Path; "
        "import src.features.pipeline as pipeline; "
        "pipeline.DATA_DIR=Path(sys.argv.pop(1)); "
        "runpy.run_path('scripts/phase0_baseline.py',run_name='__main__')"
    )
    result = subprocess.run(
        [sys.executable, "-c", bootstrap, str(tmp_path), "verify-goal-golden", "--quiet"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

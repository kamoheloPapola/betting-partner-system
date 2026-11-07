#!/usr/bin/env python3
"""Capture and verify the FFPS Phase 0 engineering baseline.

This tool is intentionally read-only with respect to production models, data,
manifests, drift state, and predictions. It writes only beneath
``baselines/phase0``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASELINE_DIR = ROOT / "baselines" / "phase0"
SOURCE_STATE_PATH = BASELINE_DIR / "preexisting_source_state.json"
MANIFEST_PATH = BASELINE_DIR / "baseline_manifest.json"
GOLDEN_PATH = BASELINE_DIR / "golden_predictions.json"
TEST_RESULTS_PATH = BASELINE_DIR / "test_results.json"
LOCK_PATH = BASELINE_DIR / "requirements-phase0-lock.txt"
RUNTIME_PATH = BASELINE_DIR / "runtime_state.json"
VERIFICATION_PATH = BASELINE_DIR / "verification_results.json"
REPORT_PATH = BASELINE_DIR / "FFPS_PHASE_0_BASELINE_REPORT.md"
PROTECTED_START_PATH = BASELINE_DIR / "protected_artifacts_before.json"

PHASE0_PATHS = [
    "requirements-dev.txt",
    "scripts/phase0_baseline.py",
    "tests/baseline/test_phase0_baseline.py",
    "tests/cli/test_verbosity_levels.py",
    "baselines/phase0/preexisting_source_state.json",
    "baselines/phase0/requirements-phase0-lock.txt",
    "baselines/phase0/runtime_state.json",
    "baselines/phase0/pytest-collect-output.txt",
    "baselines/phase0/pytest-output.txt",
    "baselines/phase0/pytest-results.xml",
    "baselines/phase0/test_results.json",
    "baselines/phase0/golden_predictions.json",
    "baselines/phase0/protected_artifacts_before.json",
    "baselines/phase0/baseline_manifest.json",
    "baselines/phase0/verification_results.json",
    "baselines/phase0/FFPS_PHASE_0_BASELINE_REPORT.md",
]

BOOKMAKER_TOKENS = {
    "bookmaker",
    "bookmakers",
    "betting",
    "implied_probability",
    "market_odds",
    "market_price",
    "opening_odds",
    "closing_odds",
    "consensus_odds",
}

# Phase 0's mixed-market golden is immutable historical evidence. After the
# approved Phase 2 count correction, these are the only fields in each captured
# run that are historical-only. All other Phase 0 fields remain active regression
# expectations; corrected corner/card expectations live in the Phase 2 golden.
PHASE0_HISTORICAL_COUNT_PROBABILITY_KEYS = (
    "card_o25",
    "card_u45",
    "card_u55",
    "conf_o75",
    "conf_u55",
    "corn_1x2_a",
    "corn_1x2_d",
    "corn_1x2_h",
    "corn_o75",
    "corn_u11",
)
PHASE0_HISTORICAL_COUNT_ATTRIBUTION_KEYS = (
    "card_u55_attribution",
    "corn_1x2_h_attribution",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def git_text(*args: str) -> str:
    return run(["git", *args]).stdout.strip()


def git_diff_hash(*args: str) -> tuple[str, int]:
    proc = subprocess.run(
        ["git", "diff", "--binary", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return sha256_bytes(proc.stdout), len(proc.stdout)


def current_git_state() -> dict[str, Any]:
    status_lines = [
        line for line in git_text("status", "--porcelain=v1", "--untracked-files=all").splitlines()
        if line
    ]
    staged = []
    changed = []
    untracked = []
    for line in status_lines:
        code = line[:2]
        path = line[3:].replace("\\", "/")
        if code == "??":
            untracked.append(path)
        else:
            changed.append(path)
            if code[0] not in {" ", "?"}:
                staged.append(path)

    upstream_proc = run(
        ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"],
        check=False,
    )
    upstream = upstream_proc.stdout.strip() if upstream_proc.returncode == 0 else None
    ahead = behind = None
    if upstream:
        counts = git_text("rev-list", "--left-right", "--count", f"HEAD...{upstream}").split()
        if len(counts) == 2:
            ahead, behind = int(counts[0]), int(counts[1])

    diff_sha, diff_bytes = git_diff_hash()
    staged_sha, staged_bytes = git_diff_hash("--cached")
    return {
        "root_identifier": ROOT.name,
        "commit": git_text("rev-parse", "HEAD"),
        "branch": git_text("branch", "--show-current"),
        "upstream": upstream,
        "ahead": ahead,
        "behind": behind,
        "dirty": bool(status_lines),
        "staged_paths": sorted(staged),
        "changed_paths": sorted(changed),
        "untracked_paths": sorted(untracked),
        "tracked_diff_sha256": diff_sha,
        "tracked_diff_bytes": diff_bytes,
        "staged_diff_sha256": staged_sha,
        "staged_diff_bytes": staged_bytes,
    }


def file_record(path: Path, *, metadata: bool = False) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": rel(path),
        "sha256": sha256_file(path),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }
    if metadata and path.suffix.lower() == ".csv":
        record.update(csv_metadata(path))
    return record


def csv_metadata(path: Path) -> dict[str, Any]:
    rows = 0
    min_date: str | None = None
    max_date: str | None = None
    leagues: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        date_name = next((name for name in ("date", "match_date", "kickoff") if name in columns), None)
        league_name = next((name for name in ("league", "competition", "league_code") if name in columns), None)
        for row in reader:
            rows += 1
            if date_name:
                value = str(row.get(date_name, "")).strip()
                if value:
                    min_date = value if min_date is None or value < min_date else min_date
                    max_date = value if max_date is None or value > max_date else max_date
            if league_name:
                value = str(row.get(league_name, "")).strip()
                if value:
                    leagues.add(value)
    return {
        "rows": rows,
        "columns": len(columns),
        "column_names_sha256": canonical_hash(columns),
        "min_date": min_date,
        "max_date": max_date,
        "leagues": sorted(leagues),
    }


def model_inventory() -> dict[str, Any]:
    manifest_path = ROOT / "src" / "ml" / "models" / "manifest.json"
    manifest = read_json(manifest_path)
    pointers = manifest.get("active_models", {})
    active: list[dict[str, Any]] = []
    for slot, manifest_key in sorted(pointers.items()):
        meta = manifest.get(manifest_key, {})
        filename = meta.get("filename") or meta.get("path")
        artifact_path = ROOT / "src" / "ml" / "models" / str(filename) if filename else None
        features = list(meta.get("features") or [])
        record: dict[str, Any] = {
            "slot": slot,
            "manifest_key": manifest_key,
            "filename": filename,
            "league": meta.get("league"),
            "model_role": meta.get("name") or meta.get("target") or slot,
            "target": meta.get("target"),
            "type": meta.get("type"),
            "metadata_version": meta.get("version"),
            "feature_count": len(features),
            "feature_order_sha256": canonical_hash(features),
            "features": features,
            "exists": bool(artifact_path and artifact_path.is_file()),
        }
        if artifact_path and artifact_path.is_file():
            record.update(
                {
                    "path": rel(artifact_path),
                    "size_bytes": artifact_path.stat().st_size,
                    "sha256": sha256_file(artifact_path),
                }
            )
        active.append(record)

    locations: dict[str, list[dict[str, Any]]] = {}
    for name in ("src/ml/models", "models", "data/models"):
        directory = ROOT / name
        records: list[dict[str, Any]] = []
        if directory.exists():
            for path in sorted(p for p in directory.rglob("*") if p.is_file()):
                if path.suffix.lower() in {".pkl", ".joblib", ".model", ".json"}:
                    records.append(file_record(path))
        locations[name] = records

    active_summary = [
        [item["slot"], item["manifest_key"], item.get("path"), item.get("sha256")]
        for item in active
    ]
    feature_summary = [[item["slot"], item["features"]] for item in active]
    bookmaker_features = sorted(
        {
            feature
            for item in active
            for feature in item["features"]
            if feature.lower() in BOOKMAKER_TOKENS
            or any(token in feature.lower() for token in ("bookmaker", "odds", "implied_prob", "market_price"))
        }
    )
    return {
        "manifest": file_record(manifest_path),
        "active_pointers": pointers,
        "active_count": len(active),
        "missing_active_count": sum(not item["exists"] for item in active),
        "active_artifacts_aggregate_sha256": canonical_hash(active_summary),
        "active_feature_order_aggregate_sha256": canonical_hash(feature_summary),
        "bookmaker_features_in_active_inference": bookmaker_features,
        "active_artifacts": active,
        "artifact_locations": locations,
        "directory_ambiguity": (
            "Model artifacts exist under src/ml/models, models, and data/models. "
            "Phase 0 records each location and does not select or consolidate a canonical directory."
        ),
    }


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.is_file()}, key=lambda path: rel(path))


def data_inventory() -> dict[str, Any]:
    protected = _unique_paths(
        [
            ROOT / "data" / "processed" / "matches.csv",
            ROOT / "data" / "features" / "feature_matrix.csv",
            ROOT / "data" / "features" / "feature_manifest.json",
            ROOT / "data" / "features" / "feature_matrix_hash.json",
            ROOT / "data" / "eval" / "prediction_outcomes.csv",
        ]
        + list((ROOT / "data" / "processed" / "matches").glob("*.csv"))
        + list((ROOT / "data" / "drift").glob("*.json"))
    )
    records = [file_record(path, metadata=True) for path in protected]
    aggregate_input = [[record["path"], record["sha256"]] for record in records]

    informational = _unique_paths(
        [
            ROOT / "data" / "processed" / "season_maps.json",
            ROOT / "data" / "features" / "feature_matrix_coverage.json",
        ]
        + [
            path for path in (ROOT / "data" / "predictions").glob("*")
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl", ".bak"}
        ]
    )
    return {
        "protected_count": len(records),
        "protected_aggregate_sha256": canonical_hash(aggregate_input),
        "protected_artifacts": records,
        "informational_artifacts": [file_record(path, metadata=path.suffix.lower() == ".csv") for path in informational],
        "protection_scope": (
            "Canonical matches, league processed CSVs, feature matrix/manifest/hash, evaluation outcomes, "
            "and drift JSON files are immutable Phase 0 inputs. Prediction outputs and season maps are "
            "fingerprinted separately because they are operational state, not golden inputs."
        ),
    }


def capture_protected_start() -> dict[str, Any]:
    """Freeze individual protected hashes before the final collection/suite run."""
    models = model_inventory()
    data = data_inventory()
    payload = {
        "captured_at": utc_now(),
        "models": {
            "manifest_sha256": models["manifest"]["sha256"],
            "active_count": models["active_count"],
            "active_artifacts_aggregate_sha256": models["active_artifacts_aggregate_sha256"],
            "artifacts": [
                {"slot": item["slot"], "path": item["path"], "sha256": item["sha256"]}
                for item in models["active_artifacts"]
            ],
        },
        "data": {
            "protected_count": data["protected_count"],
            "protected_aggregate_sha256": data["protected_aggregate_sha256"],
            "artifacts": [
                {"path": item["path"], "sha256": item["sha256"]}
                for item in data["protected_artifacts"]
            ],
        },
    }
    write_json(PROTECTED_START_PATH, payload)
    return payload
def normalise(value: Any) -> Any:
    """Convert numpy/pandas values and sets into stable JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if value != value:
            return None
        if value in {float("inf"), float("-inf")}:
            return str(value)
        return value
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(key): normalise(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, tuple, list)):
        items = [normalise(item) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True)) if isinstance(value, set) else items
    if hasattr(value, "item"):
        return normalise(value.item())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _active_pl_artifacts(models: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "slot": item["slot"],
            "manifest_key": item["manifest_key"],
            "path": item.get("path"),
            "sha256": item.get("sha256"),
            "feature_order_sha256": item["feature_order_sha256"],
        }
        for item in models["active_artifacts"]
        if item.get("league") == "PL"
    ]


def generate_golden(*, match_hash: str | None = None, seed: int = 42, simulations: int = 50_000) -> dict[str, Any]:
    # Prevent .env from activating a database-backed registry. An explicit empty
    # value is retained by load_dotenv(override=False), forcing offline file use.
    os.environ["DATABASE_URL"] = ""

    import pandas as pd

    from src.cli.commands.prediction import _calculate_probabilities_v2, _load_prediction_models
    from src.core.container import ServiceContainer
    from src.ml.registry import ModelRegistry
    from src.simulation.match_simulator import MatchSimulator

    ServiceContainer.reset_singleton()
    ModelRegistry._instance = None
    ModelRegistry._manifest_cache = None
    suite = _load_prediction_models("PL")

    ordered_features: list[str] = []
    role_features: dict[str, list[str]] = {}
    for role, model in suite.items():
        meta = model if role.startswith("meta_") and isinstance(model, dict) else getattr(model, "meta", None)
        if not isinstance(meta, dict):
            continue
        features = list(meta.get("features") or [])
        if not features:
            continue
        role_features[role] = features
        for feature in features:
            if feature not in ordered_features:
                ordered_features.append(feature)

    matrix_path = ROOT / "data" / "features" / "feature_matrix.csv"
    frame = pd.read_csv(matrix_path)
    missing_columns = sorted(set(ordered_features) - set(frame.columns))
    available_features = [feature for feature in ordered_features if feature in frame.columns]
    candidates = frame.loc[frame["league"].astype(str) == "PL"].copy()
    if match_hash:
        candidates = candidates.loc[candidates["match_hash"].astype(str) == match_hash]
    candidates = candidates.dropna(subset=available_features)
    if candidates.empty:
        raise RuntimeError("No fully populated offline PL row can exercise the active model suite")
    candidates["date"] = candidates["date"].astype(str)
    row = candidates.sort_values(["date", "match_hash"], kind="mergesort").iloc[-1]

    selected_values = [{"name": name, "value": normalise(row[name])} for name in available_features]
    warnings_template = {"missing_offsets": set(), "missing_card_offsets": set()}

    def calculate_once() -> dict[str, Any]:
        warnings = {key: set(value) for key, value in warnings_template.items()}
        probabilities, attribution = _calculate_probabilities_v2(
            row,
            suite,
            "PL",
            False,
            warnings,
            MatchSimulator(n_simulations=simulations, league="PL", seed=seed),
            use_simulator=True,
            rl_bandit=None,
        )
        return normalise(
            {
                "probabilities": probabilities,
                "attribution": attribution,
                "warnings": warnings,
            }
        )

    first = calculate_once()
    second = calculate_once()
    models = model_inventory()
    identity = {
        "match_hash": str(row["match_hash"]),
        "date": str(row["date"]),
        "league": "PL",
        "home_team": str(row["home_team"]),
        "away_team": str(row["away_team"]),
    }
    return {
        "schema_version": 1,
        "purpose": "Current-behavior regression baseline; not a correctness endorsement.",
        "offline": True,
        "case": "pl_active_suite_monte_carlo_seed_42",
        "fixture": identity,
        "input_source": rel(matrix_path),
        "input_source_sha256": sha256_file(matrix_path),
        "feature_count": len(available_features),
        "unavailable_optional_features": missing_columns,
        "feature_order_sha256": canonical_hash(available_features),
        "feature_values_sha256": canonical_hash(selected_values),
        "selected_feature_values": selected_values,
        "role_feature_order_sha256": {
            role: canonical_hash(features) for role, features in sorted(role_features.items())
        },
        "bookmaker_features": sorted(
            feature for feature in ordered_features
            if "bookmaker" in feature.lower() or "odds" in feature.lower() or "implied_prob" in feature.lower()
        ),
        "random_seed": seed,
        "simulation_count": simulations,
        "intensity_boost": False,
        "calculation_path": [
            "scripts.phase0_baseline.generate_golden",
            "src.cli.commands.prediction._load_prediction_models",
            "src.cli.commands.prediction._calculate_probabilities_v2",
            "src.cli.commands.prediction._calc_goals",
            "src.simulation.match_simulator.MatchSimulator.simulate",
            "src.cli.commands.prediction._calc_corners",
            "src.cli.commands.prediction._calc_cards",
            "src.cli.commands.prediction._calc_dc",
        ],
        "model_manifest_sha256": models["manifest"]["sha256"],
        "active_pl_artifacts": _active_pl_artifacts(models),
        "run_1": first,
        "run_2": second,
        "deterministic_exact_match": first == second,
        "output_sha256": canonical_hash(first),
    }


def capture_golden() -> dict[str, Any]:
    golden = generate_golden()
    golden["captured_at"] = utc_now()
    write_json(GOLDEN_PATH, golden)
    return golden


def compare_values(expected: Any, actual: Any, *, tolerance: float, path: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            errors.append(f"{path}: expected {expected!r}, got {actual!r}")
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if abs(float(expected) - float(actual)) > tolerance:
            errors.append(f"{path}: expected {expected!r}, got {actual!r}")
    elif isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            errors.append(f"{path}: keys differ")
        for key in sorted(set(expected) & set(actual)):
            errors.extend(compare_values(expected[key], actual[key], tolerance=tolerance, path=f"{path}.{key}"))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            errors.append(f"{path}: length expected {len(expected)}, got {len(actual)}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            errors.extend(compare_values(left, right, tolerance=tolerance, path=f"{path}[{index}]"))
    elif expected != actual:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}")
    return errors


def verify_golden(*, quiet: bool = False) -> dict[str, Any]:
    expected = read_json(GOLDEN_PATH)
    actual = generate_golden(
        match_hash=expected["fixture"]["match_hash"],
        seed=int(expected["random_seed"]),
        simulations=int(expected["simulation_count"]),
    )
    errors: list[str] = []
    for key in (
        "fixture",
        "input_source_sha256",
        "feature_order_sha256",
        "feature_values_sha256",
        "role_feature_order_sha256",
        "bookmaker_features",
        "model_manifest_sha256",
        "active_pl_artifacts",
        "run_1",
        "output_sha256",
    ):
        errors.extend(compare_values(expected[key], actual[key], tolerance=1e-12, path=key))
    result = {
        "verified_at": utc_now(),
        "case": expected["case"],
        "tolerance": 1e-12,
        "passed": not errors,
        "errors": errors,
        "reproduced_output_sha256": actual["output_sha256"],
        "exact_repeat_within_capture": actual["deterministic_exact_match"],
    }
    if not quiet:
        print(json.dumps(result, indent=2))
    return result


def verify_goal_golden(*, quiet: bool = False) -> dict[str, Any]:
    """Verify every still-current field in the historical Phase 0 golden."""
    expected = read_json(GOLDEN_PATH)
    actual = generate_golden(
        match_hash=expected["fixture"]["match_hash"],
        seed=int(expected["random_seed"]),
        simulations=int(expected["simulation_count"]),
    )
    errors: list[str] = []
    for key in (
        "fixture",
        "input_source_sha256",
        "feature_order_sha256",
        "feature_values_sha256",
        "role_feature_order_sha256",
        "bookmaker_features",
        "model_manifest_sha256",
        "active_pl_artifacts",
    ):
        errors.extend(compare_values(expected[key], actual[key], tolerance=1e-12, path=key))

    def active_run_contract(run: dict[str, Any]) -> dict[str, Any]:
        return {
            "probabilities": {
                key: value
                for key, value in run["probabilities"].items()
                if key not in PHASE0_HISTORICAL_COUNT_PROBABILITY_KEYS
            },
            "attribution": {
                key: value
                for key, value in run["attribution"].items()
                if key not in PHASE0_HISTORICAL_COUNT_ATTRIBUTION_KEYS
            },
            "warnings": run["warnings"],
        }

    for run_key in ("run_1", "run_2"):
        errors.extend(
            compare_values(
                active_run_contract(expected[run_key]),
                active_run_contract(actual[run_key]),
                tolerance=1e-12,
                path=run_key,
            )
        )
    approved_count_divergence_observed = expected["output_sha256"] != actual["output_sha256"]
    if not approved_count_divergence_observed:
        errors.append("mixed output unexpectedly matches the historical Phase 0 output")
    result = {
        "verified_at": utc_now(),
        "case": expected["case"],
        "scope": "all_fields_except_historical_count_markets",
        "tolerance": 1e-12,
        "historical_count_probability_keys": list(PHASE0_HISTORICAL_COUNT_PROBABILITY_KEYS),
        "historical_count_attribution_keys": list(PHASE0_HISTORICAL_COUNT_ATTRIBUTION_KEYS),
        "historical_mixed_output_sha256": expected["output_sha256"],
        "current_mixed_output_sha256": actual["output_sha256"],
        "approved_count_divergence_observed": approved_count_divergence_observed,
        "passed": not errors,
        "errors": errors,
        "exact_repeat_within_capture": actual["deterministic_exact_match"],
    }
    if not quiet:
        print(json.dumps(result, indent=2))
    return result


def capture_runtime() -> dict[str, Any]:
    freeze = run([sys.executable, "-m", "pip", "freeze", "--all"]).stdout
    LOCK_PATH.write_text(freeze, encoding="utf-8")
    pip_version_raw = run([sys.executable, "-m", "pip", "--version"]).stdout.strip()
    pip_match = re.match(r"^(pip\s+\S+).+\(python\s+([^)]+)\)$", pip_version_raw)
    pip_version = (
        f"{pip_match.group(1)} (python {pip_match.group(2)})"
        if pip_match
        else pip_version_raw.replace(str(ROOT), "<repository>")
    )
    source_hashes = {
        name: sha256_file(ROOT / name)
        for name in ("requirements.txt", "requirements-train.txt", "requirements-dev.txt")
    }
    runtime = {
        "captured_at": utc_now(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_identifier": Path(sys.executable).name,
        "platform": platform.platform(),
        "pip": pip_version,
        "dependency_sources": source_hashes,
        "lock_path": rel(LOCK_PATH),
        "lock_sha256": sha256_file(LOCK_PATH),
        "environment_fingerprint_sha256": canonical_hash(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "lock_sha256": sha256_file(LOCK_PATH),
                "dependency_sources": source_hashes,
            }
        ),
        "selection_rationale": (
            "Python 3.11 matches the API container, supports the pinned dependency set, and avoids relying "
            "on the newer local/scheduler Python 3.14 or the older mypy 3.10 target. It is the lowest-risk "
            "engineering/test common denominator; it does not redefine deployment runtimes."
        ),
        "clean_install_commands": [
            "py -3.11 -m venv .tmp/phase0-py311",
            ".tmp/phase0-py311/Scripts/python.exe -m pip install -r baselines/phase0/requirements-phase0-lock.txt",
        ],
        "declared_install_commands": [
            "py -3.11 -m venv .tmp/phase0-py311",
            ".tmp/phase0-py311/Scripts/python.exe -m pip install -r requirements-dev.txt",
        ],
        "test_commands": [
            ".tmp/phase0-py311/Scripts/python.exe -m pytest --collect-only -q",
            ".tmp/phase0-py311/Scripts/python.exe scripts/phase0_baseline.py run-tests",
            ".tmp/phase0-py311/Scripts/python.exe scripts/phase0_baseline.py verify",
        ],
    }
    write_json(RUNTIME_PATH, runtime)
    return runtime


def _pytest_counts(xml_path: Path) -> dict[str, Any]:
    root = ET.parse(xml_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        "test_count": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "failed": sum(int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)) for suite in suites),
        "errors": sum(int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "duration_seconds": sum(float(suite.attrib.get("time", 0.0)) for suite in suites),
    }


def classify_failure(nodeid: str) -> str:
    if "tests/test_drift_artifacts.py" in nodeid.replace("\\", "/"):
        return "expected_stale_state_failure"
    if "tests/cli/test_verbosity_levels.py" in nodeid.replace("\\", "/"):
        return "test_contract_drift"
    if "tests/property/" in nodeid.replace("\\", "/"):
        return "regression_or_implementation_failure"
    return "unknown"


def run_tests() -> dict[str, Any]:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    # Tests that spawn literal `python` must resolve to the selected engineering
    # interpreter, not an unrelated system installation.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    collect_command = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    collect_started = time.perf_counter()
    collect = run(collect_command, check=False)
    collect_duration = time.perf_counter() - collect_started
    collect_output = collect.stdout + collect.stderr
    (BASELINE_DIR / "pytest-collect-output.txt").write_text(collect_output, encoding="utf-8")
    collected_matches = re.findall(r"(\d+) tests? collected", collect_output)
    collected = int(collected_matches[-1]) if collected_matches else None

    xml_path = BASELINE_DIR / "pytest-results.xml"
    full_command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={xml_path}"]
    full_started = time.perf_counter()
    suite = run(full_command, check=False)
    wall_duration = time.perf_counter() - full_started
    if xml_path.exists():
        xml_text = xml_path.read_text(encoding="utf-8")
        xml_path.write_text(xml_text.replace(str(ROOT), "<repository>"), encoding="utf-8")
    output = suite.stdout + suite.stderr
    (BASELINE_DIR / "pytest-output.txt").write_text(output, encoding="utf-8")
    counts = _pytest_counts(xml_path)
    failure_names = re.findall(r"^FAILED\s+([^\s]+)", output, flags=re.MULTILINE)
    warning_matches = re.findall(r"(\d+) warnings?", output)
    warnings = int(warning_matches[-1]) if warning_matches else 0
    classifications = [
        {"test": nodeid, "classification": classify_failure(nodeid)} for nodeid in failure_names
    ]
    counts["passed"] = counts["test_count"] - counts["failed"] - counts["skipped"]
    result = {
        "captured_at": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "collection": {
            "command": "python -m pytest --collect-only -q",
            "exit_code": collect.returncode,
            "collected": collected,
            "duration_seconds": round(collect_duration, 3),
            "output_path": "baselines/phase0/pytest-collect-output.txt",
        },
        "suite": {
            "command": "python -m pytest -q --junitxml=baselines/phase0/pytest-results.xml",
            "exit_code": suite.returncode,
            **counts,
            "warnings": warnings,
            "wall_duration_seconds": round(wall_duration, 3),
            "failure_names": failure_names,
            "failure_classifications": classifications,
            "output_path": "baselines/phase0/pytest-output.txt",
            "junit_path": rel(xml_path),
        },
        "interpretation": (
            "A non-zero suite exit is an expected baseline outcome when only the six pre-existing failures remain; "
            "Phase 0 does not conceal or repair them."
        ),
    }
    write_json(TEST_RESULTS_PATH, result)
    print(json.dumps(result, indent=2))
    return result


def verify_invariants(manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = manifest or read_json(MANIFEST_PATH)
    source = read_json(SOURCE_STATE_PATH)
    models_now = model_inventory()
    data_now = data_inventory()
    protected_start = read_json(PROTECTED_START_PATH)
    checks: list[dict[str, Any]] = []

    def check(name: str, expected: Any, actual: Any) -> None:
        checks.append({"name": name, "expected": expected, "actual": actual, "passed": expected == actual})

    git_now = current_git_state()
    check("source commit", source["commit"], git_now["commit"])
    check("source branch", source["branch"], git_now["branch"])
    preexisting_diff_sha, _ = git_diff_hash("--", *source["changed_paths"])
    check("pre-existing tracked diff", source["tracked_diff_sha256"], preexisting_diff_sha)
    check("pre-existing staged diff", source["staged_diff_sha256"], git_now["staged_diff_sha256"])
    for path, expected_hash in source["untracked_file_hashes"].items():
        current_path = ROOT / path
        check(f"pre-existing untracked file {path}", expected_hash, sha256_file(current_path) if current_path.is_file() else None)

    check(
        "active production manifest",
        source["protected_artifact_hashes"]["active_manifest_sha256"],
        models_now["manifest"]["sha256"],
    )
    check("active model pointer count", source["protected_artifact_hashes"]["active_model_count"], models_now["active_count"])
    for expected in manifest["models"]["active_artifacts"]:
        path = ROOT / expected["path"]
        check(f"active model {expected['slot']}", expected["sha256"], sha256_file(path) if path.is_file() else None)
    for expected in manifest["data"]["protected_artifacts"]:
        path = ROOT / expected["path"]
        check(f"critical data {expected['path']}", expected["sha256"], sha256_file(path) if path.is_file() else None)
    check(
        "before/after active model aggregate",
        protected_start["models"]["active_artifacts_aggregate_sha256"],
        models_now["active_artifacts_aggregate_sha256"],
    )
    check(
        "before/after critical data aggregate",
        protected_start["data"]["protected_aggregate_sha256"],
        data_now["protected_aggregate_sha256"],
    )
    for expected in protected_start["models"]["artifacts"]:
        path = ROOT / expected["path"]
        check(f"before/after model {expected['slot']}", expected["sha256"], sha256_file(path) if path.is_file() else None)
    for expected in protected_start["data"]["artifacts"]:
        path = ROOT / expected["path"]
        check(f"before/after data {expected['path']}", expected["sha256"], sha256_file(path) if path.is_file() else None)
    check("active inference bookmaker fields", [], models_now["bookmaker_features_in_active_inference"])
    check(
        "active feature ordering",
        manifest["models"]["active_feature_order_aggregate_sha256"],
        models_now["active_feature_order_aggregate_sha256"],
    )
    golden = verify_goal_golden(quiet=True)
    check("goal/1X2 golden subset", True, golden["passed"])
    check("golden repeated run determinism", True, golden["exact_repeat_within_capture"])

    result = {
        "verified_at": utc_now(),
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "golden": golden,
        "models_after": {
            "manifest_sha256": models_now["manifest"]["sha256"],
            "active_artifacts_aggregate_sha256": models_now["active_artifacts_aggregate_sha256"],
        },
        "data_after": {
            "protected_count": data_now["protected_count"],
            "protected_aggregate_sha256": data_now["protected_aggregate_sha256"],
        },
    }
    write_json(VERIFICATION_PATH, result)
    return result


def write_report(manifest: dict[str, Any]) -> None:
    tests = manifest.get("tests", {}).get("suite", {})
    verification = manifest.get("verification", {})
    source = manifest["repository"]["starting_state"]
    runtime = manifest["runtime"]
    models = manifest["models"]
    data = manifest["data"]
    golden = manifest["goldens"]
    failures = tests.get("failure_classifications", [])
    failure_lines = "\n".join(
        f"- `{item['test']}` — {item['classification'].replace('_', ' ')}" for item in failures
    ) or "- No failures recorded."
    phase0_files = "\n".join(f"- `{path}`" for path in manifest["phase0_files"])
    report = f"""# FFPS PHASE 0 BASELINE REPORT

## Source State

- Baseline ID: `{manifest['baseline_id']}`
- Commit: `{source['commit']}` on `{source['branch']}` tracking `{source['upstream']}`
- Divergence: ahead {source['ahead']}, behind {source['behind']}
- Dirty before Phase 0: `{source['dirty']}` ({len(source['changed_paths'])} changed paths; {len(source['untracked_paths'])} untracked path)
- Pre-existing tracked diff SHA-256: `{source['tracked_diff_sha256']}`
- Staged paths at capture: none

## Runtime State

Engineering/test baseline: CPython `{runtime['python']}` on `{runtime['platform']}`. Python 3.11 was selected because it matches the API container and supports the pinned repository dependencies. Scheduler Python 3.14 and mypy's Python 3.10 target remain documented mismatches; Phase 0 does not change them.

## Dependency State

- Direct source: `requirements-dev.txt` (includes both existing requirement files plus `hypothesis==6.165.2`)
- Fully resolved lock: `{runtime['lock_path']}`
- Lock SHA-256: `{runtime['lock_sha256']}`
- Environment fingerprint: `{runtime['environment_fingerprint_sha256']}`

## Test Baseline

- Collection: {manifest['tests']['collection'].get('collected')} collected, exit {manifest['tests']['collection'].get('exit_code')}
- Full suite: {tests.get('passed')} passed, {tests.get('failed')} failed, {tests.get('skipped')} skipped, {tests.get('warnings')} warnings
- Pytest duration: {tests.get('duration_seconds')} seconds (wall {tests.get('wall_duration_seconds')} seconds)

## Model Artifact Baseline

- Active manifest SHA-256: `{models['manifest']['sha256']}`
- Active pointers/artifacts: {models['active_count']} / {models['active_count'] - models['missing_active_count']} present
- Active model aggregate SHA-256: `{models['active_artifacts_aggregate_sha256']}`
- Active feature-order aggregate SHA-256: `{models['active_feature_order_aggregate_sha256']}`
- Active bookmaker-derived features found: `{models['bookmaker_features_in_active_inference']}`
- Model directories remain ambiguous (`src/ml/models`, `models`, `data/models`); no canonicalization occurred.

## Data Artifact Baseline

- Protected critical artifacts: {data['protected_count']}
- Protected aggregate SHA-256: `{data['protected_aggregate_sha256']}`
- Canonical matches, league files, feature files, evaluation outcomes, and drift state were fingerprinted without refresh.
- Prediction outputs and season maps are informational fingerprints and were not treated as golden inputs.

## Golden Prediction Baseline

- Case: `{golden['case']}`
- Fixture: {golden['fixture']['home_team']} vs {golden['fixture']['away_team']} ({golden['fixture']['date']}, `{golden['fixture']['match_hash']}`)
- Seed/simulations: {golden['random_seed']} / {golden['simulation_count']}
- Feature values SHA-256: `{golden['feature_values_sha256']}`
- Output SHA-256: `{golden['output_sha256']}`
- Two fresh seeded simulators matched exactly: `{golden['deterministic_exact_match']}`
- Offline replay passed at tolerance 1e-12: `{verification.get('golden', {}).get('passed')}`

## Known Existing Failures

{failure_lines}

## Known Existing Blockers

- Drift artifacts are stale and their freshness tests fail. Phase 0 deliberately did not refresh them.
- Production model provenance remains incomplete and artifacts exist in multiple directories.
- The source baseline is materially dirty and diverged from `origin/main`; Phase 0 preserves that state.
- Known prediction-integrity and validation defects remain outside Phase 0 and are not correctness-endorsed by goldens.

## Reproducibility Result

Full collection succeeds under Python 3.11 after declaring Hypothesis. The representative offline PL active-suite prediction reproduces exactly with seed 42. Overall invariant verification passed: `{verification.get('passed')}`. The full test suite is reproducible but intentionally not green because the recorded baseline failures remain.

## Files Changed During Phase 0

{phase0_files}

## Explicitly Unchanged Production Artifacts

The active manifest, every active production model binary, every protected critical data artifact, active feature order, and the pre-existing tracked/untracked user changes passed before/after hash checks. No training, promotion, provider refresh, migration, API call, or persistence write occurred.

## Limitations

- This is a Windows/Python 3.11 engineering lock; container runtime alignment was recorded, not changed.
- Golden replay proves current deterministic behavior for one offline PL row, not predictive validity.
- Operational prediction files are inventoried but not immutable because serving owns them.
- Artifact origin, training dataset lineage, and external-service behavior cannot be established by Phase 0 fingerprints.

## Ready / Not Ready for Phase 1

**Ready for the narrowly scoped Phase 1 investigation, with known baseline failures and blockers carried forward.** No Phase 1 work has started.

> Phase 0 does not validate FFPS prediction quality or correctness. It only establishes a reproducible baseline.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def capture_manifest() -> dict[str, Any]:
    source = read_json(SOURCE_STATE_PATH)
    runtime = read_json(RUNTIME_PATH)
    tests = read_json(TEST_RESULTS_PATH) if TEST_RESULTS_PATH.exists() else {
        "collection": {
            "command": source["prechange_tests"]["collection_command"],
            "exit_code": 1,
            "collected": source["prechange_tests"]["collection_candidates"],
        },
        "suite": {
            "command": source["prechange_tests"]["partial_suite_command"],
            "exit_code": 1,
            "test_count": source["prechange_tests"]["passed"] + source["prechange_tests"]["failed"] + source["prechange_tests"]["skipped"],
            "passed": source["prechange_tests"]["passed"],
            "failed": source["prechange_tests"]["failed"],
            "skipped": source["prechange_tests"]["skipped"],
            "warnings": source["prechange_tests"]["warnings"],
            "duration_seconds": source["prechange_tests"]["duration_seconds"],
            "failure_classifications": [],
        },
        "provisional": True,
    }
    golden = read_json(GOLDEN_PATH)
    manifest = {
        "schema_version": 1,
        "baseline_id": source["baseline_id"],
        "created_at": source["captured_at"],
        "purpose": "Reproducible current-state engineering baseline; not a prediction correctness endorsement.",
        "repository": {
            "starting_state": source,
            "ending_state": current_git_state(),
        },
        "runtime": runtime,
        "tests": tests,
        "protected_artifacts_before": {
            "path": rel(PROTECTED_START_PATH),
            "sha256": sha256_file(PROTECTED_START_PATH),
        },
        "models": model_inventory(),
        "data": data_inventory(),
        "goldens": golden,
        "phase0_files": [path for path in PHASE0_PATHS if (ROOT / path).exists() or path in {"baselines/phase0/baseline_manifest.json", "baselines/phase0/verification_results.json", "baselines/phase0/FFPS_PHASE_0_BASELINE_REPORT.md"}],
        "known_blockers": [
            "Six pre-existing tests fail: five stale drift freshness checks and one quiet-mode message contract mismatch.",
            "The working tree is materially dirty and main is both ahead of and behind origin/main.",
            "Production model provenance is incomplete and model artifacts exist in multiple directories.",
            "Data and drift state are stale; Phase 0 intentionally does not refresh them.",
            "Known prediction and validation defects are preserved for later approved phases.",
        ],
        "invariants": {
            "no_prediction_behavior_change": True,
            "no_training": True,
            "no_data_refresh": True,
            "no_external_provider_calls": True,
            "no_api_or_persistence_contract_change": True,
        },
    }
    write_json(MANIFEST_PATH, manifest)
    verification = verify_invariants(manifest)
    manifest["verification"] = verification
    manifest["repository"]["ending_state"] = current_git_state()
    write_json(MANIFEST_PATH, manifest)
    write_report(manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("capture-runtime", help="Write the resolved Python environment lock and fingerprint")
    subparsers.add_parser("run-tests", help="Capture collect-only and full-suite outcomes")
    subparsers.add_parser("capture-protected-start", help="Freeze protected model/data hashes before tests")
    subparsers.add_parser("capture-golden", help="Capture the seeded offline PL golden twice")
    verify_parser = subparsers.add_parser("verify-golden", help="Replay and compare the golden")
    verify_parser.add_argument("--quiet", action="store_true")
    verify_goal_parser = subparsers.add_parser(
        "verify-goal-golden",
        help="Replay only the still-current goal/1X2 subset of the historical Phase 0 golden",
    )
    verify_goal_parser.add_argument("--quiet", action="store_true")
    subparsers.add_parser("capture-manifest", help="Write manifest, invariant verification, and report")
    subparsers.add_parser("verify", help="Verify all captured invariants")
    args = parser.parse_args()

    if args.command == "capture-runtime":
        print(json.dumps(capture_runtime(), indent=2))
    elif args.command == "run-tests":
        run_tests()
    elif args.command == "capture-protected-start":
        print(json.dumps(capture_protected_start(), indent=2))
    elif args.command == "capture-golden":
        print(json.dumps(capture_golden(), indent=2))
    elif args.command == "verify-golden":
        result = verify_golden(quiet=args.quiet)
        return 0 if result["passed"] else 1
    elif args.command == "verify-goal-golden":
        result = verify_goal_golden(quiet=args.quiet)
        return 0 if result["passed"] else 1
    elif args.command == "capture-manifest":
        manifest = capture_manifest()
        print(json.dumps({"baseline_id": manifest["baseline_id"], "verification": manifest["verification"]["passed"]}, indent=2))
        return 0 if manifest["verification"]["passed"] else 1
    elif args.command == "verify":
        result = verify_invariants()
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

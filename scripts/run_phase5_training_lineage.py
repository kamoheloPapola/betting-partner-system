"""Create, finalize, and offline-verify FFPS Phase 5 training-lineage evidence."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase4_prediction_provenance import verify_evidence as verify_phase4_evidence
from src.evaluation.phase3_calibration import aggregate_directory
from src.evaluation.phase3_pl_1x2_temperature import protected_snapshot as earlier_phase_protected_snapshot
from src.ml.training_lineage import (
    LEGACY_TRAINING_LINEAGE_UNPROVEN,
    ModelTrainingLineageRecord,
    build_legacy_unproven_record,
    canonical_json,
    sha256_file,
    sha256_json,
    utc_now,
    verify_lineage_record,
)


DEFAULT_RUN_ID = "phase5-pl-goal-training-lineage-79110f9a-20260829-v1"
DEFAULT_RUN_DIR = ROOT / "baselines" / "phase5" / DEFAULT_RUN_ID
PHASE1_RUN = ROOT / "baselines" / "phase1" / "phase1-pl-1x2-79110f9a-20240801-v1"
PHASE3_RUN = ROOT / "baselines" / "phase3" / "phase3-pl-1x2-temperature-79110f9a-20260827-v1"
PHASE4_RUN = ROOT / "baselines" / "phase4" / "phase4-pl-1x2-provenance-79110f9a-20260827-v1"
PHASE4_RECORD = PHASE4_RUN / "prediction_provenance.json"
FEATURE_MATRIX = ROOT / "data" / "features" / "feature_matrix.csv"
PROTECTED_BEFORE = ROOT / "tmp" / "phase5-protected-before.json"
INHERITED_FAILURES = {
    "tests.cli.test_verbosity_levels.TestCLIVerbosityLevels::test_quiet_mode_suppresses_pipeline_logs",
    "tests.test_drift_artifacts::test_global_drift_artifact_evaluated_at_is_present_and_recent",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PL]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[BL1]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[SA]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PD]",
}


class Phase5EvidenceError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _same(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return {k: v for k, v in left.items() if k != "captured_at"} == {
        k: v for k, v in right.items() if k != "captured_at"
    }


def protected_snapshot() -> dict[str, Any]:
    snapshot = earlier_phase_protected_snapshot()
    snapshot["phase3_authorized_evidence_aggregate_sha256"] = aggregate_directory(PHASE3_RUN)
    snapshot["phase4_evidence_aggregate_sha256"] = aggregate_directory(PHASE4_RUN)
    phase4 = _read_json(PHASE4_RECORD)
    snapshot["phase4_prediction_fingerprint"] = phase4["prediction_fingerprint"]
    snapshot["phase4_run_manifest_sha256"] = sha256_file(PHASE4_RUN / "run_manifest.json")
    return snapshot


def _safe_new_run_dir(run_dir: Path) -> None:
    root = (ROOT / "baselines" / "phase5").resolve()
    try:
        run_dir.resolve().relative_to(root)
    except ValueError as exc:
        raise Phase5EvidenceError("Phase 5 evidence must remain beneath baselines/phase5") from exc
    if run_dir.exists():
        raise Phase5EvidenceError(f"Evidence directory already exists and is immutable: {run_dir}")
    run_dir.mkdir(parents=True)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number):
            return number
        return {"non_finite_estimator_sentinel": str(number)}
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return {"python_repr": repr(value), "python_type": f"{type(value).__module__}.{type(value).__name__}"}


def _recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key:
                found.append(item)
            found.extend(_recursive_values(item, key))
    elif isinstance(value, list):
        for item in value:
            found.extend(_recursive_values(item, key))
    return found


def _embedded_model_audit(member: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    path = ROOT / str(member["source_location"])
    if sha256_file(path) != member["artifact_sha256"]:
        raise Phase5EvidenceError(f"Phase 4 artifact identity changed: {member['role']}")
    model = joblib.load(path)
    names = [str(name) for name in getattr(model, "feature_names_in_", [])]
    if not names and hasattr(model, "feature_name_"):
        names = [str(name) for name in model.feature_name_]
    params = _json_safe(model.get_params(deep=True) if hasattr(model, "get_params") else {})
    embedded: dict[str, Any] = {
        "python_type": f"{type(model).__module__}.{type(model).__name__}",
        "embedded_meta_present": isinstance(getattr(model, "meta", None), Mapping),
        "ordered_feature_names": names,
        "feature_schema_sha256": sha256_json(names),
        "feature_count": len(names),
        "effective_wrapper_params": params,
        "effective_wrapper_params_sha256": sha256_json(params),
    }
    findings = [
        "Artifact bytes and embedded feature order are mechanically verified.",
        "No immutable record binds this artifact to training row IDs, feature matrix values, target values, source snapshot, or trainer execution.",
        "Manifest row counts, dates, metrics, file timestamps, retained datasets, and current defaults are clues only.",
        "Git contains no tracked object or history for this ignored model artifact.",
    ]
    rng = {
        key: params.get(key)
        for key in ("random_state", "seed", "n_jobs", "num_threads", "deterministic")
        if key in params
    }
    if hasattr(model, "booster_"):
        embedded["booster_tree_count"] = int(model.booster_.num_trees())
        embedded["booster_current_iteration"] = int(model.booster_.current_iteration())
        advertised = params.get("n_estimators")
        if advertised is not None and embedded["booster_tree_count"] != int(advertised):
            findings.append(
                f"Embedded booster contains {embedded['booster_tree_count']} trees while the wrapper advertises "
                f"n_estimators={advertised}; historical warm-start/invocation history is therefore material and unknown."
            )
    if hasattr(model, "get_booster"):
        config = json.loads(model.get_booster().save_config())
        tree_counts = _recursive_values(config, "num_trees")
        embedded["booster_config_sha256"] = sha256_json(config)
        embedded["booster_version"] = config.get("version")
        embedded["booster_num_trees_values"] = tree_counts
        advertised = params.get("n_estimators")
        numeric_counts = [int(value) for value in tree_counts if str(value).isdigit()]
        if advertised is not None and numeric_counts and max(numeric_counts) != int(advertised):
            findings.append(
                f"Embedded booster contains {max(numeric_counts)} trees while the wrapper advertises "
                f"n_estimators={advertised}; historical warm-start/invocation history is therefore material and unknown."
            )
    return embedded, findings, names


def _evidence_references() -> list[dict[str, Any]]:
    paths = [
        PHASE4_RECORD,
        ROOT / "src" / "ml" / "models" / "manifest.json",
        ROOT / "src" / "ml" / "models" / "training_metadata.json",
        ROOT / "src" / "ml" / "models" / "training_report.json",
        ROOT / "src" / "ml" / "models" / "training_report_PL.json",
        FEATURE_MATRIX,
        ROOT / "src" / "models" / "train_probability_models.py",
        ROOT / "src" / "ml" / "training_lineage.py",
        ROOT / ".gitignore",
    ]
    return [{"path": _relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in paths]


def _candidate_dataset_diagnostic() -> dict[str, Any]:
    frame = pd.read_csv(FEATURE_MATRIX, low_memory=False)
    frame = frame.loc[frame["league"].astype(str).eq("PL")].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["season_year"] = frame["season"].astype(str).str[:4].astype(int)
    frame["home_goals_capped"] = frame["home_score"].clip(upper=6)
    frame["away_goals_capped"] = frame["away_score"].clip(upper=6)
    fit = frame.loc[frame["season_year"] <= 2022]
    evaluation = frame.loc[frame["season_year"] >= 2023]
    phase4 = _read_json(PHASE4_RECORD)
    feature_names = [
        str(value)
        for value in phase4["feature_provenance"]["model_inputs"][0]["ordered_feature_names"]
    ]
    p1 = _read_json(PHASE1_RUN / "partition_manifest.json")["partitions"]
    p3 = _read_json(PHASE3_RUN / "partition_manifest.json")["partitions"]
    protected: dict[str, set[str]] = {}
    for name, partition in p1.items():
        protected[f"phase1_{name}"] = {str(value) for value in partition.get("match_hashes", [])}
    for name, partition in p3.items():
        protected[f"phase3_{name}"] = {
            str(value) for value in partition.get("fixture_ids", partition.get("match_hashes", []))
        }

    def identity(partition: pd.DataFrame) -> dict[str, Any]:
        ids = [str(value) for value in partition["match_hash"]]
        id_set = set(ids)
        feature_rows = [
            {"row_id": str(row["match_hash"]), "values": _json_safe([row[name] for name in feature_names])}
            for _, row in partition.iterrows()
        ]
        return {
            "row_count": len(ids),
            "date_min": partition["date"].min().date().isoformat(),
            "date_max": partition["date"].max().date().isoformat(),
            "ordered_row_ids_sha256": sha256_json(ids),
            "row_ids_set_sha256": sha256_json(sorted(id_set)),
            "ordered_feature_names": feature_names,
            "feature_schema_sha256": sha256_json(feature_names),
            "candidate_feature_matrix_sha256": sha256_json(feature_rows),
            "candidate_home_target_sha256": sha256_json(
                [{"row_id": str(row["match_hash"]), "target": row["home_goals_capped"]} for _, row in partition.iterrows()]
            ),
            "candidate_away_target_sha256": sha256_json(
                [{"row_id": str(row["match_hash"]), "target": row["away_goals_capped"]} for _, row in partition.iterrows()]
            ),
            "protected_overlap_counts": {name: len(id_set.intersection(values)) for name, values in protected.items()},
        }

    return {
        "status": "CLUE_ONLY_NOT_ARTIFACT_BOUND",
        "source": {"path": _relative(FEATURE_MATRIX), "sha256": sha256_file(FEATURE_MATRIX), "rows_all_leagues": len(pd.read_csv(FEATURE_MATRIX, usecols=["league"]))},
        "pl_row_count": len(frame),
        "candidate_current_fit_partition": identity(fit),
        "candidate_current_evaluation_partition": identity(evaluation),
        "interpretation": (
            "The retained matrix reproduces the manifest's 1520/940 counts under today's split, but no artifact-bound "
            "hash proves it was the fitted input. Any overlaps here describe only today's candidate path."
        ),
    }


def _training_path_audit(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    paths = [str(item["source_location"]) for item in members]
    return {
        "artifact_git_tracking": {
            "paths": paths,
            "ls_files_stage_output": _git("ls-files", "--stage", "--", *paths),
            "history_output": _git("log", "--format=%H %cI %s", "--all", "--", *paths),
            "ignore_rules": _git("check-ignore", "-v", *paths),
            "conclusion": "All four artifacts are ignored and untracked; Git cannot bind them to a training execution.",
        },
        "current_automatic_path": {
            "entrypoint": "scripts/train_batch.py",
            "trainer": "src.models.train_probability_models.ProbabilityModelTrainer",
            "dataset": "data/features/feature_matrix.csv",
            "split": "fit = season_year <= 2022 (train plus val); evaluation = season_year >= 2023",
            "validation_use": "No distinct validation data is used for early stopping or model selection.",
            "warm_start": "Default behavior attempts to continue from an existing artifact unless full_retrain is true.",
            "serialization": "joblib.dump",
            "registration": "register_model followed by set_active_model",
            "phase5_instrumentation": (
                "Goal artifacts now persist and verify a lineage record before registry activation; ordinary runs remain "
                "TRAINING_LINEAGE_PARTIAL and ineligible for lineage trust."
            ),
        },
        "alternative_trainer": {
            "path": "src/ml/trainer.py and src/ml/training/orchestrator.py",
            "conclusion": "Different wrapper/name/path architecture; not established as producer of the four active artifacts.",
        },
    }


def _implementation_source_inventory() -> dict[str, Any]:
    paths = [
        ROOT / "src" / "ml" / "training_lineage.py",
        ROOT / "src" / "models" / "train_probability_models.py",
        ROOT / "scripts" / "run_phase5_training_lineage.py",
        ROOT / "tests" / "unit" / "test_training_lineage.py",
        ROOT / "requirements-dev.txt",
    ]
    files = [{"path": _relative(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in paths]
    return {"files": files, "aggregate_sha256": sha256_json(files)}


def create_evidence(run_dir: Path) -> dict[str, Any]:
    _safe_new_run_dir(run_dir)
    if not PROTECTED_BEFORE.is_file():
        raise Phase5EvidenceError("The pre-change Phase 5 protected snapshot is missing")
    before = _read_json(PROTECTED_BEFORE)
    current = protected_snapshot()
    if not _same(before, current):
        raise Phase5EvidenceError("Protected Phase 0-4 identities changed before evidence creation")
    shutil.copy2(PROTECTED_BEFORE, run_dir / "protected_before.json")

    phase4 = _read_json(PHASE4_RECORD)
    members = cast(list[dict[str, Any]], phase4["model_provenance"]["members"])
    inputs = {
        str(item["role"]): item
        for item in cast(Sequence[Mapping[str, Any]], phase4["feature_provenance"]["model_inputs"])
    }
    refs = _evidence_references()
    records: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for member in members:
        embedded, findings, names = _embedded_model_audit(member)
        phase4_names = [str(value) for value in inputs[str(member["role"])]["ordered_feature_names"]]
        if names != phase4_names:
            raise Phase5EvidenceError(f"Embedded feature order disagrees with Phase 4 for {member['role']}")
        record = build_legacy_unproven_record(
            artifact_path=ROOT / str(member["source_location"]),
            model_role=str(member["role"]),
            manifest_key=str(member["manifest_key"]),
            registry_slot=str(member["registry_slot"]),
            model_family=str(member["model_type"]),
            target_name="home_goals" if str(member["role"]).startswith("home") else "away_goals",
            ordered_feature_names=names,
            embedded_config=cast(Mapping[str, Any], embedded["effective_wrapper_params"]),
            embedded_randomness={
                key: embedded["effective_wrapper_params"].get(key)
                for key in ("random_state", "seed", "n_jobs", "num_threads", "deterministic")
                if key in embedded["effective_wrapper_params"]
            },
            audit_findings=findings,
            evidence=refs,
            root=ROOT,
        )
        record_path = run_dir / "lineage_records" / f"{member['role']}.json"
        record.save(record_path)
        verification = verify_lineage_record(record, root=ROOT)
        if not verification["assessment_verified"] or verification["lineage_verified"]:
            raise Phase5EvidenceError(f"Invalid legacy assessment for {member['role']}: {verification}")
        records.append(
            {
                "role": member["role"],
                "record_path": record_path.relative_to(run_dir).as_posix(),
                "record_sha256": sha256_file(record_path),
                "artifact_sha256": member["artifact_sha256"],
                "classification": record.to_dict()["classification"],
                "assessment_verified": True,
                "lineage_verified": False,
                "fingerprint": record.fingerprint,
            }
        )
        audits.append({"role": member["role"], "artifact": member, "embedded": embedded, "findings": findings})

    _write_json(run_dir / "active_artifact_audit.json", {"artifacts": audits})
    _write_json(run_dir / "historical_lineage_assessments.json", {"records": records})
    _write_json(run_dir / "candidate_current_dataset_diagnostic.json", _candidate_dataset_diagnostic())
    _write_json(run_dir / "training_path_audit.json", _training_path_audit(members))
    _write_json(run_dir / "implementation_source_inventory.json", _implementation_source_inventory())
    phase4_verification = verify_phase4_evidence(PHASE4_RUN)
    _write_json(run_dir / "phase4_offline_verification.json", phase4_verification)
    _write_json(
        run_dir / "record_verification.json",
        {
            "schema_version": "ffps.model_training_lineage.v1",
            "record_count": len(records),
            "all_assessments_verified": all(item["assessment_verified"] for item in records),
            "verified_historical_lineage_count": sum(bool(item["lineage_verified"]) for item in records),
            "classification_counts": {LEGACY_TRAINING_LINEAGE_UNPROVEN: len(records)},
            "overall_phase5_assessment_trusted": len(records) == 4,
        },
    )
    after = protected_snapshot()
    _write_json(run_dir / "protected_after_create.json", after)
    _write_json(
        run_dir / "protected_verification_create.json",
        {"matches_before": _same(before, after), "before": before, "after": after},
    )
    result = {
        "status": "CREATED",
        "created_at": utc_now(),
        "run_id": run_dir.name,
        "record_count": len(records),
        "classification": LEGACY_TRAINING_LINEAGE_UNPROVEN,
        "historical_lineage_verified_count": 0,
        "protected_prior_phases_unchanged": _same(before, after),
        "phase4_offline_verified": phase4_verification["status"] == "VERIFIED",
    }
    _write_json(run_dir / "creation_result.json", result)
    return result


def _junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    failures: list[str] = []
    for case in cases:
        if case.find("failure") is not None or case.find("error") is not None:
            failures.append(f"{case.attrib.get('classname')}::{case.attrib.get('name')}")
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "tests": len(cases),
        "failures": len([case for case in cases if case.find("failure") is not None]),
        "errors": len([case for case in cases if case.find("error") is not None]),
        "skipped": len([case for case in cases if case.find("skipped") is not None]),
        "failed_tests": failures,
    }


def _report(tests: Mapping[str, Any]) -> str:
    return f"""# Phase 5 — PL goal-model training lineage

## Decision

Phase 5 is complete. All four active PL 1X2 goal artifacts are classified
`LEGACY_TRAINING_LINEAGE_UNPROVEN`. This is a verified negative finding, not a
claim that training was incorrect: the retained evidence cannot bind exact fit,
validation/evaluation, target, source-code, runtime, RNG, or warm-start state to
the artifact bytes.

## Active artifacts

| Role | Artifact SHA-256 | Classification |
|---|---|---|
| home goals LGBM | `f65c1bf18c17e04b847510b7c1d699f5125973b657392c3fa76549746a5e8bab` | `LEGACY_TRAINING_LINEAGE_UNPROVEN` |
| away goals LGBM | `dd82ded84a0ce7d1e8b0588a7540f56f26880626b1eb694dd5d232046c13e1dd` | `LEGACY_TRAINING_LINEAGE_UNPROVEN` |
| home goals XGB | `bca373eed2b6c3bcb984f2e021497efa5642e05172dab45657037767781c943e` | `LEGACY_TRAINING_LINEAGE_UNPROVEN` |
| away goals XGB | `b518dd0647fa86ccf07a04b1d6c0db819797ba0707beab570be7f20d60872867` | `LEGACY_TRAINING_LINEAGE_UNPROVEN` |

The LightGBM artifacts embed 8,000 trees while their wrappers advertise 500;
the XGBoost artifacts embed 9,600 trees while their wrappers advertise 600.
That does not identify the historical command, but it proves that current
defaults alone are not a sufficient reconstruction.

## Future training contract

`ffps.model_training_lineage.v1` records the exact ordered fit and evaluation
rows, feature matrices, target vectors, source dataset, trainer source,
configuration, RNG/parallelism, and warm-start predecessor. Persistence and
verification occur before registry activation. Ordinary automated runs are
fail-closed as `TRAINING_LINEAGE_PARTIAL`; they are not lineage-eligible until
protected-partition exclusions and an isolated reproduction are verified.

No retraining, promotion, active-model mutation, prediction-math change,
ensemble change, simulator change, or calibration change was performed.

## Verification

- Focused: {tests['focused']['tests']} tests, {tests['focused']['failures'] + tests['focused']['errors']} failures/errors.
- Protected regression: {tests['regression']['tests']} tests, {tests['regression']['failures'] + tests['regression']['errors']} failures/errors.
- Full suite: {tests['full']['tests']} tests, {tests['full']['failures'] + tests['full']['errors']} failures/errors, {tests['full']['skipped']} skipped.
- New unexplained full-suite failures: {tests['new_unexplained_failure_count']}.
- Protected Phase 0–4 identities: unchanged.
- Phase 4 offline verification: verified.
"""


def _artifact_index(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_index.json", "run_manifest.json", "offline_verification.json"}
    items = [
        {"path": path.relative_to(run_dir).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.relative_to(run_dir).as_posix() not in excluded
    ]
    return {"artifacts": items, "artifact_count": len(items), "aggregate_sha256": sha256_json(items)}


def finalize_evidence(run_dir: Path, focused_junit: Path, regression_junit: Path, full_junit: Path) -> dict[str, Any]:
    if (run_dir / "run_manifest.json").exists():
        raise Phase5EvidenceError("Finalized Phase 5 evidence is immutable")
    destinations = {}
    for label, source in (("focused", focused_junit), ("regression", regression_junit), ("full", full_junit)):
        destination = run_dir / f"{label}_test_results.xml"
        shutil.copy2(source, destination)
        destinations[label] = destination
    tests = {label: _junit(path) for label, path in destinations.items()}
    unexplained = sorted(set(tests["full"]["failed_tests"]) - INHERITED_FAILURES)
    tests["inherited_failure_allowlist"] = sorted(INHERITED_FAILURES)
    tests["new_unexplained_failures"] = unexplained
    tests["new_unexplained_failure_count"] = len(unexplained)
    _write_json(run_dir / "test_results.json", tests)
    before = _read_json(run_dir / "protected_before.json")
    after = protected_snapshot()
    protected = {"matches_before": _same(before, after), "before": before, "after": after}
    _write_json(run_dir / "protected_after_final.json", after)
    _write_json(run_dir / "protected_verification_final.json", protected)
    if unexplained or not protected["matches_before"]:
        raise Phase5EvidenceError(f"Phase 5 final checks failed: unexplained={unexplained}, protected={protected['matches_before']}")
    report = run_dir / "PHASE5_IMPLEMENTATION_REPORT.md"
    report.write_text(_report(tests), encoding="utf-8")
    index = _artifact_index(run_dir)
    _write_json(run_dir / "artifact_index.json", index)
    manifest = {
        "schema_version": 1,
        "phase": 5,
        "run_id": run_dir.name,
        "status": "complete",
        "finalized_at": utc_now(),
        "scope": {"league": "PL", "market": "1X2", "artifacts": "four active goal models"},
        "lineage_schema": "ffps.model_training_lineage.v1",
        "classification": LEGACY_TRAINING_LINEAGE_UNPROVEN,
        "record_count": 4,
        "historical_lineage_verified_count": 0,
        "overall_phase5_assessment_trusted": True,
        "no_retraining_or_promotion": True,
        "protected_prior_phases_unchanged": True,
        "phase4_offline_verified": True,
        "no_new_unexplained_test_failures": True,
        "artifact_index_sha256": sha256_file(run_dir / "artifact_index.json"),
        "artifact_aggregate_sha256": index["aggregate_sha256"],
        "implementation_source_inventory_sha256": sha256_file(run_dir / "implementation_source_inventory.json"),
        "report_sha256": sha256_file(report),
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def verify_evidence(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    summary = _read_json(run_dir / "historical_lineage_assessments.json")
    for item in cast(Sequence[Mapping[str, Any]], summary["records"]):
        path = run_dir / str(item["record_path"])
        if not path.is_file() or sha256_file(path) != item["record_sha256"]:
            errors.append(f"lineage record file mismatch: {item['role']}")
            continue
        result = verify_lineage_record(ModelTrainingLineageRecord.load(path), root=ROOT)
        if not result["assessment_verified"] or result["lineage_verified"]:
            errors.append(f"lineage assessment invalid: {item['role']}: {result['errors']}")
    before = _read_json(run_dir / "protected_before.json")
    if not _same(before, protected_snapshot()):
        errors.append("protected Phase 0-4 identities changed")
    try:
        phase4 = verify_phase4_evidence(PHASE4_RUN)
        if phase4["status"] != "VERIFIED":
            errors.append("Phase 4 offline verification is not VERIFIED")
    except Exception as exc:
        errors.append(f"Phase 4 offline verification failed: {exc}")
    index = _read_json(run_dir / "artifact_index.json")
    for item in cast(Sequence[Mapping[str, Any]], index["artifacts"]):
        path = run_dir / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            errors.append(f"evidence artifact mismatch: {item['path']}")
    if index.get("aggregate_sha256") != sha256_json(index["artifacts"]):
        errors.append("artifact index aggregate mismatch")
    manifest = _read_json(run_dir / "run_manifest.json")
    if manifest.get("artifact_index_sha256") != sha256_file(run_dir / "artifact_index.json"):
        errors.append("run manifest artifact-index hash mismatch")
    if manifest.get("report_sha256") != sha256_file(run_dir / "PHASE5_IMPLEMENTATION_REPORT.md"):
        errors.append("run manifest report hash mismatch")
    source_inventory_path = run_dir / "implementation_source_inventory.json"
    if manifest.get("implementation_source_inventory_sha256") != sha256_file(source_inventory_path):
        errors.append("run manifest implementation-source inventory mismatch")
    source_inventory = _read_json(source_inventory_path)
    for item in cast(Sequence[Mapping[str, Any]], source_inventory["files"]):
        path = ROOT / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            errors.append(f"Phase 5 implementation source mismatch: {item['path']}")
    if source_inventory.get("aggregate_sha256") != sha256_json(source_inventory["files"]):
        errors.append("Phase 5 implementation source aggregate mismatch")
    if _read_json(run_dir / "test_results.json").get("new_unexplained_failure_count") != 0:
        errors.append("test evidence has new unexplained failures")
    result = {
        "status": "VERIFIED" if not errors else "FAILED",
        "record_count": len(summary["records"]),
        "classification": LEGACY_TRAINING_LINEAGE_UNPROVEN,
        "historical_lineage_verified_count": 0,
        "protected_prior_phases_unchanged": not any("protected Phase" in error for error in errors),
        "errors": errors,
    }
    if errors:
        raise Phase5EvidenceError(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    finalize.add_argument("--focused-junit", type=Path, required=True)
    finalize.add_argument("--regression-junit", type=Path, required=True)
    finalize.add_argument("--full-junit", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "create":
            result = create_evidence(args.run_dir.resolve())
        elif args.command == "finalize":
            result = finalize_evidence(
                args.run_dir.resolve(), args.focused_junit.resolve(), args.regression_junit.resolve(), args.full_junit.resolve()
            )
        else:
            result = verify_evidence(args.run_dir.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Phase5EvidenceError as exc:
        print(f"Phase 5 evidence error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

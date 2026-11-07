"""Create, finalize, and offline-verify FFPS Phase 4 PL 1X2 provenance evidence."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from src.cli.commands.prediction import _calc_goals, _load_prediction_models
from src.evaluation.phase3_calibration import aggregate_directory
from src.evaluation.phase3_pl_1x2_temperature import protected_snapshot as earlier_phase_protected_snapshot
from src.predictions.provenance import (
    PredictionCapture,
    PredictionProvenanceRecord,
    build_file_snapshot,
    data_provenance_for_fixture,
    mark_output_reproduced,
    replay_pl_1x2,
    sha256_bytes,
    sha256_file,
    sha256_json,
    source_event_index,
    utc_now,
    verify_record,
)
from src.simulation.match_simulator import DEFAULT_N_SIMULATIONS, MatchSimulator


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "phase4-pl-1x2-provenance-79110f9a-20260827-v1"
DEFAULT_RUN_DIR = ROOT / "baselines" / "phase4" / DEFAULT_RUN_ID
FEATURE_MATRIX = ROOT / "data" / "features" / "feature_matrix.csv"
RAW_MATCHES = ROOT / "data" / "processed" / "matches.csv"
PHASE1_RUN = ROOT / "baselines" / "phase1" / "phase1-pl-1x2-79110f9a-20240801-v1"
PHASE3_AUTHORIZED_RUN = ROOT / "baselines" / "phase3" / "phase3-pl-1x2-temperature-79110f9a-20260827-v1"
INHERITED_FAILURES = {
    "tests.cli.test_verbosity_levels.TestCLIVerbosityLevels::test_quiet_mode_suppresses_pipeline_logs",
    "tests.test_drift_artifacts::test_global_drift_artifact_evaluated_at_is_present_and_recent",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PL]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[BL1]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[SA]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PD]",
}


class Phase4EvidenceError(RuntimeError):
    pass


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def protected_snapshot() -> dict[str, Any]:
    """Extend prior-phase identities through the finalized authorized Phase 3 run."""
    snapshot = earlier_phase_protected_snapshot()
    snapshot["phase3_authorized_evidence_aggregate_sha256"] = aggregate_directory(PHASE3_AUTHORIZED_RUN)
    return snapshot


def _read_first_phase1_fixture() -> dict[str, Any]:
    path = PHASE1_RUN / "fixture_predictions.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        first = next(line for line in handle if line.strip())
    return cast(dict[str, Any], json.loads(first))


def _protected_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return {key: value for key, value in left.items() if key != "captured_at"} == {
        key: value for key, value in right.items() if key != "captured_at"
    }


def _safe_new_run_dir(run_dir: Path) -> None:
    resolved = run_dir.resolve()
    phase4_root = (ROOT / "baselines" / "phase4").resolve()
    try:
        resolved.relative_to(phase4_root)
    except ValueError as exc:
        raise Phase4EvidenceError("Phase 4 evidence must remain beneath baselines/phase4") from exc
    if resolved.exists():
        raise Phase4EvidenceError(f"Evidence directory already exists and is immutable: {resolved}")
    resolved.mkdir(parents=True)


def _input_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_bytes = FEATURE_MATRIX.read_bytes()
    raw_bytes = RAW_MATCHES.read_bytes()
    matrix = pd.read_csv(FEATURE_MATRIX, low_memory=False)
    raw = pd.read_csv(RAW_MATCHES, low_memory=False)
    matrix["date"] = pd.to_datetime(matrix["date"], utc=True, errors="coerce")
    raw_date_column = "date" if "date" in raw.columns else "match_date"
    raw["date"] = pd.to_datetime(raw[raw_date_column], utc=True, errors="coerce")
    if "home_score" not in raw.columns and "home_goals" in raw.columns:
        raw["home_score"] = raw["home_goals"]
    if "away_score" not in raw.columns and "away_goals" in raw.columns:
        raw["away_score"] = raw["away_goals"]
    matrix = matrix.loc[matrix["league"].astype(str).eq("PL")].copy()
    raw = raw.loc[raw["league"].astype(str).eq("PL")].copy()
    artifacts = [
        {
            "path": _relative(FEATURE_MATRIX),
            "sha256": sha256_bytes(matrix_bytes),
            "size_bytes": len(matrix_bytes),
            "row_count": len(matrix),
            "load_order": 0,
            "role": "exact_precomputed_inference_frame",
        },
        {
            "path": _relative(RAW_MATCHES),
            "sha256": sha256_bytes(raw_bytes),
            "size_bytes": len(raw_bytes),
            "row_count": len(raw),
            "load_order": 1,
            "role": "causal_source_event_audit",
        },
    ]
    matrix.attrs["ffps_data_snapshot"] = build_file_snapshot(
        frame=matrix,
        artifacts=artifacts,
        source_type="phase4_frozen_pl_inference_and_source_rows",
    )
    matrix.attrs["ffps_source_events"] = source_event_index(raw)
    return matrix, raw


def _model_summary(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": item["role"],
            "manifest_key": item["manifest_key"],
            "source_location": item["source_location"],
            "artifact_sha256": item["artifact_sha256"],
            "training_lineage": item["training_lineage"]["classification"],
        }
        for item in record["model_provenance"]["members"]
    ]


def _artifact_tamper_probe(record: PredictionProvenanceRecord) -> dict[str, Any]:
    value = record.to_dict()
    with tempfile.TemporaryDirectory(prefix="ffps-phase4-") as temporary:
        temporary_root = Path(temporary)
        referenced = [
            *(member["source_location"] for member in value["model_provenance"]["members"]),
            *(artifact["path"] for artifact in value["data_provenance"]["source_identity"]["artifacts"]),
            *(source["path"] for source in value["code_provenance"]["relevant_source_files"]),
        ]
        for location in referenced:
            relative = Path(location)
            target = temporary_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        before = verify_record(record, root=temporary_root, verify_referenced_artifacts=True)
        first = Path(value["model_provenance"]["members"][0]["source_location"])
        target = temporary_root / first
        target.write_bytes(target.read_bytes() + b"phase4-tamper")
        after = verify_record(record, root=temporary_root, verify_referenced_artifacts=True)
    return {
        "verified_before_tamper": before["verified"],
        "verified_after_tamper": after["verified"],
        "detected": bool(before["verified"] and not after["verified"]),
        "errors_after_tamper": after["errors"],
    }


def create_evidence(run_dir: Path) -> dict[str, Any]:
    _safe_new_run_dir(run_dir)
    before = protected_snapshot()
    _write_json(run_dir / "protected_before.json", before)

    matrix, _ = _input_frames()
    phase1 = _read_first_phase1_fixture()
    fixture_id = str(phase1["fixture"]["match_hash"])
    rows = matrix.loc[matrix["match_hash"].astype(str).eq(fixture_id)]
    if len(rows) != 1:
        raise Phase4EvidenceError(f"Expected one frozen fixture row for {fixture_id}, got {len(rows)}")
    fixture = rows.iloc[0]
    prediction_timestamp = str(phase1["prediction_time"])
    suite = _load_prediction_models("PL")

    baseline_simulator = MatchSimulator(
        n_simulations=DEFAULT_N_SIMULATIONS,
        league="PL",
        seed=42,
        seed_scope="isolated_fixture_equivalence_probe",
    )
    baseline = _calc_goals(
        fixture,
        suite,
        f"Phase4Baseline:{fixture_id}",
        simulator=baseline_simulator,
        league="PL",
        rl_weights=None,
        use_simulator=True,
    )

    instrumented_simulator = MatchSimulator(
        n_simulations=DEFAULT_N_SIMULATIONS,
        league="PL",
        seed=42,
        seed_scope="isolated_fixture_equivalence_probe",
    )
    capture = PredictionCapture(
        prediction_timestamp=prediction_timestamp,
        data_provenance=data_provenance_for_fixture(matrix, fixture, prediction_timestamp),
    )
    with capture:
        instrumented = _calc_goals(
            fixture,
            suite,
            f"Phase4Instrumented:{fixture_id}",
            simulator=instrumented_simulator,
            league="PL",
            rl_weights=None,
            use_simulator=True,
        )
    probabilities = {key: instrumented[key] for key in ("home", "draw", "away")}
    record = capture.finalize(
        fixture=fixture,
        league="PL",
        probabilities=probabilities,
        simulator=instrumented_simulator,
    )
    initial_fingerprint = record.prediction_fingerprint
    initial_verification = verify_record(record, root=ROOT, verify_referenced_artifacts=True)
    replay = replay_pl_1x2(record, root=ROOT)
    record = mark_output_reproduced(record, replay)
    if record.prediction_fingerprint != initial_fingerprint:
        raise Phase4EvidenceError("Replay finalization changed the calculation fingerprint")
    final_verification = verify_record(record, root=ROOT, verify_referenced_artifacts=True)
    if not final_verification["overall_phase4_trusted"]:
        raise Phase4EvidenceError(f"Final provenance did not verify as trusted: {final_verification}")
    record.save(run_dir / "prediction_provenance.json")

    baseline_output = [float(baseline[key]) for key in ("home", "draw", "away")]
    instrumented_output = [float(instrumented[key]) for key in ("home", "draw", "away")]
    equivalence = {
        "class_order": ["home_win", "draw", "away_win"],
        "baseline": baseline_output,
        "instrumented": instrumented_output,
        "baseline_sha256": sha256_json(baseline_output),
        "instrumented_sha256": sha256_json(instrumented_output),
        "exactly_equal": baseline_output == instrumented_output,
    }
    if not equivalence["exactly_equal"]:
        raise Phase4EvidenceError("Provenance instrumentation changed PL 1X2 output")
    _write_json(run_dir / "prediction_equivalence.json", equivalence)
    _write_json(run_dir / "replay_verification.json", replay)
    _write_json(run_dir / "record_verification.json", final_verification)

    protected_tamper = record.to_dict()
    protected_tamper["fixture"]["home_team"] = "TAMPERED"
    field_tamper = verify_record(protected_tamper, root=ROOT, verify_referenced_artifacts=True)
    tamper = {
        "protected_field_detected": not field_tamper["verified"],
        "protected_field_errors": field_tamper["errors"],
        "referenced_artifact": _artifact_tamper_probe(record),
    }
    if not tamper["protected_field_detected"] or not tamper["referenced_artifact"]["detected"]:
        raise Phase4EvidenceError("Tamper detection probe failed")
    _write_json(run_dir / "tamper_detection.json", tamper)

    value = record.to_dict()
    summary = {
        "schema_version": value["schema_version"],
        "run_id": value["run_id"],
        "fixture": value["fixture"],
        "prediction_fingerprint": value["prediction_fingerprint"],
        "record_integrity_sha256": value["record_integrity_sha256"],
        "feature_bundle_sha256": value["feature_provenance"]["feature_bundle_sha256"],
        "model_bundle_sha256": value["model_provenance"]["model_bundle_sha256"],
        "models": _model_summary(value),
        "method_config_hash": value["method_provenance"]["method_config_hash"],
        "output_sha256": value["prediction_output"]["output_sha256"],
        "probabilities": value["prediction_output"]["final_probabilities"],
        "training_lineage_classification": value["trust_assertions"]["model_training_lineage_classification"],
        "overall_phase4_trusted": final_verification["overall_phase4_trusted"],
    }
    _write_json(run_dir / "provenance_summary.json", summary)

    after = protected_snapshot()
    protected = {"before": before, "after": after, "matches_before": _protected_equal(before, after)}
    if not protected["matches_before"]:
        raise Phase4EvidenceError("A protected Phase 0-3, data, manifest, or active-model identity changed")
    _write_json(run_dir / "protected_after_create.json", after)
    _write_json(run_dir / "protected_verification_create.json", protected)
    result = {
        "created_at": utc_now(),
        "status": "evidence_created_tests_pending",
        "run_dir": _relative(run_dir),
        "prediction_fingerprint": record.prediction_fingerprint,
        "record_run_id": record.run_id,
        "reproduced": replay["reproduced"],
        "instrumentation_unchanged": equivalence["exactly_equal"],
        "tamper_detected": True,
        "protected_unchanged": protected["matches_before"],
    }
    _write_json(run_dir / "creation_result.json", result)
    return result


def _junit_summary(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    failing: list[str] = []
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    for suite in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            totals[key] += int(suite.attrib.get(key, 0))
        totals["time"] += float(suite.attrib.get("time", 0.0))
        for case in suite.findall("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failing.append(f"{case.attrib.get('classname')}::{case.attrib.get('name')}")
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    return {**totals, "failing_node_ids": sorted(failing), "sha256": sha256_file(path), "path": _relative(path)}


def _copy_junit(run_dir: Path, name: str, source: Path) -> Path:
    destination = run_dir / f"{name}_test_results.xml"
    shutil.copy2(source.resolve(), destination)
    return destination


def _artifact_index(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_index.json", "run_manifest.json", "PHASE4_IMPLEMENTATION_REPORT.md"}
    artifacts = [
        {
            "path": path.relative_to(run_dir).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]
    return {"created_at": utc_now(), "artifacts": artifacts, "aggregate_sha256": sha256_json(artifacts)}


def _report(run_dir: Path, tests: Mapping[str, Any], protected: Mapping[str, Any]) -> str:
    summary = _read_json(run_dir / "provenance_summary.json")
    equivalence = _read_json(run_dir / "prediction_equivalence.json")
    replay = _read_json(run_dir / "replay_verification.json")
    tamper = _read_json(run_dir / "tamper_detection.json")
    lines = [
        "# FFPS Phase 4 Prediction Provenance Report",
        "",
        "## Result",
        "",
        "Phase 4 is ready for its next narrow slice: one PL 1X2 execution is fully traceable and independently reproducible without changing prediction arithmetic.",
        "",
        "## Executed path",
        "",
        "`ModelRegistry.load_model` exact bytes -> four existing goal-model calls -> existing 60/40 ensemble -> existing `MatchSimulator` -> raw ordered PL 1X2 output.",
        "",
        f"- Schema: `{summary['schema_version']}`",
        f"- Prediction fingerprint: `{summary['prediction_fingerprint']}`",
        f"- Record integrity: `{summary['record_integrity_sha256']}`",
        f"- Feature bundle: `{summary['feature_bundle_sha256']}`",
        f"- Model bundle: `{summary['model_bundle_sha256']}`",
        f"- Method/config: `{summary['method_config_hash']}`",
        f"- Output: `{summary['output_sha256']}`",
        f"- Ordered probabilities: `{summary['probabilities']}`",
        "",
        "## Verification",
        "",
        f"- Before/after numerical equality: {equivalence['exactly_equal']}",
        f"- Independent exact replay: {replay['reproduced']}",
        f"- Protected-field tamper detected: {tamper['protected_field_detected']}",
        f"- Referenced-artifact tamper detected: {tamper['referenced_artifact']['detected']}",
        f"- Protected Phase 0-3/model/data identities unchanged: {protected['matches_before']}",
        f"- Focused Phase 4 tests: {tests['focused']['passed']} passed, {tests['focused']['failures']} failed, {tests['focused']['errors']} errors.",
        f"- Earlier-phase regression tests: {tests['regression']['passed']} passed, {tests['regression']['failures']} failed, {tests['regression']['errors']} errors.",
        f"- Full suite: {tests['full']['passed']} passed, {tests['full']['failures']} failed, {tests['full']['skipped']} skipped.",
        f"- New unexplained full-suite failures: {tests['new_unexplained_failure_count']}.",
        "",
        "## Trust boundary and deviations",
        "",
        "All four active legacy model artifacts are execution-traceable by exact loaded bytes. Their historical fitting lineage remains `unproven`; Phase 4 does not promote that lineage to trusted.",
        "",
        "The brief's fixture-seed premise is not true for the production API/CLI loop. Production currently advances one seed-42 RNG stream across each league batch. Phase 4 preserves that output contract and records the exact pre-call RNG state plus seed scope, which enables exact replay.",
        "",
        "The evidence fixture uses the same live calculation boundary with an isolated seed-42 simulator solely to prove instrumentation equivalence. No match-outcome classifier, bookmaker prediction input, RL weight, or Phase 3 calibrator was used.",
        "",
        "## Limitations",
        "",
        "This contract is proven only for the PL 1X2 goal-ensemble slice. Other leagues/markets, legacy trainer lineage, production deployment, and prediction quality/profitability remain outside Phase 4.",
    ]
    return "\n".join(lines) + "\n"


def finalize_evidence(
    run_dir: Path,
    *,
    focused_junit: Path,
    regression_junit: Path,
    full_junit: Path,
) -> dict[str, Any]:
    if (run_dir / "artifact_index.json").exists():
        raise Phase4EvidenceError("Evidence bundle is already finalized and immutable")
    record = PredictionProvenanceRecord.load(run_dir / "prediction_provenance.json")
    verification = verify_record(record, root=ROOT, verify_referenced_artifacts=True)
    replay = replay_pl_1x2(record, root=ROOT)
    if not verification["overall_phase4_trusted"] or not replay["reproduced"]:
        raise Phase4EvidenceError("Cannot finalize untrusted or unreproduced provenance")

    copied_junit = {
        "focused": _copy_junit(run_dir, "focused", focused_junit),
        "regression": _copy_junit(run_dir, "regression", regression_junit),
        "full": _copy_junit(run_dir, "full", full_junit),
    }
    tests = {name: _junit_summary(path) for name, path in copied_junit.items()}
    if tests["focused"]["failures"] or tests["focused"]["errors"]:
        raise Phase4EvidenceError("Focused Phase 4 tests failed")
    if tests["regression"]["failures"] or tests["regression"]["errors"]:
        raise Phase4EvidenceError("Earlier-phase regression tests failed")
    full_failures = set(tests["full"]["failing_node_ids"])
    tests["inherited_failures"] = sorted(full_failures.intersection(INHERITED_FAILURES))
    tests["new_unexplained_failures"] = sorted(full_failures.difference(INHERITED_FAILURES))
    tests["new_unexplained_failure_count"] = len(tests["new_unexplained_failures"])
    if tests["new_unexplained_failure_count"]:
        raise Phase4EvidenceError(f"Full suite introduced unexplained failures: {tests['new_unexplained_failures']}")
    _write_json(run_dir / "test_results.json", tests)

    before = _read_json(run_dir / "protected_before.json")
    after = protected_snapshot()
    protected = {"before": before, "after": after, "matches_before": _protected_equal(before, after)}
    if not protected["matches_before"]:
        raise Phase4EvidenceError("Protected identities changed before finalization")
    protected_regressions = {
        "mode": "read_only_snapshot_and_targeted_tests",
        "phase0_to_phase3_regression_tests": tests["regression"],
        "protected_snapshot_matches": protected["matches_before"],
        "all_passed": (
            tests["regression"]["failures"] == 0
            and tests["regression"]["errors"] == 0
            and protected["matches_before"]
        ),
    }
    if not protected_regressions["all_passed"]:
        raise Phase4EvidenceError(f"Protected regression failure: {protected_regressions}")
    _write_json(run_dir / "protected_regression_results.json", protected_regressions)
    _write_json(run_dir / "protected_after_final.json", after)
    _write_json(run_dir / "protected_verification_final.json", protected)

    report_path = run_dir / "PHASE4_IMPLEMENTATION_REPORT.md"
    report_path.write_text(_report(run_dir, tests, protected), encoding="utf-8")
    index = _artifact_index(run_dir)
    _write_json(run_dir / "artifact_index.json", index)
    manifest = {
        "schema_version": 1,
        "phase": 4,
        "run_id": run_dir.name,
        "status": "complete",
        "created_at": _read_json(run_dir / "creation_result.json").get("created_at"),
        "finalized_at": utc_now(),
        "scope": {"league": "PL", "market": "1X2", "calculation_path": "goal_ensemble_match_simulator"},
        "prediction_fingerprint": record.prediction_fingerprint,
        "overall_phase4_trusted": verification["overall_phase4_trusted"],
        "training_lineage_classification": verification["training_lineage_classification"],
        "no_prediction_math_change": True,
        "no_new_unexplained_test_failures": True,
        "protected_prior_phases_unchanged": protected["matches_before"],
        "artifact_index_sha256": sha256_file(run_dir / "artifact_index.json"),
        "artifact_aggregate_sha256": index["aggregate_sha256"],
        "report_sha256": sha256_file(report_path),
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def verify_evidence(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    record = PredictionProvenanceRecord.load(run_dir / "prediction_provenance.json")
    verification = verify_record(record, root=ROOT, verify_referenced_artifacts=True)
    if not verification["overall_phase4_trusted"]:
        errors.extend(verification["errors"] or ["record is not overall Phase 4 trusted"])
    replay = replay_pl_1x2(record, root=ROOT)
    if not replay["reproduced"]:
        errors.append("independent replay mismatch")
    equivalence = _read_json(run_dir / "prediction_equivalence.json")
    if not equivalence["exactly_equal"]:
        errors.append("instrumentation equivalence failed")
    tamper = _read_json(run_dir / "tamper_detection.json")
    if not tamper["protected_field_detected"] or not tamper["referenced_artifact"]["detected"]:
        errors.append("tamper evidence failed")

    before = _read_json(run_dir / "protected_before.json")
    current = protected_snapshot()
    if not _protected_equal(before, current):
        errors.append("current protected identities differ from Phase 4 start")

    index = _read_json(run_dir / "artifact_index.json")
    for item in cast(Sequence[Mapping[str, Any]], index["artifacts"]):
        path = run_dir / str(item["path"])
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != item["sha256"]:
            errors.append(f"evidence artifact mismatch: {item['path']}")
    if sha256_json(index["artifacts"]) != index["aggregate_sha256"]:
        errors.append("artifact index aggregate mismatch")
    manifest_path = run_dir / "run_manifest.json"
    report_path = run_dir / "PHASE4_IMPLEMENTATION_REPORT.md"
    if not manifest_path.is_file():
        errors.append("run manifest is missing")
    else:
        manifest = _read_json(manifest_path)
        if manifest.get("artifact_index_sha256") != sha256_file(run_dir / "artifact_index.json"):
            errors.append("run manifest artifact-index hash mismatch")
        if manifest.get("artifact_aggregate_sha256") != index.get("aggregate_sha256"):
            errors.append("run manifest artifact aggregate mismatch")
        if not report_path.is_file() or manifest.get("report_sha256") != sha256_file(report_path):
            errors.append("run manifest report hash mismatch")
        if manifest.get("prediction_fingerprint") != record.prediction_fingerprint:
            errors.append("run manifest prediction fingerprint mismatch")
    protected_regressions = _read_json(run_dir / "protected_regression_results.json")
    if not protected_regressions.get("all_passed"):
        errors.append("protected regression evidence failed")
    tests = _read_json(run_dir / "test_results.json")
    if tests.get("new_unexplained_failure_count") != 0:
        errors.append("test evidence contains new unexplained failures")
    result = {
        "status": "VERIFIED" if not errors else "FAILED",
        "prediction_fingerprint": record.prediction_fingerprint,
        "reproduced": replay["reproduced"],
        "protected_unchanged": _protected_equal(before, current),
        "errors": errors,
    }
    if errors:
        raise Phase4EvidenceError(json.dumps(result, indent=2))
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
                args.run_dir.resolve(),
                focused_junit=args.focused_junit,
                regression_junit=args.regression_junit,
                full_junit=args.full_junit,
            )
        else:
            result = verify_evidence(args.run_dir.resolve())
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Phase4EvidenceError as exc:
        print(f"Phase 4 evidence error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

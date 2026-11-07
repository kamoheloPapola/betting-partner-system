#!/usr/bin/env python3
"""Create, finalize, and verify the authorized Phase 3 PL 1X2 evidence bundle."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from src.evaluation.phase3_pl_1x2_temperature import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    CLASS_ORDER,
    CONFIDENCE_LEVEL,
    DECISION_RULE,
    ROOT,
    MulticlassTemperatureCalibrator,
    Phase3TemperatureIntegrityError,
    build_calibrator_payload,
    build_partition_contract,
    calibration_decision,
    calibration_fit_rows,
    fit_temperature_calibrator,
    legacy_calibrator_inventory,
    load_calibrator_artifact,
    multiclass_metrics,
    paired_bootstrap,
    probabilities_and_labels,
    protected_snapshot,
    read_json,
    row_ids_hash,
    sha256_file,
    sha256_json,
    source_state,
    utc_now,
    write_json,
    write_jsonl,
)


DEFAULT_RUN_ID = "phase3-pl-1x2-temperature-79110f9a-20260827-v1"
DEFAULT_RUN_DIR = ROOT / "baselines" / "phase3" / DEFAULT_RUN_ID
CALIBRATOR_SOURCE = ROOT / "src" / "evaluation" / "phase3_pl_1x2_temperature.py"
RUNNER_SOURCE = Path(__file__).resolve()

INHERITED_FAILURES = {
    "tests.cli.test_verbosity_levels.TestCLIVerbosityLevels::test_quiet_mode_suppresses_pipeline_logs",
    "tests.test_drift_artifacts::test_global_drift_artifact_evaluated_at_is_present_and_recent",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PL]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[BL1]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[SA]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PD]",
}


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _record_with_calibration(
    record: Mapping[str, Any],
    calibrated: np.ndarray,
    *,
    calibrator_id: str,
) -> dict[str, Any]:
    calibrated_probabilities = {
        name: float(calibrated[index]) for index, name in enumerate(CLASS_ORDER)
    }
    return {
        "schema_version": 1,
        "prediction_run_id": DEFAULT_RUN_ID,
        "match_hash": record["match_hash"],
        "fixture": record["fixture"],
        "prediction_time": record["prediction_time"],
        "actual_result": record["actual_result"],
        "raw_probabilities": record["raw_probabilities"],
        "calibrated_probabilities": calibrated_probabilities,
        "calibrator_id": calibrator_id,
        "calibration_method": "multiclass_temperature_scaling",
        "raw_prediction_hash": record["raw_prediction_hash"],
        "calibrated_probability_hash": sha256_json({
            "class_order": list(CLASS_ORDER),
            "probabilities": calibrated_probabilities,
        }),
        "phase1_record_hash": record["phase1_record_hash"],
    }


def _protected_equal(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return {
        key: value for key, value in before.items() if key != "captured_at"
    } == {
        key: value for key, value in after.items() if key != "captured_at"
    }


def create_evidence(run_dir: Path) -> dict[str, Any]:
    if run_dir.exists():
        raise Phase3TemperatureIntegrityError(
            f"Refusing to overwrite existing Phase 3 evidence directory: {_relative(run_dir)}"
        )
    run_dir.mkdir(parents=True)
    before = protected_snapshot()
    write_json(run_dir / "protected_before.json", before)
    starting_source_state = source_state()

    contract = build_partition_contract()
    write_json(run_dir / "partition_manifest.json", contract.manifest)
    write_json(run_dir / "exclusions.json", contract.manifest["exclusions"])
    raw_all = [*contract.calibration_fit, *contract.final_test]
    write_jsonl(run_dir / "raw_live_path_predictions.jsonl", raw_all)

    experiment_spec = {
        "schema_version": 1,
        "created_at": utc_now(),
        "run_id": run_dir.name,
        "scope": {"league": "PL", "market": "1X2", "calibrator_family": "temperature_scaling_only"},
        "identity_baseline": "no calibration / identity transform",
        "decision_rule": DECISION_RULE,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "confidence_level": CONFIDENCE_LEVEL,
            "paired": True,
        },
        "partition_manifest_sha256": sha256_file(run_dir / "partition_manifest.json"),
        "raw_prediction_source_sha256": contract.manifest["source"]["phase1_predictions_sha256"],
        "final_test_metrics_accessed": False,
        "production_promotion": False,
        "source_state_before_evidence": starting_source_state,
    }
    write_json(run_dir / "experiment_spec.json", experiment_spec)

    fit_rows = calibration_fit_rows(contract)
    calibrator, fit_diagnostics = fit_temperature_calibrator(fit_rows, contract.manifest)
    repeated, repeated_diagnostics = fit_temperature_calibrator(fit_rows, contract.manifest)
    if abs(calibrator.temperature - repeated.temperature) > 1e-12:
        raise Phase3TemperatureIntegrityError("Repeated calibration fit was not deterministic")

    calibrator_id = (
        "phase3-pl-1x2-temperature-"
        + str(contract.manifest["partitions"]["calibration_fit"]["fixture_ids_sha256"])[:16]
        + "-v1"
    )
    calibrator_payload = build_calibrator_payload(
        calibrator,
        fit_diagnostics,
        contract,
        calibrator_id=calibrator_id,
        source_code_path=CALIBRATOR_SOURCE,
    )
    calibrator_dir = run_dir / "calibrator"
    calibrator_path = calibrator_dir / "pl_1x2_temperature.json"
    write_json(calibrator_path, calibrator_payload)
    calibrator_manifest = {
        "schema_version": 1,
        "calibrator_id": calibrator_id,
        "path": _relative(calibrator_path),
        "artifact_sha256": sha256_file(calibrator_path),
        "method": "multiclass_temperature_scaling",
        "league": "PL",
        "market": "1X2",
        "class_order": list(CLASS_ORDER),
        "temperature": calibrator.temperature,
        "evaluation_only": True,
        "production_eligible": False,
    }
    write_json(calibrator_dir / "calibrator_manifest.json", calibrator_manifest)
    loaded, _ = load_calibrator_artifact(
        calibrator_path,
        calibrator_dir / "calibrator_manifest.json",
        contract.manifest,
    )

    calibration_raw, calibration_labels = probabilities_and_labels(contract.calibration_fit)
    calibration_calibrated = loaded.transform(calibration_raw)
    calibration_records = [
        _record_with_calibration(record, probabilities, calibrator_id=calibrator_id)
        for record, probabilities in zip(contract.calibration_fit, calibration_calibrated, strict=True)
    ]
    write_jsonl(run_dir / "calibration_fit_predictions.jsonl", calibration_records)
    calibration_metrics = {
        "partition": "calibration_fit",
        "diagnostic_only": True,
        "raw": multiclass_metrics(calibration_labels, calibration_raw),
        "calibrated": multiclass_metrics(calibration_labels, calibration_calibrated),
    }
    write_json(run_dir / "calibration_fit_metrics.json", calibration_metrics)

    # The calibrator and predeclared decision rule are frozen before this point.
    experiment_spec["calibrator_artifact_sha256"] = calibrator_manifest["artifact_sha256"]
    experiment_spec["calibrator_frozen_before_final_test"] = True
    write_json(run_dir / "experiment_spec_frozen.json", experiment_spec)

    final_raw, final_labels = probabilities_and_labels(contract.final_test)
    final_calibrated = loaded.transform(final_raw)
    if not np.array_equal(np.argmax(final_raw, axis=1), np.argmax(final_calibrated, axis=1)):
        raise Phase3TemperatureIntegrityError("Temperature scaling changed a final-test argmax")
    raw_final_records = [
        {
            "schema_version": 1,
            "prediction_run_id": DEFAULT_RUN_ID,
            "match_hash": record["match_hash"],
            "fixture": record["fixture"],
            "prediction_time": record["prediction_time"],
            "actual_result": record["actual_result"],
            "raw_probabilities": record["raw_probabilities"],
            "raw_prediction_hash": record["raw_prediction_hash"],
            "phase1_record_hash": record["phase1_record_hash"],
        }
        for record in contract.final_test
    ]
    calibrated_final_records = [
        _record_with_calibration(record, probabilities, calibrator_id=calibrator_id)
        for record, probabilities in zip(contract.final_test, final_calibrated, strict=True)
    ]
    write_jsonl(run_dir / "final_test_raw_predictions.jsonl", raw_final_records)
    write_jsonl(run_dir / "final_test_calibrated_predictions.jsonl", calibrated_final_records)

    raw_metrics = multiclass_metrics(final_labels, final_raw)
    calibrated_metrics = multiclass_metrics(final_labels, final_calibrated)
    final_metrics = {
        "partition": "final_test",
        "evaluation_count": 1,
        "raw": raw_metrics,
        "calibrated": calibrated_metrics,
        "accuracy_unchanged": raw_metrics["accuracy"] == calibrated_metrics["accuracy"],
    }
    comparison = paired_bootstrap(final_labels, final_raw, final_calibrated)
    decision = calibration_decision(raw_metrics, calibrated_metrics, comparison)
    write_json(run_dir / "final_test_metrics.json", final_metrics)
    write_json(run_dir / "paired_bootstrap.json", comparison)
    write_json(run_dir / "calibration_decision.json", decision)
    write_json(run_dir / "reliability_data.json", {
        "binning": raw_metrics["binning"],
        "calibration_fit": {
            "raw": calibration_metrics["raw"]["top_label_reliability"],
            "calibrated": calibration_metrics["calibrated"]["top_label_reliability"],
        },
        "final_test": {
            "raw_top_label": raw_metrics["top_label_reliability"],
            "calibrated_top_label": calibrated_metrics["top_label_reliability"],
            "raw_classwise": raw_metrics["classwise"],
            "calibrated_classwise": calibrated_metrics["classwise"],
        },
    })
    write_json(run_dir / "legacy_calibrator_inventory.json", legacy_calibrator_inventory())
    write_json(run_dir / "reproducibility.json", {
        "fit_1_temperature": calibrator.temperature,
        "fit_2_temperature": repeated.temperature,
        "temperature_exact_match": calibrator.temperature == repeated.temperature,
        "fit_identity_hash_exact_match": (
            fit_diagnostics["fit_fixture_ids_sha256"] == repeated_diagnostics["fit_fixture_ids_sha256"]
        ),
        "serialized_reload_output_max_abs_difference": float(
            np.max(np.abs(calibration_calibrated - loaded.transform(calibration_raw)))
        ),
        "raw_prediction_mutation_count": 0,
    })
    write_json(run_dir / "leakage_contract.json", {
        "fit_interface_fields": ["fixture_id", "raw_probabilities", "actual_result"],
        "fit_partition": "calibration_fit only",
        "fit_fixture_ids_sha256": row_ids_hash(item["fixture_id"] for item in fit_rows),
        "final_test_fixture_ids_sha256": contract.manifest["partitions"]["final_test"]["fixture_ids_sha256"],
        "fit_final_overlap_count": 0,
        "bookmaker_inputs": False,
        "class_order": list(CLASS_ORDER),
        "mechanical_policy": "Any non-calibration identity, final-test identity, extra fit field, bookmaker key, or reordered class mapping fails closed.",
    })

    after = protected_snapshot()
    write_json(run_dir / "protected_after_core_evaluation.json", after)
    write_json(run_dir / "protected_core_verification.json", {
        "matches_before": _protected_equal(before, after),
        "before": before,
        "after": after,
    })
    if not _protected_equal(before, after):
        raise Phase3TemperatureIntegrityError("Protected artifacts changed during Phase 3 core evaluation")

    summary = {
        "run_id": run_dir.name,
        "run_dir": _relative(run_dir),
        "temperature": calibrator.temperature,
        "decision": decision["decision"],
        "calibration_fit_count": len(contract.calibration_fit),
        "final_test_count": len(contract.final_test),
        "calibrator_sha256": calibrator_manifest["artifact_sha256"],
        "status": "CORE_EVALUATION_COMPLETE_AWAITING_TEST_FINALIZATION",
    }
    write_json(run_dir / "core_evaluation_summary.json", summary)
    return summary


def _junit_summary(path: Path) -> dict[str, Any]:
    suite = ET.parse(path).getroot().find("testsuite")
    if suite is None:
        root = ET.parse(path).getroot()
        suite = root if root.tag == "testsuite" else None
    if suite is None:
        raise Phase3TemperatureIntegrityError(f"No testsuite in {path}")
    failures = []
    for case in suite.findall("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            failures.append(f"{case.attrib.get('classname')}::{case.attrib.get('name')}")
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "collected": int(suite.attrib.get("tests", 0)),
        "failures": int(suite.attrib.get("failures", 0)),
        "errors": int(suite.attrib.get("errors", 0)),
        "skipped": int(suite.attrib.get("skipped", 0)),
        "passed": (
            int(suite.attrib.get("tests", 0))
            - int(suite.attrib.get("failures", 0))
            - int(suite.attrib.get("errors", 0))
            - int(suite.attrib.get("skipped", 0))
        ),
        "time_seconds": float(suite.attrib.get("time", 0.0)),
        "failing_node_ids": failures,
    }


def _report(run_dir: Path, test_results: Mapping[str, Any], protected: Mapping[str, Any]) -> str:
    partition = read_json(run_dir / "partition_manifest.json")
    calibrator = read_json(run_dir / "calibrator" / "pl_1x2_temperature.json")
    fit_metrics = read_json(run_dir / "calibration_fit_metrics.json")
    final_metrics = read_json(run_dir / "final_test_metrics.json")
    paired = read_json(run_dir / "paired_bootstrap.json")
    decision = read_json(run_dir / "calibration_decision.json")
    legacy = read_json(run_dir / "legacy_calibrator_inventory.json")
    raw = final_metrics["raw"]
    calibrated = final_metrics["calibrated"]
    temperature = float(calibrator["temperature"])
    direction = "softened (raw probabilities were directionally overconfident)" if temperature > 1.0 else (
        "sharpened (raw probabilities were directionally underconfident)" if temperature < 1.0 else "unchanged"
    )
    parts = partition["partitions"]
    full = test_results["full"]
    lines = [
        "# FFPS PHASE 3 — PL 1X2 TEMPERATURE CALIBRATION EVIDENCE",
        "",
        "> Evaluation-only research evidence. No production model or calibrator was promoted.",
        "",
        "## Scope and partition contract",
        "",
    ]
    for name in ("model_train", "model_validation", "calibration_fit", "final_test"):
        item = parts[name]
        lines.append(f"- {name}: {item['count']} fixtures, {item['date_min']} through {item['date_max']}, identity hash `{item['fixture_ids_sha256']}`.")
    lines.extend([
        "- All four match-hash and canonical fixture identity sets are pairwise disjoint.",
        "- Same-day source events remain excluded; all 558 reused Phase 1 predictions satisfy max source date < prediction date.",
        "",
        "## Raw prediction lineage",
        "",
        "- Frozen Phase 1 goal-ensemble + MatchSimulator replay reused without recomputation or mutation.",
        f"- Raw input SHA-256: `{partition['source']['phase1_predictions_sha256']}`.",
        "- Goal ensemble: LGBM/XGBoost 60/40; 50,000 fixture-seeded simulations; no calibrators or bookmaker inputs.",
        "",
        "## Calibrator",
        "",
        f"- ID: `{calibrator['calibrator_id']}`.",
        f"- Temperature: `{temperature:.12f}`; probabilities were {direction}.",
        "- Method: one scalar multiclass temperature, fit by NLL over unconstrained log(T), calibration-fit rows only.",
        f"- Artifact SHA-256: `{read_json(run_dir / 'calibrator' / 'calibrator_manifest.json')['artifact_sha256']}`.",
        "",
        "## Calibration-fit diagnostics",
        "",
        f"- Raw: accuracy {fit_metrics['raw']['accuracy']:.6f}, log loss {fit_metrics['raw']['log_loss']:.6f}, Brier {fit_metrics['raw']['multiclass_brier']:.6f}, top-label ECE {fit_metrics['raw']['top_label_ece']:.6f}.",
        f"- Calibrated: accuracy {fit_metrics['calibrated']['accuracy']:.6f}, log loss {fit_metrics['calibrated']['log_loss']:.6f}, Brier {fit_metrics['calibrated']['multiclass_brier']:.6f}, top-label ECE {fit_metrics['calibrated']['top_label_ece']:.6f}.",
        "",
        "## Final-test results",
        "",
        f"- Raw: accuracy {raw['accuracy']:.6f}, log loss {raw['log_loss']:.6f}, Brier {raw['multiclass_brier']:.6f}, top-label ECE {raw['top_label_ece']:.6f}, entropy {raw['average_entropy_nats']:.6f}, mean max probability {raw['mean_max_probability']:.6f}.",
        f"- Calibrated: accuracy {calibrated['accuracy']:.6f}, log loss {calibrated['log_loss']:.6f}, Brier {calibrated['multiclass_brier']:.6f}, top-label ECE {calibrated['top_label_ece']:.6f}, entropy {calibrated['average_entropy_nats']:.6f}, mean max probability {calibrated['mean_max_probability']:.6f}.",
        "- Accuracy is unchanged because positive temperature scaling preserves class ordering; this was verified on every fixture.",
        "",
        "## Paired uncertainty",
        "",
        f"- Log-loss delta: {paired['log_loss']['point_delta_calibrated_minus_raw']:.6f} [{paired['log_loss']['interval_lower']:.6f}, {paired['log_loss']['interval_upper']:.6f}].",
        f"- Brier delta: {paired['multiclass_brier']['point_delta_calibrated_minus_raw']:.6f} [{paired['multiclass_brier']['interval_lower']:.6f}, {paired['multiclass_brier']['interval_upper']:.6f}].",
        f"- Top-label ECE delta: {paired['top_label_ece']['point_delta_calibrated_minus_raw']:.6f} [{paired['top_label_ece']['interval_lower']:.6f}, {paired['top_label_ece']['interval_upper']:.6f}].",
        f"- Procedure: {paired['samples']} paired bootstrap resamples, seed {paired['seed']}, {paired['confidence_level']:.0%} percentile intervals.",
        "",
        "## Decision",
        "",
        f"**{decision['decision']}**",
        "",
        "The predeclared rule was frozen before final-test evaluation. This is not a production-promotion decision.",
        "",
        "## Legacy calibration audit",
        "",
        f"- {legacy['manifest_pointer_count']} manifest calibrator pointers and {legacy['standalone_artifact_count']} standalone calibrator files were inventoried.",
        "- Existing artifacts/workflows lack fixture-level proof that calibration-fit and final-test identities are disjoint and are classified `LEGACY_CALIBRATION_VALIDITY_UNPROVEN`.",
        "- The earlier multi-market Phase 3 directory was preserved as a legacy diagnostic outside this authorization.",
        "",
        "## Tests and protected evidence",
        "",
        f"- Focused Phase 3: {test_results['focused']['passed']} passed, {test_results['focused']['failures']} failed.",
        f"- Phase 0/1/2 regressions: {test_results['regression']['passed']} passed, {test_results['regression']['failures']} failed.",
        f"- Full suite: {full['collected']} collected; {full['passed']} passed; {full['failures']} failed; {full['skipped']} skipped; {full['errors']} errors.",
        "- The six failures exactly match the inherited quiet-mode and five stale-drift failures; no new failures exist.",
        f"- Protected before/after match: {protected['matches_before']}.",
        "",
        "## Limitations",
        "",
        "- The 2025/26 final-test rows are protocol-isolated from calibration fitting, but were previously included in Phase 1 aggregate reporting. They are not globally pristine research data.",
        "- A future post-protocol dataset is required for genuinely pristine certification evidence.",
        "- No profitability, ROI, deployment, or production-readiness claim is made.",
        "",
    ])
    return "\n".join(lines)


def finalize_evidence(
    run_dir: Path,
    *,
    focused_junit: Path,
    regression_junit: Path,
    full_junit: Path,
) -> dict[str, Any]:
    if (run_dir / "artifact_index.json").exists():
        raise Phase3TemperatureIntegrityError("Evidence bundle is already finalized and immutable")
    required = (
        "experiment_spec_frozen.json",
        "partition_manifest.json",
        "raw_live_path_predictions.jsonl",
        "calibration_fit_predictions.jsonl",
        "final_test_raw_predictions.jsonl",
        "final_test_calibrated_predictions.jsonl",
        "calibrator/pl_1x2_temperature.json",
        "calibrator/calibrator_manifest.json",
        "calibration_fit_metrics.json",
        "final_test_metrics.json",
        "paired_bootstrap.json",
        "reliability_data.json",
        "legacy_calibrator_inventory.json",
        "calibration_decision.json",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise Phase3TemperatureIntegrityError(f"Cannot finalize incomplete evidence: {missing}")
    results = {
        "focused": _junit_summary(focused_junit),
        "regression": _junit_summary(regression_junit),
        "full": _junit_summary(full_junit),
    }
    if results["focused"]["failures"] or results["focused"]["errors"]:
        raise Phase3TemperatureIntegrityError("Focused Phase 3 tests did not pass")
    if results["regression"]["failures"] or results["regression"]["errors"]:
        raise Phase3TemperatureIntegrityError("Phase 0/1/2 regression tests did not pass")
    if set(results["full"]["failing_node_ids"]) != INHERITED_FAILURES:
        raise Phase3TemperatureIntegrityError(
            f"Full-suite failures differ from inherited baseline: {results['full']['failing_node_ids']}"
        )
    write_json(run_dir / "test_results.json", results)
    write_json(run_dir / "leakage_test_evidence.json", {
        "focused_junit_sha256": results["focused"]["sha256"],
        "focused_tests_passed": results["focused"]["passed"],
        "sentinels": [
            "final-test fixture rejected by fit",
            "partition overlap rejected",
            "final-test outcome row rejected",
            "bookmaker input rejected",
            "class-order change rejected",
            "missing provenance cannot be trusted",
            "artifact hash and round-trip enforced",
        ],
        "passed": True,
    })
    before = read_json(run_dir / "protected_before.json")
    after = protected_snapshot()
    verification = {
        "matches_before": _protected_equal(before, after),
        "before": before,
        "after": after,
    }
    if not verification["matches_before"]:
        raise Phase3TemperatureIntegrityError("Protected artifacts changed before finalization")
    write_json(run_dir / "protected_after_final.json", after)
    write_json(run_dir / "protected_verification.json", verification)

    run_manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "created_at": read_json(run_dir / "experiment_spec.json")["created_at"],
        "finalized_at": utc_now(),
        "status": "complete",
        "phase": 3,
        "scope": {"league": "PL", "market": "1X2", "method": "multiclass_temperature_scaling"},
        "decision": read_json(run_dir / "calibration_decision.json")["decision"],
        "evaluation_only": True,
        "production_eligible": False,
        "production_promotion": False,
        "prior_phase_evidence_unchanged": True,
        "source_files": {
            _relative(CALIBRATOR_SOURCE): sha256_file(CALIBRATOR_SOURCE),
            _relative(RUNNER_SOURCE): sha256_file(RUNNER_SOURCE),
        },
        "source_state_at_finalization": source_state(),
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "test_contract": {
            "six_inherited_failures_only": True,
            "new_unexplained_failures": 0,
        },
        "limitations": [
            "Final-test period was previously included in Phase 1 aggregate reporting.",
            "No profitability or production-readiness claim.",
        ],
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    report = _report(run_dir, results, verification)
    (run_dir / "PHASE3_IMPLEMENTATION_REPORT.md").write_text(report, encoding="utf-8", newline="\n")

    artifacts = [
        {
            "path": _relative(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name != "artifact_index.json"
    ]
    artifact_index = {
        "schema_version": 1,
        "created_at": utc_now(),
        "artifacts": artifacts,
        "aggregate_sha256": sha256_json(artifacts),
    }
    write_json(run_dir / "artifact_index.json", artifact_index)
    return {
        "run_id": run_dir.name,
        "status": "complete",
        "decision": run_manifest["decision"],
        "artifacts": len(artifacts),
        "artifact_aggregate_sha256": artifact_index["aggregate_sha256"],
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [cast(dict[str, Any], json.loads(line)) for line in handle if line.strip()]


def verify_evidence(run_dir: Path) -> dict[str, Any]:
    index = read_json(run_dir / "artifact_index.json")
    errors: list[str] = []
    for item in index["artifacts"]:
        path = ROOT / item["path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != item["sha256"]:
            errors.append(f"artifact {item['path']}: expected={item['sha256']} actual={actual}")
    if sha256_json(index["artifacts"]) != index["aggregate_sha256"]:
        errors.append("artifact aggregate hash mismatch")
    partition = read_json(run_dir / "partition_manifest.json")
    try:
        load_calibrator_artifact(
            run_dir / "calibrator" / "pl_1x2_temperature.json",
            run_dir / "calibrator" / "calibrator_manifest.json",
            partition,
        )
    except Phase3TemperatureIntegrityError as exc:
        errors.append(str(exc))
    raw = _read_jsonl(run_dir / "final_test_raw_predictions.jsonl")
    calibrated = _read_jsonl(run_dir / "final_test_calibrated_predictions.jsonl")
    if [item["match_hash"] for item in raw] != [item["match_hash"] for item in calibrated]:
        errors.append("raw/calibrated final-test identities differ")
    for raw_item, calibrated_item in zip(raw, calibrated, strict=True):
        if raw_item["raw_probabilities"] != calibrated_item["raw_probabilities"]:
            errors.append(f"raw probability mutation: {raw_item['match_hash']}")
            break
    protected = read_json(run_dir / "protected_verification.json")
    if not protected["matches_before"]:
        errors.append("protected before/after mismatch")
    current = protected_snapshot()
    if not _protected_equal(protected["after"], current):
        errors.append("current protected identities differ from finalized evidence")
    result = {
        "verified_at": utc_now(),
        "run_id": run_dir.name,
        "passed": not errors,
        "errors": errors,
        "artifact_count": len(index["artifacts"]),
        "artifact_aggregate_sha256": index["aggregate_sha256"],
        "decision": read_json(run_dir / "calibration_decision.json")["decision"],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    finalize_parser.add_argument("--focused-junit", type=Path, required=True)
    finalize_parser.add_argument("--regression-junit", type=Path, required=True)
    finalize_parser.add_argument("--full-junit", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    if args.command == "create":
        result = create_evidence(args.run_dir.resolve())
    elif args.command == "finalize":
        result = finalize_evidence(
            args.run_dir.resolve(),
            focused_junit=args.focused_junit.resolve(),
            regression_junit=args.regression_junit.resolve(),
            full_junit=args.full_junit.resolve(),
        )
    else:
        result = verify_evidence(args.run_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())

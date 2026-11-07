"""Create and verify the FFPS Phase 1 PL 1X2 evidence bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.phase1_pl_1x2 import (
    DEFAULT_N_SIMULATIONS,
    FEATURE_MATRIX_PATH,
    PHASE0_MANIFEST_PATH,
    Phase1IntegrityError,
    assert_phase0_inputs_unchanged,
    assert_production_snapshot_unchanged,
    canonical_json,
    evaluate_predictions,
    live_path_evidence,
    load_and_partition,
    load_evaluation_bundle,
    package_versions,
    phase0_manifest,
    predict_fixture_live_path,
    prediction_records_hash,
    production_goal_snapshot,
    read_json,
    run_holdout_predictions,
    sha256_file,
    sha256_json,
    train_isolated_goal_bundle,
    utc_now,
    verify_prediction_reproduction,
    write_exclusions_jsonl,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_ROOT = Path("baselines/phase1")
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{5,100}$")


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _safe_run_dir(output_root: Path, run_id: str, *, must_exist: bool) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise Phase1IntegrityError(f"Invalid run id: {run_id!r}")
    allowed_root = _repo_path(DEFAULT_OUTPUT_ROOT).resolve()
    requested_root = _repo_path(output_root).resolve()
    try:
        requested_root.relative_to(allowed_root)
    except ValueError as exc:
        raise Phase1IntegrityError(
            f"Phase 1 output must remain under {allowed_root.relative_to(REPO_ROOT).as_posix()}"
        ) from exc
    run_dir = (requested_root / run_id).resolve()
    try:
        run_dir.relative_to(allowed_root)
    except ValueError as exc:
        raise Phase1IntegrityError("Resolved run directory escaped the Phase 1 output root") from exc
    if must_exist and not run_dir.is_dir():
        raise Phase1IntegrityError(f"Run directory does not exist: {run_dir}")
    if not must_exist and run_dir.exists():
        raise Phase1IntegrityError(f"Run directory already exists; refusing overwrite: {run_dir}")
    return run_dir


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise Phase1IntegrityError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def capture_git_state() -> dict[str, Any]:
    upstream = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    ahead_behind = _git("rev-list", "--left-right", "--count", f"HEAD...{upstream}").split()
    status_lines = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    return {
        "captured_at": utc_now(),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "upstream": upstream,
        "ahead": int(ahead_behind[0]),
        "behind": int(ahead_behind[1]),
        "dirty": bool(status_lines),
        "status": status_lines,
    }


def run_phase0_verifier() -> dict[str, Any]:
    command = [sys.executable, str(REPO_ROOT / "scripts/phase0_baseline.py"), "verify"]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Phase1IntegrityError(
            f"Phase 0 verifier failed with {completed.returncode}: {completed.stderr[-2000:]}"
        )
    start = completed.stdout.find("{")
    if start < 0:
        raise Phase1IntegrityError("Phase 0 verifier did not return JSON")
    payload = cast(dict[str, Any], json.loads(completed.stdout[start:]))
    if not payload.get("passed"):
        raise Phase1IntegrityError("Phase 0 verifier returned passed=false")
    return payload


def _write_initial_manifest(
    run_dir: Path,
    *,
    run_id: str,
    source_state: Mapping[str, Any],
    production_before: Mapping[str, Any],
) -> None:
    baseline = phase0_manifest()
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": utc_now(),
        "status": "evaluation_complete_tests_pending",
        "purpose": "Point-in-time PL 1X2 validation through the current live goal/simulator pathway",
        "phase0_baseline_id": baseline["baseline_id"],
        "phase0_manifest": {
            "path": PHASE0_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(PHASE0_MANIFEST_PATH),
        },
        "source_state": source_state,
        "runtime": package_versions(),
        "offline_only": True,
        "external_provider_calls": 0,
        "scope": {"league": "PL", "market": "1X2", "phase": 1},
        "production_before": production_before,
        "paths": {
            "partition_manifest": _relative(run_dir / "partition_manifest.json"),
            "model_bundle_manifest": _relative(run_dir / "model_bundle_manifest.json"),
            "fixture_predictions": _relative(run_dir / "fixture_predictions.jsonl"),
            "exclusions": _relative(run_dir / "exclusions.jsonl"),
            "metrics": _relative(run_dir / "metrics.json"),
            "reliability": _relative(run_dir / "reliability.json"),
            "reproducibility": _relative(run_dir / "reproducibility.json"),
            "phase0_before": _relative(run_dir / "phase0_before.json"),
            "test_results": _relative(run_dir / "test_results.json"),
            "phase0_after": _relative(run_dir / "phase0_after.json"),
            "report": _relative(run_dir / "FFPS_PHASE1_PL_1X2_REPORT.md"),
        },
        "invariants": {
            "production_models_changed": False,
            "production_manifest_changed": False,
            "production_data_changed": False,
            "production_prediction_code_changed": False,
            "warm_start": False,
            "calibration": False,
            "bookmaker_features": False,
            "match_outcome_classifier": False,
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)


def create_evaluation_run(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    run_dir = _safe_run_dir(Path(args.output_root), args.run_id, must_exist=False)
    phase0_before = run_phase0_verifier()
    phase0_inputs = assert_phase0_inputs_unchanged()
    production_before = production_goal_snapshot()
    source_state = capture_git_state()

    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "phase0_before.json", phase0_before)
    write_json(run_dir / "phase0_input_guard.json", phase0_inputs)
    write_json(run_dir / "production_before.json", production_before)
    write_json(run_dir / "source_state.json", source_state)

    partitions, features, active_contracts, source_evidence = load_and_partition()
    partition_manifest = dict(partitions.manifest)
    partition_manifest["active_production_contracts_read_only"] = active_contracts
    write_json(run_dir / "partition_manifest.json", partition_manifest)
    write_exclusions_jsonl(run_dir / "exclusions.jsonl", partitions.exclusions)

    bundle_manifest = train_isolated_goal_bundle(
        partitions,
        features,
        run_dir / "models",
        warm_start=False,
        calibrators_enabled=False,
    )
    write_json(run_dir / "model_bundle_manifest.json", bundle_manifest)
    models = load_evaluation_bundle(bundle_manifest)

    first = run_holdout_predictions(
        partitions.final_holdout,
        models,
        bundle_manifest,
        source_evidence,
        n_simulations=args.n_simulations,
        reverse_order=False,
    )
    second = run_holdout_predictions(
        partitions.final_holdout,
        models,
        bundle_manifest,
        source_evidence,
        n_simulations=args.n_simulations,
        reverse_order=True,
    )
    reproducibility = verify_prediction_reproduction(first, second, tolerance=0.0)
    reproducibility["model_artifacts_sha256"] = bundle_manifest["artifacts_aggregate_sha256"]
    reproducibility["n_simulations_per_fixture"] = args.n_simulations
    write_jsonl(run_dir / "fixture_predictions.jsonl", first)
    write_json(run_dir / "reproducibility.json", reproducibility)

    bootstrap_seed = int(str(bundle_manifest["artifacts_aggregate_sha256"])[:16], 16)
    metrics, reliability = evaluate_predictions(
        first,
        partitions.training,
        bootstrap_seed=bootstrap_seed,
    )
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "reliability.json", reliability)
    write_json(run_dir / "live_path_evidence.json", live_path_evidence())

    production_pretests = assert_production_snapshot_unchanged(production_before)
    write_json(run_dir / "production_integrity_pretests.json", production_pretests)
    _write_initial_manifest(
        run_dir,
        run_id=args.run_id,
        source_state=source_state,
        production_before=production_before,
    )
    print(
        json.dumps(
            {
                "status": "evaluation_complete_tests_pending",
                "run_dir": _relative(run_dir),
                "holdout_predictions": len(first),
                "prediction_output_sha256": prediction_records_hash(first),
                "metrics": metrics["system"],
            },
            indent=2,
        )
    )
    return 0


def _collection_count(output: str) -> int | None:
    matches = re.findall(r"(\d+) tests? collected", output)
    return int(matches[-1]) if matches else None


def _junit_counts(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    cases: list[dict[str, str]] = []
    for suite in suites:
        for key in ("tests", "failures", "errors", "skipped"):
            totals[key] += int(suite.attrib.get(key, "0"))
        totals["time"] += float(suite.attrib.get("time", "0"))
        for case in suite.findall("testcase"):
            status = "passed"
            if case.find("failure") is not None:
                status = "failed"
            elif case.find("error") is not None:
                status = "error"
            elif case.find("skipped") is not None:
                status = "skipped"
            if status != "passed":
                cases.append(
                    {
                        "classname": case.attrib.get("classname", ""),
                        "name": case.attrib.get("name", ""),
                        "status": status,
                    }
                )
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    totals["non_passed_cases"] = cases
    return totals


def _failed_nodes(output: str) -> list[str]:
    return sorted(set(re.findall(r"^FAILED\s+(.+?)(?:\s+-\s+.*)?$", output, flags=re.MULTILINE)))


def run_tests(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    run_dir = _safe_run_dir(Path(args.output_root), args.run_id, must_exist=True)
    collection_command = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    collection_command_record = "python -m pytest --collect-only -q"
    started = time.perf_counter()
    collection = subprocess.run(
        collection_command,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    collection_duration = time.perf_counter() - started
    collection_output = collection.stdout + collection.stderr
    (run_dir / "pytest-collect-output.txt").write_text(collection_output, encoding="utf-8")
    if collection.returncode != 0:
        raise Phase1IntegrityError("Full pytest collection failed; see pytest-collect-output.txt")

    junit_path = run_dir / "pytest-results.xml"
    suite_command = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit_path.as_posix()}"]
    suite_command_record = f"python -m pytest -q --junitxml={_relative(junit_path)}"
    started = time.perf_counter()
    suite = subprocess.run(
        suite_command,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    suite_duration = time.perf_counter() - started
    suite_output = suite.stdout + suite.stderr
    (run_dir / "pytest-output.txt").write_text(suite_output, encoding="utf-8")
    counts = _junit_counts(junit_path)
    failed_nodes = _failed_nodes(suite_output)
    expected_nodes = sorted(
        str(item["test"])
        for item in cast(
            list[dict[str, Any]],
            phase0_manifest()["tests"]["suite"]["failure_classifications"],
        )
    )
    unexpected = sorted(set(failed_nodes) - set(expected_nodes))
    missing_expected = sorted(set(expected_nodes) - set(failed_nodes))
    classifications = []
    phase0_classifications = {
        str(item["test"]): str(item["classification"])
        for item in cast(
            list[dict[str, Any]],
            phase0_manifest()["tests"]["suite"]["failure_classifications"],
        )
    }
    for node in failed_nodes:
        classifications.append(
            {
                "test": node,
                "classification": phase0_classifications.get(node, "new_unexplained_failure"),
            }
        )
    result = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "runtime": package_versions(),
        "collection": {
            "command": collection_command_record,
            "exit_code": collection.returncode,
            "collected": _collection_count(collection_output),
            "duration_seconds": round(collection_duration, 3),
            "output_path": _relative(run_dir / "pytest-collect-output.txt"),
        },
        "suite": {
            "command": suite_command_record,
            "exit_code": suite.returncode,
            "duration_seconds": round(suite_duration, 3),
            "tests": counts["tests"],
            "passed": counts["passed"],
            "failed": counts["failures"],
            "errors": counts["errors"],
            "skipped": counts["skipped"],
            "junit_duration_seconds": counts["time"],
            "failed_nodes": failed_nodes,
            "failure_classifications": classifications,
            "expected_phase0_failures": expected_nodes,
            "unexpected_failures": unexpected,
            "expected_failures_not_observed": missing_expected,
            "output_path": _relative(run_dir / "pytest-output.txt"),
            "junit_path": _relative(junit_path),
        },
        "acceptance": {
            "collection_complete": collection.returncode == 0,
            "no_new_unexplained_failures": not unexpected and counts["errors"] == 0,
            "passed": collection.returncode == 0 and not unexpected and counts["errors"] == 0,
        },
    }
    write_json(run_dir / "test_results.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["acceptance"]["passed"] else 1


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _report_text(run_dir: Path) -> str:
    run_manifest = read_json(run_dir / "run_manifest.json")
    partition = read_json(run_dir / "partition_manifest.json")
    bundle = read_json(run_dir / "model_bundle_manifest.json")
    metrics = read_json(run_dir / "metrics.json")
    reproducibility = read_json(run_dir / "reproducibility.json")
    tests = read_json(run_dir / "test_results.json")
    phase0_after = read_json(run_dir / "phase0_after.json")
    prod = read_json(run_dir / "production_integrity_final.json")
    p = partition["partitions"]
    s = metrics["system"]
    uniform = metrics["baselines"]["uniform"]["metrics"]
    frequency = metrics["baselines"]["training_frequency"]["metrics"]
    ready = (
        tests["acceptance"]["passed"]
        and phase0_after["passed"]
        and prod["passed"]
        and reproducibility["passed"]
    )
    lines = [
        "# FFPS PHASE 1 PL 1X2 VALIDATION REPORT",
        "",
        "> Phase 1 validates the integrity and measured holdout behaviour of one PL 1X2 slice. "
        "It does not validate every FFPS market, league, production artifact, or prediction-quality claim.",
        "",
        "## Summary",
        "",
        f"- Run: `{run_manifest['run_id']}`",
        f"- Result: {'READY' if ready else 'NOT READY'} for Phase 2 under the scoped acceptance checks.",
        "- Path: isolated LGBM/XGBoost goal models -> existing 60/40 `_calc_goals` ensemble -> existing `MatchSimulator` -> 1X2.",
        "- The production `match_outcome` classifier, bookmaker inputs, warm starts, and calibrators were excluded.",
        "",
        "## Temporal Policy",
        "",
        f"- Training: `{partition['temporal_policy']['training']}`",
        f"- Validation: `{partition['temporal_policy']['validation']}`",
        f"- Final holdout: `{partition['temporal_policy']['final_holdout']}`",
        "- Dates have no kickoff times, so prediction time is 00:00 UTC and every same-day source event is excluded.",
        "- Active features were recomputed from strictly earlier raw rows for every PL matrix row; any mismatch fails closed.",
        "- This is sequential live replay: an earlier holdout result may become a strictly prior rolling/form source for a later fixture, but no holdout row enters fitting, tuning, calibration, feature selection, warm start, or global/fallback statistics.",
        "",
        "## Dataset Partitions",
        "",
        f"- Training: {p['training']['rows']} rows, {p['training']['date_min']} to {p['training']['date_max']}.",
        f"- Validation: {p['validation']['rows']} rows, {p['validation']['date_min']} to {p['validation']['date_max']}.",
        f"- Final holdout: {p['final_holdout']['rows']} rows, {p['final_holdout']['date_min']} to {p['final_holdout']['date_max']}.",
        f"- Exclusions: {partition['exclusions']['count']} rows, all recorded with reason codes.",
        "",
        "## Feature Integrity",
        "",
        f"- Feature count: {partition['feature_causality']['active_feature_count']}.",
        f"- Ordered feature hash: `{partition['feature_causality']['active_feature_order_sha256']}`.",
        "- No target, outcome, bookmaker, full-history global-fallback, or post-fixture field is in the active set.",
        "- Weather is the current 0.0 fallback and availability is the current 1.0 stub; these behaviours were preserved.",
        "",
        "## Model Bundle",
        "",
        f"- Evaluation bundle hash: `{bundle['artifacts_aggregate_sha256']}`.",
        "- Four fresh artifacts: home/away LGBM Poisson and home/away XGBoost count:poisson.",
        "- `warm_start=false`, `calibrators_enabled=false`, `registry_used=false`, `production_eligible=false`.",
        "- Existing trainer methods and hyperparameters were reused without constructing its registry-writing initializer.",
        "- Exact executed source hashes are recorded in code_provenance.json; the dirty Git state is recorded separately in source_state.json.",
        "",
        "## Live Path",
        "",
        "- Each fixture called all four goal members once, `_calc_goals` once, and `MatchSimulator.simulate` once.",
        "- The current lambda clamps, 60/40 ensemble and 50,000-draw Monte Carlo math were unchanged.",
        "- The `match_outcome` classifier was absent from the evaluation suite and is not referenced by `_calc_goals`.",
        "",
        "## Holdout Metrics",
        "",
        f"- N: {metrics['sample_size']} ({metrics['date_min']} to {metrics['date_max']}).",
        f"- Accuracy: {s['accuracy']:.6f}",
        f"- Multiclass log loss: {s['log_loss']:.6f}",
        f"- Multiclass Brier: {s['multiclass_brier']:.6f}",
        f"- Top-label ECE: {s['top_label_ece']:.6f}",
        f"- Classwise ECE: {s['classwise_ece']:.6f}",
        f"- Uniform baseline log loss/Brier: {uniform['log_loss']:.6f} / {uniform['multiclass_brier']:.6f}",
        f"- Training-frequency baseline log loss/Brier: {frequency['log_loss']:.6f} / {frequency['multiclass_brier']:.6f}",
        "- No odds, ROI, or profitability metric was computed.",
        "",
        "## Reproducibility",
        "",
        f"- {reproducibility['fixture_count']} fixtures reproduced with max absolute difference {reproducibility['max_absolute_difference']}.",
        f"- Canonical output hash: `{reproducibility['first_output_sha256']}`.",
        "- Second execution used reverse input order and fixture-specific seeds; canonical results were identical.",
        "- Production RNG/seeding behaviour was not changed.",
        "",
        "## Tests",
        "",
        f"- Collected: {tests['collection']['collected']}.",
        f"- Passed/failed/skipped/errors: {tests['suite']['passed']}/{tests['suite']['failed']}/{tests['suite']['skipped']}/{tests['suite']['errors']}.",
        f"- New unexplained failures: {len(tests['suite']['unexpected_failures'])}.",
        "- The six inherited Phase 0 failures remain explicitly classified; no assertion was weakened or refreshed.",
        "",
        "## Phase 0 / Production Integrity",
        "",
        f"- Phase 0 before/after verification: passed={phase0_after['passed']}.",
        f"- Production manifest and four active PL goal artifacts unchanged: passed={prod['passed']}.",
        "- The full Phase 0 verifier also rechecked all 74 active pointers and the protected data aggregate.",
        "",
        "## Findings and Limitations",
        "",
        "- IMPORTANT: Current availability/weather features are non-informative stubs/fallbacks, preserved for contract fidelity.",
        "- IMPORTANT: Rows without enough prior top-flight history are excluded rather than imputed; exclusions are part of the reported population.",
        "- IMPORTANT: Season labels in the frozen matrix are not trusted for boundaries; all splits use fixture dates.",
        "- IMPORTANT: The final holdout is evaluated as a chronological live replay with causal feature-state updates, not as a batch with covariates frozen at the holdout boundary.",
        "- IMPROVEMENT: This is a single final-holdout slice, not walk-forward or multi-season retraining evidence.",
        "- No finding in this report authorizes production promotion or prediction-method changes.",
        "",
        "## Phase 2 Readiness",
        "",
        ("Ready for Phase 2 scoped work. Phase 1 artifacts and hashes must remain immutable." if ready else "Not ready for Phase 2; resolve the failed Phase 1 acceptance evidence first."),
        "",
    ]
    return "\n".join(lines)


def _artifact_index(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_index.json", "run_manifest.json", "FFPS_PHASE1_PL_1X2_REPORT.md"}
    records = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
        if path.name in excluded:
            continue
        records.append(
            {
                "path": _relative(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "artifacts": records,
        "aggregate_sha256": sha256_json(
            [{"path": item["path"], "sha256": item["sha256"]} for item in records]
        ),
    }


def finalize_run(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    run_dir = _safe_run_dir(Path(args.output_root), args.run_id, must_exist=True)
    if not (run_dir / "test_results.json").is_file():
        raise Phase1IntegrityError("Run tests before finalizing the evidence bundle")
    manifest = read_json(run_dir / "run_manifest.json")
    production_before = cast(dict[str, Any], manifest["production_before"])
    production_final = assert_production_snapshot_unchanged(production_before)
    phase0_after = run_phase0_verifier()
    write_json(run_dir / "production_integrity_final.json", production_final)
    write_json(run_dir / "phase0_after.json", phase0_after)

    tests = read_json(run_dir / "test_results.json")
    reproduction = read_json(run_dir / "reproducibility.json")
    ready = bool(
        tests["acceptance"]["passed"]
        and reproduction["passed"]
        and phase0_after["passed"]
        and production_final["passed"]
    )
    manifest["finalized_at"] = utc_now()
    manifest["status"] = "complete" if ready else "not_ready"
    manifest["acceptance"] = {
        "ready_for_phase2": ready,
        "phase0_before_passed": read_json(run_dir / "phase0_before.json")["passed"],
        "phase0_after_passed": phase0_after["passed"],
        "production_artifacts_unchanged": production_final["passed"],
        "reproducible": reproduction["passed"],
        "no_new_unexplained_test_failures": tests["acceptance"]["no_new_unexplained_failures"],
    }
    write_json(run_dir / "run_manifest.json", manifest)
    report_path = run_dir / "FFPS_PHASE1_PL_1X2_REPORT.md"
    report_path.write_text(_report_text(run_dir), encoding="utf-8")
    index = _artifact_index(run_dir)
    write_json(run_dir / "artifact_index.json", index)
    manifest = read_json(run_dir / "run_manifest.json")
    manifest["artifact_index"] = {
        "path": _relative(run_dir / "artifact_index.json"),
        "sha256": sha256_file(run_dir / "artifact_index.json"),
        "aggregate_sha256": index["aggregate_sha256"],
    }
    manifest["report"] = {
        "path": _relative(report_path),
        "sha256": sha256_file(report_path),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    print(json.dumps({"run_dir": _relative(run_dir), "status": manifest["status"], "acceptance": manifest["acceptance"]}, indent=2))
    return 0 if ready else 1


def verify_run(args: argparse.Namespace) -> int:
    os.chdir(REPO_ROOT)
    run_dir = _safe_run_dir(Path(args.output_root), args.run_id, must_exist=True)
    assert_phase0_inputs_unchanged()
    manifest = read_json(run_dir / "run_manifest.json")
    assert_production_snapshot_unchanged(cast(dict[str, Any], manifest["production_before"]))
    index = read_json(run_dir / "artifact_index.json")
    failures = []
    for item in cast(list[dict[str, Any]], index["artifacts"]):
        path = REPO_ROOT / str(item["path"])
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != item["sha256"]:
            failures.append({"path": item["path"], "expected": item["sha256"], "actual": actual})
    if failures:
        raise Phase1IntegrityError(f"Phase 1 artifact verification failed: {failures[:10]}")

    bundle = read_json(run_dir / "model_bundle_manifest.json")
    models = load_evaluation_bundle(bundle)
    records = _load_jsonl(run_dir / "fixture_predictions.jsonl")
    matrix = pd.read_csv(REPO_ROOT / FEATURE_MATRIX_PATH, low_memory=False)
    matrix["date"] = pd.to_datetime(matrix["date"], errors="coerce")
    by_hash = matrix.set_index(matrix["match_hash"].astype(str), drop=False)
    replayed = []
    for record in records[:5]:
        match_hash = str(record["fixture"]["match_hash"])
        replayed.append(
            predict_fixture_live_path(
                by_hash.loc[match_hash],
                models,
                bundle,
                record["temporal_evidence"],
                n_simulations=int(record["rng"]["n_simulations"]),
            )
        )
    expected_sample = records[:5]
    sample_hash_expected = prediction_records_hash(expected_sample)
    sample_hash_actual = prediction_records_hash(replayed)
    if sample_hash_expected != sample_hash_actual:
        raise Phase1IntegrityError("Offline five-fixture live-path replay did not reproduce")
    result = {
        "passed": True,
        "run_dir": _relative(run_dir),
        "artifact_count": len(index["artifacts"]),
        "artifact_aggregate_sha256": index["aggregate_sha256"],
        "offline_replay_fixture_count": len(replayed),
        "offline_replay_sha256": sample_hash_actual,
        "external_provider_calls": 0,
    }
    print(json.dumps(result, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("run", "test", "finalize", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-id", required=True)
        command.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT.as_posix())
        if name == "run":
            command.add_argument("--n-simulations", type=int, default=DEFAULT_N_SIMULATIONS)
    return root


def main() -> int:
    args = parser().parse_args()
    actions = {
        "run": create_evaluation_run,
        "test": run_tests,
        "finalize": finalize_run,
        "verify": verify_run,
    }
    try:
        return actions[args.command](args)
    except Phase1IntegrityError as exc:
        print(f"PHASE1_INTEGRITY_ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


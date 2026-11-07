"""Prepare, freeze, evaluate, and verify the FFPS Phase 3 PL experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.phase1_pl_1x2 import json_safe, multiclass_metrics, package_versions, reliability_tables, row_ids_hash, sha256_file, sha256_json, utc_now, write_json
from src.evaluation.phase3_calibration import (
    BINARY_MARKETS, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, HOOK_MARKETS, MODEL_SPECS,
    CausalContextBuilder, Phase3IntegrityError, binary_metrics, confidence_baselines,
    derive_validation_hce, fit_evaluation_calibrator, git_source_state, load_base_bundle,
    load_evaluation_calibrator, load_phase3_partitions, paired_bootstrap, protected_snapshot,
    run_period_predictions, train_isolated_base_bundle, write_jsonl,
)
from src.simulation.match_simulator import DEFAULT_N_SIMULATIONS


CODE_PATHS = (Path("src/evaluation/phase3_calibration.py"), Path("scripts/run_phase3_calibration.py"))


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _selection_manifest(base: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(base)
    value["artifacts_aggregate_sha256"] = sha256_json({"selection": "training-only fixed models"})
    return value


def _set_selection_metadata(models: Mapping[str, Any], features: Sequence[str]) -> None:
    for role, target, family, objective in MODEL_SPECS:
        model_type = "lgbm_regressor_poisson" if family == "lgbm" else (
            "xgb_regressor_count_poisson" if objective == "count:poisson" else "xgb_regressor_tweedie"
        )
        models[role].meta = {"name": role, "version": "phase3-selection-only-v1", "type": model_type, "target": target, "league": "PL", "features": list(features), "calibrator_filename": None, "metrics": {"calibration_score": 0.05, "source": "evaluation-only guard metadata"}}


def _calibration_arrays(records: Sequence[Mapping[str, Any]], market: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = [str(cast(Mapping[str, Any], item["fixture"])["match_hash"]) for item in records]
    p = np.asarray([float(cast(Mapping[str, Any], item["probability_stages"])[market]["p_pre_calibration_hook"]) for item in records])
    y = np.asarray([int(cast(Mapping[str, Any], item["actuals"])[market]) for item in records])
    return ids, p, y


def prepare(run_dir: Path) -> None:
    if run_dir.exists():
        raise Phase3IntegrityError(f"Refusing to mutate existing Phase 3 run: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "protected_before.json", protected_snapshot())
    partitions, features, contracts = load_phase3_partitions(reveal_final_outcomes=False)
    write_json(run_dir / "partition_manifest.json", partitions.manifest)
    selection_models, _, bundle = train_isolated_base_bundle(partitions, features, run_dir / "models")
    write_json(run_dir / "base_model_bundle.json", bundle)
    _set_selection_metadata(selection_models, features)
    context = CausalContextBuilder()
    train_baselines = confidence_baselines(partitions.training)
    validation_records = run_period_predictions(
        partitions.validation, selection_models, _selection_manifest(bundle), context,
        baselines=train_baselines, hce_buckets={},
    )
    write_jsonl(run_dir / "validation_live_predictions.jsonl", validation_records)
    hce = derive_validation_hce(validation_records)
    write_json(run_dir / "validation_confidence_policy.json", {"baselines": train_baselines, "hce_buckets": hce, "source": "validation predictions from training-only fixed models"})
    final_models = load_base_bundle(bundle)
    refit = pd.concat([partitions.training, partitions.validation], ignore_index=True)
    calibration_records = run_period_predictions(
        partitions.calibration, final_models, bundle, context,
        baselines=confidence_baselines(refit), hce_buckets=hce,
    )
    calibration_path = run_dir / "calibration_period_predictions.jsonl"
    write_jsonl(calibration_path, calibration_records)
    calibration_hash = sha256_file(calibration_path)
    calibrators = []
    for market in HOOK_MARKETS:
        ids, probability, labels = _calibration_arrays(calibration_records, market)
        calibrators.append(fit_evaluation_calibrator(
            market=market, fixture_ids=ids, probability=probability, labels=labels,
            base_bundle_sha256=str(bundle["artifacts_aggregate_sha256"]),
            calibration_predictions_sha256=calibration_hash,
            output_path=run_dir / "calibrators" / f"{market}.pkl",
        ))
    write_json(run_dir / "calibrator_manifest.json", {"schema_version": 1, "artifacts": calibrators})
    spec = {
        "schema_version": 1, "phase": 3, "status": "FROZEN_BEFORE_FINAL_TEST", "frozen_at": utc_now(),
        "league": "PL", "partition_manifest_sha256": sha256_file(run_dir / "partition_manifest.json"),
        "base_bundle_sha256": str(bundle["artifacts_aggregate_sha256"]),
        "base_bundle_manifest_sha256": sha256_file(run_dir / "base_model_bundle.json"),
        "calibration_predictions_sha256": calibration_hash,
        "calibrator_manifest_sha256": sha256_file(run_dir / "calibrator_manifest.json"),
        "validation_confidence_policy_sha256": sha256_file(run_dir / "validation_confidence_policy.json"),
        "feature_order": list(features), "feature_order_sha256": sha256_json(list(features)),
        "feature_contracts_sha256": sha256_json(contracts),
        "partitions": partitions.manifest["temporal_policy"],
        "partition_counts": {name: len(getattr(partitions, name)) for name in ("training", "validation", "calibration", "final_test")},
        "final_test_row_ids_sha256": partitions.manifest["partitions"]["final_test"]["match_hashes_sha256"],
        "final_test_outcomes_accessed": False,
        "base_fit_policy": "fixed current trainer family/config; select on training/validation; refit training+validation; no calibration/final rows",
        "calibration_policy": "chronological 70% development/30% method selection within calibration; minimum N=200, unique=20, each class=20; lowest ECE; refit full calibration",
        "calibrator_hooks": list(HOOK_MARKETS), "no_calibrator_contract": ["1x2", "cards_o25", "cards_u45", "cards_u55"],
        "probability_path": "unchanged _calculate_probabilities_v2 with causal point-in-time H2H/offset/confidence dependencies",
        "simulation": {"count": DEFAULT_N_SIMULATIONS, "fixture_hash_seeded": True, "rl_weights": None},
        "metrics": {"binary": ["brier", "log_loss", "ece"], "multiclass": ["accuracy", "log_loss", "multiclass_brier", "top_label_ece"], "bins": 10},
        "paired_bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED, "interval": 0.95},
        "acceptance_rule": "SUPPORTED only when delta Brier and log-loss are negative and Brier 95% upper bound < 0; both point deltas nonnegative => NOT SUPPORTED; otherwise INSUFFICIENT EVIDENCE",
        "baselines": ["training+validation event frequency", "0.5 binary", "uniform 1/3", "training+validation 1X2 class frequency"],
        "bookmaker_inputs": False, "profitability_metrics": False, "threshold_optimization": False,
        "code_state": git_source_state(), "code_files": {path.as_posix(): sha256_file(path) for path in CODE_PATHS},
        "environment": package_versions(), "protected_before_sha256": sha256_file(run_dir / "protected_before.json"),
    }
    write_json(run_dir / "experiment_spec.json", spec)
    _write_text(run_dir / "experiment_spec.sha256", sha256_file(run_dir / "experiment_spec.json") + "\n")


def verify_frozen(run_dir: Path) -> dict[str, Any]:
    spec_path = run_dir / "experiment_spec.json"
    hash_path = run_dir / "experiment_spec.sha256"
    if not spec_path.is_file() or not hash_path.is_file():
        raise Phase3IntegrityError("Final evaluation refused: experiment specification is not frozen")
    expected = hash_path.read_text(encoding="utf-8").strip()
    if sha256_file(spec_path) != expected:
        raise Phase3IntegrityError("Final evaluation refused: experiment specification hash mismatch")
    spec = _read_json(spec_path)
    for path_text, digest in cast(Mapping[str, str], spec["code_files"]).items():
        if sha256_file(Path(path_text)) != digest:
            raise Phase3IntegrityError(f"Final evaluation refused: frozen code changed: {path_text}")
    for filename, key in (("partition_manifest.json", "partition_manifest_sha256"), ("base_model_bundle.json", "base_bundle_manifest_sha256"), ("calibration_period_predictions.jsonl", "calibration_predictions_sha256"), ("calibrator_manifest.json", "calibrator_manifest_sha256"), ("validation_confidence_policy.json", "validation_confidence_policy_sha256")):
        if sha256_file(run_dir / filename) != spec[key]:
            raise Phase3IntegrityError(f"Final evaluation refused: frozen artifact changed: {filename}")
    return spec


def _loaded_calibrators(run_dir: Path, spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    manifest = _read_json(run_dir / "calibrator_manifest.json")
    loaded: dict[str, dict[str, Any]] = {}
    for item in cast(Sequence[Mapping[str, Any]], manifest["artifacts"]):
        if item["status"] != "FITTED_EVALUATION_ONLY":
            continue
        loaded[str(item["market"])] = load_evaluation_calibrator(
            item,
            expected_bundle_sha256=str(spec["base_bundle_sha256"]),
            expected_calibration_predictions_sha256=str(spec["calibration_predictions_sha256"]),
        )
    return loaded


def _assert_paired_path(raw: Mapping[str, Any], calibrated: Mapping[str, Any]) -> None:
    for key in ("fixture", "actuals", "goal_lambdas", "count_means", "rng", "temporal_evidence", "feature_values_sha256", "model_call_hashes"):
        if raw[key] != calibrated[key]:
            raise Phase3IntegrityError(f"Raw/calibrated live paths differ outside calibration: {key}")
    raw_stages = cast(Mapping[str, Mapping[str, float]], raw["probability_stages"])
    cal_stages = cast(Mapping[str, Mapping[str, float]], calibrated["probability_stages"])
    for market in BINARY_MARKETS:
        if raw_stages[market]["p_raw_distribution_or_simulation"] != cal_stages[market]["p_raw_distribution_or_simulation"]:
            raise Phase3IntegrityError(f"Raw distribution changed in calibrated path: {market}")


def _multiclass_report(records: Sequence[Mapping[str, Any]], refit: pd.DataFrame) -> dict[str, Any]:
    from src.evaluation.phase1_pl_1x2 import bootstrap_confidence_intervals

    mapping = {"H": 0, "D": 1, "A": 2}
    labels = np.asarray([mapping[str(cast(Mapping[str, Any], item["actuals"])["1x2"])] for item in records], dtype=int)
    probabilities = np.asarray([[float(cast(Mapping[str, Any], item["one_x_two"])[key]) for key in ("H", "D", "A")] for item in records])
    refit_labels = np.where(refit["home_score"] > refit["away_score"], 0, np.where(refit["home_score"] < refit["away_score"], 2, 1))
    frequencies = np.bincount(refit_labels, minlength=3).astype(float) / len(refit_labels)
    uniform = np.tile(np.asarray([1 / 3, 1 / 3, 1 / 3]), (len(labels), 1))
    historical = np.tile(frequencies, (len(labels), 1))
    return {
        "sample_size": len(labels), "live_path": multiclass_metrics(labels, probabilities),
        "reliability": reliability_tables(labels, probabilities),
        "bootstrap": bootstrap_confidence_intervals(labels, probabilities, seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES),
        "baselines": {"uniform_one_third": multiclass_metrics(labels, uniform), "training_validation_class_frequency": {"frequencies": {key: float(frequencies[index]) for index, key in enumerate(("H", "D", "A"))}, "metrics": multiclass_metrics(labels, historical)}},
    }


def _binary_reports(
    raw_records: Sequence[Mapping[str, Any]], calibrated_records: Sequence[Mapping[str, Any]], refit: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from src.evaluation.phase3_calibration import _event_labels

    reports: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    fixture_ids = [str(cast(Mapping[str, Any], item["fixture"])["match_hash"]) for item in raw_records]
    calibrated_markets = {"goals_u25", "goals_o25", "goals_u35", "goals_btts_yes", "goals_btts_no", "corners_o75", "corners_u11_5"}
    for market in BINARY_MARKETS:
        y = np.asarray([int(cast(Mapping[str, Any], item["actuals"])[market]) for item in raw_records])
        raw = np.asarray([float(cast(Mapping[str, Any], item["probability_stages"])[market]["p_final_live"]) for item in raw_records])
        event_frequency = float(_event_labels(refit, market).mean())
        reports[market] = {
            "raw": binary_metrics(y, raw),
            "baselines": {
                "training_validation_event_frequency": {"probability": event_frequency, "metrics": binary_metrics(y, np.full(len(y), event_frequency))},
                "constant_0_5": binary_metrics(y, np.full(len(y), 0.5)),
            },
        }
        if market in calibrated_markets:
            calibrated = np.asarray([float(cast(Mapping[str, Any], item["probability_stages"])[market]["p_final_live"]) for item in calibrated_records])
            reports[market]["calibrated"] = binary_metrics(y, calibrated)
            comparison = paired_bootstrap(fixture_ids, y, raw, calibrated)
            deltas = comparison["deltas"]
            if deltas["brier"]["point"] < 0 and deltas["log_loss"]["point"] < 0 and deltas["brier"]["upper"] < 0:
                classification = "CALIBRATION SUPPORTED"
            elif deltas["brier"]["point"] >= 0 and deltas["log_loss"]["point"] >= 0:
                classification = "CALIBRATION NOT SUPPORTED"
            else:
                classification = "INSUFFICIENT EVIDENCE"
            paired[market] = {**comparison, "classification": classification}
    return reports, paired


def _count_diagnostics(records: Sequence[Mapping[str, Any]], refit: pd.DataFrame) -> dict[str, Any]:
    historical_corner_max = float((refit["home_corners"] + refit["away_corners"]).max())
    tail = [
        {"fixture": item["fixture"], "expected_corners": item["count_means"]["corner_total"], "observed_corners": item["actuals"]["total_corners"]}
        for item in records if float(cast(Mapping[str, Any], item["count_means"])["corner_total"]) > historical_corner_max
    ]
    intensity: dict[str, Any] = {}
    for flag in (False, True):
        group = [item for item in records if bool(cast(Mapping[str, Any], item["card_intensity"])["flagged"]) is flag]
        intensity[str(flag).lower()] = {
            "n": len(group),
            "mean_model_cards": float(np.mean([item["count_means"]["card_model_total"] for item in group])) if group else None,
            "mean_distribution_cards": float(np.mean([item["count_means"]["card_distribution_total"] for item in group])) if group else None,
            "mean_observed_cards": float(np.mean([item["actuals"]["total_cards"] for item in group])) if group else None,
        }
    corner_means = np.asarray([float(item["count_means"]["corner_total"]) for item in records])
    card_means = np.asarray([float(item["count_means"]["card_distribution_total"]) for item in records])
    return {
        "corners": {"historical_training_validation_observed_max": historical_corner_max, "final_expected_mean_summary": {"min": float(corner_means.min()), "mean": float(corner_means.mean()), "max": float(corner_means.max())}, "above_historical_max_count": len(tail), "above_historical_max_cases": tail},
        "cards": {"final_distribution_mean_summary": {"min": float(card_means.min()), "mean": float(card_means.mean()), "max": float(card_means.max())}, "intensity_multiplier_frozen": 1.6, "intensity_groups": intensity, "methodology_decision": "REQUIRES DECISION"},
    }


def evaluate(run_dir: Path) -> None:
    spec = verify_frozen(run_dir)
    partitions, _, _ = load_phase3_partitions(reveal_final_outcomes=True)
    if row_ids_hash(partitions.final_test["match_hash"].astype(str)) != spec["final_test_row_ids_sha256"]:
        raise Phase3IntegrityError("Final-test row identity changed after experiment freeze")
    bundle = _read_json(run_dir / "base_model_bundle.json")
    models = load_base_bundle(bundle)
    calibrators = _loaded_calibrators(run_dir, spec)
    policy = _read_json(run_dir / "validation_confidence_policy.json")
    refit = pd.concat([partitions.training, partitions.validation], ignore_index=True)
    baselines = confidence_baselines(refit)
    context = CausalContextBuilder()
    raw_records = run_period_predictions(partitions.final_test, models, bundle, context, baselines=baselines, hce_buckets=policy["hce_buckets"])
    calibrated_records = run_period_predictions(partitions.final_test, models, bundle, context, calibrator_payloads=calibrators, baselines=baselines, hce_buckets=policy["hce_buckets"])
    for raw, calibrated in zip(raw_records, calibrated_records, strict=True):
        _assert_paired_path(raw, calibrated)
    write_jsonl(run_dir / "final_test_raw_predictions.jsonl", raw_records)
    write_jsonl(run_dir / "final_test_calibrated_predictions.jsonl", calibrated_records)
    write_json(run_dir / "multiclass_metrics.json", _multiclass_report(raw_records, refit))
    binary, paired = _binary_reports(raw_records, calibrated_records, refit)
    write_json(run_dir / "binary_metrics.json", binary)
    write_json(run_dir / "paired_calibration_comparison.json", paired)
    write_json(run_dir / "count_market_diagnostics.json", _count_diagnostics(raw_records, refit))
    after = protected_snapshot()
    write_json(run_dir / "protected_after.json", after)
    before = _read_json(run_dir / "protected_before.json")
    keys = sorted(set(before) - {"captured_at"})
    integrity = {key: {"before": before[key], "after": after[key], "unchanged": before[key] == after[key]} for key in keys}
    if not all(item["unchanged"] for item in integrity.values()):
        raise Phase3IntegrityError("A protected prior-phase or production artifact changed")
    write_json(run_dir / "protected_verification.json", integrity)
    index_items = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name not in {"artifact_index.json", "PHASE3_REPORT.md"}:
            index_items.append({"path": path.relative_to(run_dir).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
    write_json(run_dir / "artifact_index.json", {"created_at": utc_now(), "experiment_spec_sha256": sha256_file(run_dir / "experiment_spec.json"), "artifacts": index_items, "aggregate_sha256": sha256_json(index_items)})
    write_report(run_dir)


def write_report(run_dir: Path) -> None:
    spec = _read_json(run_dir / "experiment_spec.json")
    multi = _read_json(run_dir / "multiclass_metrics.json")
    binary = _read_json(run_dir / "binary_metrics.json")
    paired = _read_json(run_dir / "paired_calibration_comparison.json")
    diagnostics = _read_json(run_dir / "count_market_diagnostics.json")
    calibrators = _read_json(run_dir / "calibrator_manifest.json")
    lines = [
        "# FFPS PHASE 3 EVIDENCE REPORT", "", "> Phase 3 evaluates probability integrity; it does not promote artifacts, tune thresholds, or establish profitability.", "",
        "## Frozen experiment", "", f"- Specification SHA-256: {sha256_file(run_dir / 'experiment_spec.json')}", f"- Partitions: {spec['partitions']}", f"- Counts: {spec['partition_counts']}", "- Final outcomes were accessed only after frozen-spec verification.", "",
        "## 1X2 final test", "", f"- N: {multi['sample_size']}", f"- Live metrics: {multi['live_path']}", f"- Top-label ECE: {multi['reliability']['top_label']['ece']:.6f}", f"- Baselines: {multi['baselines']}", "",
        "## Binary final test", "",
    ]
    for market, result in binary.items():
        lines.append(f"- {market} raw: Brier {result['raw']['brier']:.6f}, log loss {result['raw']['log_loss']:.6f}, ECE {result['raw']['ece']:.6f}, N {result['raw']['sample_size']}")
        if market in paired:
            delta = paired[market]["deltas"]
            lines.append(f"  Calibrated: Brier {result['calibrated']['brier']:.6f}, log loss {result['calibrated']['log_loss']:.6f}, ECE {result['calibrated']['ece']:.6f}; delta Brier {delta['brier']['point']:.6f} [{delta['brier']['lower']:.6f}, {delta['brier']['upper']:.6f}] - {paired[market]['classification']}")
    lines.extend(["", "## Calibration artifacts", "", f"{calibrators['artifacts']}", "", "## Count diagnostics", "", f"{diagnostics}", "", "## Scope", "", "PL only. No bookmaker inputs, ROI, threshold optimization, production promotion, or card-intensity change.", ""])
    _write_text(run_dir / "PHASE3_REPORT.md", "\n".join(lines))


def verify(run_dir: Path) -> None:
    spec = verify_frozen(run_dir)
    required = ("final_test_raw_predictions.jsonl", "final_test_calibrated_predictions.jsonl", "multiclass_metrics.json", "binary_metrics.json", "paired_calibration_comparison.json", "count_market_diagnostics.json", "protected_verification.json", "artifact_index.json", "PHASE3_REPORT.md")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise Phase3IntegrityError(f"Incomplete Phase 3 evidence: {missing}")
    if not all(item["unchanged"] for item in _read_json(run_dir / "protected_verification.json").values()):
        raise Phase3IntegrityError("Protected artifact verification failed")
    print(json.dumps({"status": "VERIFIED", "experiment_spec_sha256": sha256_file(run_dir / "experiment_spec.json"), "base_bundle_sha256": spec["base_bundle_sha256"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "evaluate", "verify"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "evaluate": evaluate, "verify": verify}[args.command](args.run_dir)


if __name__ == "__main__":
    main()

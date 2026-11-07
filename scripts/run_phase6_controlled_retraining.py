"""Prepare, execute, compare, integrate, finalize, and verify FFPS Phase 6."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
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

from scripts.run_phase4_prediction_provenance import _input_frames
from scripts.run_phase5_training_lineage import verify_evidence as verify_phase5_evidence
from scripts.run_phase5_training_lineage import protected_snapshot as phase5_base_protected_snapshot
from src.cli.commands.prediction import _calc_goals
from src.evaluation.phase3_calibration import aggregate_directory
from src.ml.phase6_controlled_training import (
    CANDIDATE_CLASSIFICATION,
    MODEL_ROLE_ORDER,
    Phase6TrainingError,
    Phase6TrainingSpecification,
    build_candidate_assessment,
    build_reproduced_lineage_record,
    compare_predictions,
    explicit_model_configs,
    fit_fresh_candidate,
    row_ids_hash,
    seal_specification,
    serialize_candidate,
    source_runtime_identity,
    utc_now,
    verify_candidate_assessment,
    verify_partition_sets,
)
from src.ml.training_lineage import ModelTrainingLineageRecord, build_partition_record, verify_lineage_record
from src.predictions.provenance import (
    PredictionCapture,
    PredictionProvenanceRecord,
    data_provenance_for_fixture,
    mark_output_reproduced,
    replay_pl_1x2,
    sha256_file,
    sha256_json,
    verify_record,
)
from src.simulation.match_simulator import DEFAULT_N_SIMULATIONS, MatchSimulator


DEFAULT_RUN_ID = "phase6-controlled-pl-goals-79110f9a-20260829-v1"
DEFAULT_RUN_DIR = ROOT / "baselines" / "phase6" / DEFAULT_RUN_ID
PHASE1_RUN = ROOT / "baselines" / "phase1" / "phase1-pl-1x2-79110f9a-20240801-v1"
PHASE3_RUN = ROOT / "baselines" / "phase3" / "phase3-pl-1x2-temperature-79110f9a-20260827-v1"
PHASE4_RUN = ROOT / "baselines" / "phase4" / "phase4-pl-1x2-provenance-79110f9a-20260827-v1"
PHASE5_RUN = ROOT / "baselines" / "phase5" / "phase5-pl-goal-training-lineage-79110f9a-20260829-v1"
PHASE6_PROTECTED_BEFORE = ROOT / "tmp" / "phase6-protected-before.json"
FEATURE_MATRIX = ROOT / "data" / "features" / "feature_matrix.csv"
ACTIVE_MANIFEST = ROOT / "src" / "ml" / "models" / "manifest.json"
EXPECTED_PARTITION_HASHES = {
    "training": "bc9c216388bfc0b507767d440308d9b7b23a0c94d6543c8e34d78a8ef23afb03",
    "validation": "dcff70ab6469eb3b424b554d87c1c0c8d2eb3a10efc29fdfa0b1ebcc85f1d04f",
    "calibration_fit": "2e73e3a810299695038256be5a1179237d700b16f3dd2fda77c613d40115d503",
    "final_test": "c98063a31980a7e35a6b880b51ec48436893d6bc0b53e05310da6ebe952c0a04",
}
INHERITED_FAILURES = {
    "tests.cli.test_verbosity_levels.TestCLIVerbosityLevels::test_quiet_mode_suppresses_pipeline_logs",
    "tests.test_drift_artifacts::test_global_drift_artifact_evaluated_at_is_present_and_recent",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PL]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[BL1]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[SA]",
    "tests.test_drift_artifacts::test_league_drift_artifact_evaluated_at_is_present_and_recent[PD]",
}


class Phase6EvidenceError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _same(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return {key: value for key, value in left.items() if key != "captured_at"} == {
        key: value for key, value in right.items() if key != "captured_at"
    }


def protected_snapshot() -> dict[str, Any]:
    snapshot = phase5_base_protected_snapshot()
    snapshot["phase5_evidence_aggregate_sha256"] = aggregate_directory(PHASE5_RUN)
    snapshot["phase5_run_manifest_sha256"] = sha256_file(PHASE5_RUN / "run_manifest.json")
    snapshot["phase5_offline_verification_sha256"] = sha256_file(PHASE5_RUN / "offline_verification.json")
    return snapshot


def _safe_new_run_dir(run_dir: Path) -> None:
    allowed = (ROOT / "baselines" / "phase6").resolve()
    try:
        run_dir.resolve().relative_to(allowed)
    except ValueError as exc:
        raise Phase6EvidenceError("Phase 6 evidence must remain beneath baselines/phase6") from exc
    if run_dir.exists():
        raise Phase6EvidenceError(f"Phase 6 directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)


def _phase4_features() -> list[str]:
    record = _read_json(PHASE4_RUN / "prediction_provenance.json")
    calls = record["feature_provenance"]["model_inputs"]
    schemas = [[str(value) for value in item["ordered_feature_names"]] for item in calls]
    if not schemas or any(names != schemas[0] for names in schemas[1:]):
        raise Phase6EvidenceError("Phase 4 goal models do not share one ordered feature schema")
    return schemas[0]


def _partition_sources() -> tuple[dict[str, list[str]], dict[str, Any]]:
    phase1 = _read_json(PHASE1_RUN / "partition_manifest.json")
    phase3 = _read_json(PHASE3_RUN / "partition_manifest.json")
    partitions = {
        "training": [str(value) for value in phase1["partitions"]["training"]["match_hashes"]],
        "validation": [str(value) for value in phase1["partitions"]["validation"]["match_hashes"]],
        "calibration_fit": [str(value) for value in phase3["partitions"]["calibration_fit"]["fixture_ids"]],
        "final_test": [str(value) for value in phase3["partitions"]["final_test"]["fixture_ids"]],
    }
    source_claims = {
        "phase1_training": phase1["partitions"]["training"]["match_hashes_sha256"],
        "phase1_validation": phase1["partitions"]["validation"]["match_hashes_sha256"],
        "phase3_training": phase3["partitions"]["model_train"]["fixture_ids_sha256"],
        "phase3_validation": phase3["partitions"]["model_validation"]["fixture_ids_sha256"],
        "phase3_calibration_fit": phase3["partitions"]["calibration_fit"]["fixture_ids_sha256"],
        "phase3_final_test": phase3["partitions"]["final_test"]["fixture_ids_sha256"],
    }
    if source_claims != {
        "phase1_training": EXPECTED_PARTITION_HASHES["training"],
        "phase1_validation": EXPECTED_PARTITION_HASHES["validation"],
        "phase3_training": EXPECTED_PARTITION_HASHES["training"],
        "phase3_validation": EXPECTED_PARTITION_HASHES["validation"],
        "phase3_calibration_fit": EXPECTED_PARTITION_HASHES["calibration_fit"],
        "phase3_final_test": EXPECTED_PARTITION_HASHES["final_test"],
    }:
        raise Phase6EvidenceError(f"Published partition claims changed: {source_claims}")
    return partitions, source_claims


def _materialize_inputs() -> dict[str, Any]:
    partitions, _ = _partition_sources()
    features = _phase4_features()
    columns = [
        "match_hash",
        "date",
        "league",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        *features,
    ]
    frame = pd.read_csv(FEATURE_MATRIX, usecols=columns, low_memory=False)
    frame = frame.loc[frame["league"].astype(str).eq("PL")].copy()
    frame["match_hash"] = frame["match_hash"].astype(str)
    if frame["match_hash"].duplicated().any():
        raise Phase6EvidenceError("Duplicate PL match_hash in feature matrix")
    indexed = frame.set_index("match_hash", drop=False)

    def select(name: str) -> pd.DataFrame:
        ids = partitions[name]
        missing = sorted(set(ids) - set(indexed.index))
        if missing:
            raise Phase6EvidenceError(f"Feature matrix is missing {name} rows: {missing[:5]}")
        selected = indexed.loc[ids].copy()
        if list(selected["match_hash"]) != ids:
            raise Phase6EvidenceError(f"Could not preserve frozen {name} row order")
        return selected

    training = select("training")
    validation = select("validation")
    for name, selected in (("training", training), ("validation", validation)):
        matrix = selected[features].astype(np.float64)
        if not np.isfinite(matrix.to_numpy()).all():
            raise Phase6EvidenceError(f"{name} feature matrix contains non-finite values")
        for target in ("home_score", "away_score"):
            values = selected[target].clip(upper=6).astype(np.float64)
            if not np.isfinite(values.to_numpy()).all():
                raise Phase6EvidenceError(f"{name} {target} target contains non-finite values")
    return {
        "features": features,
        "partitions": partitions,
        "training_rows": training,
        "validation_rows": validation,
        "X_training": training[features].astype(np.float64),
        "X_validation": validation[features].astype(np.float64),
        "y_training_home": training["home_score"].clip(upper=6).astype(np.float64),
        "y_training_away": training["away_score"].clip(upper=6).astype(np.float64),
        "y_validation_home": validation["home_score"].clip(upper=6).astype(np.float64),
        "y_validation_away": validation["away_score"].clip(upper=6).astype(np.float64),
    }


def _partition_summary(
    *,
    role: str,
    features: pd.DataFrame,
    target: pd.Series,
    rows: pd.DataFrame,
    target_name: str,
) -> dict[str, Any]:
    record = build_partition_record(
        role=role,
        features=features,
        targets=target,
        row_ids=list(rows["match_hash"].astype(str)),
        event_times=list(rows["date"].astype(str)),
        target_name=target_name,
        missing_value_policy="reject_all_non_finite_values",
    )
    return {key: value for key, value in record.items() if key != "rows"}


def _input_contract(inputs: Mapping[str, Any]) -> dict[str, Any]:
    X_train = cast(pd.DataFrame, inputs["X_training"])
    X_val = cast(pd.DataFrame, inputs["X_validation"])
    train_rows = cast(pd.DataFrame, inputs["training_rows"])
    val_rows = cast(pd.DataFrame, inputs["validation_rows"])
    return {
        "schema_version": "ffps.phase6_input_contract.v1",
        "source_dataset": {
            "path": _relative(FEATURE_MATRIX),
            "sha256": sha256_file(FEATURE_MATRIX),
            "size_bytes": FEATURE_MATRIX.stat().st_size,
        },
        "ordered_feature_names": list(inputs["features"]),
        "feature_schema_sha256": sha256_json(list(inputs["features"])),
        "training": {
            "home": _partition_summary(
                role="model_fit",
                features=X_train,
                target=cast(pd.Series, inputs["y_training_home"]),
                rows=train_rows,
                target_name="home_goals",
            ),
            "away": _partition_summary(
                role="model_fit",
                features=X_train,
                target=cast(pd.Series, inputs["y_training_away"]),
                rows=train_rows,
                target_name="away_goals",
            ),
        },
        "validation": {
            "home": _partition_summary(
                role="frozen_validation_reproduction_corpus",
                features=X_val,
                target=cast(pd.Series, inputs["y_validation_home"]),
                rows=val_rows,
                target_name="home_goals",
            ),
            "away": _partition_summary(
                role="frozen_validation_reproduction_corpus",
                features=X_val,
                target=cast(pd.Series, inputs["y_validation_away"]),
                rows=val_rows,
                target_name="away_goals",
            ),
        },
    }


def _verify_input_contract(contract: Mapping[str, Any], inputs: Mapping[str, Any]) -> None:
    current = _input_contract(inputs)
    if current != contract:
        raise Phase6EvidenceError("Materialized Phase 6 inputs differ from the frozen pre-fit contract")


def _source_files() -> list[Path]:
    return [
        ROOT / "src" / "ml" / "phase6_controlled_training.py",
        ROOT / "src" / "ml" / "training_lineage.py",
        ROOT / "src" / "models" / "train_probability_models.py",
        ROOT / "scripts" / "run_phase6_controlled_retraining.py",
    ]


def _verify_spec_sources(spec: Phase6TrainingSpecification) -> None:
    runtime = cast(Mapping[str, Any], spec.to_dict()["source_runtime"])
    for item in cast(Sequence[Mapping[str, Any]], runtime["source_files"]):
        path = ROOT / str(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise Phase6EvidenceError(f"Phase 6 source changed after specification freeze: {item['path']}")


def prepare(run_dir: Path) -> dict[str, Any]:
    _safe_new_run_dir(run_dir)
    before = _read_json(PHASE6_PROTECTED_BEFORE)
    current = protected_snapshot()
    if not _same(before, current):
        raise Phase6EvidenceError("Protected Phase 0-5 state changed before Phase 6 preparation")
    shutil.copy2(PHASE6_PROTECTED_BEFORE, run_dir / "protected_before.json")
    partitions, source_claims = _partition_sources()
    proof = verify_partition_sets(partitions, expected_hashes=EXPECTED_PARTITION_HASHES)
    inputs = _materialize_inputs()
    identity_frame = pd.read_csv(
        FEATURE_MATRIX,
        usecols=["match_hash", "date", "league", "home_team", "away_team"],
        low_memory=False,
    )
    identity_frame = identity_frame.loc[identity_frame["league"].astype(str).eq("PL")].copy()
    identity_frame["match_hash"] = identity_frame["match_hash"].astype(str)
    identity_frame = identity_frame.set_index("match_hash")
    for name, partition in cast(dict[str, dict[str, Any]], proof["partitions"]).items():
        ids = cast(list[str], partition["row_ids"])
        rows = identity_frame.loc[ids]
        canonical = [
            f"{index}|{row['date']}|{row['league']}|{row['home_team']}|{row['away_team']}"
            for index, row in rows.iterrows()
        ]
        partition["canonical_fixture_identities_sha256"] = sha256_json(sorted(canonical))
    proof["source_claims"] = source_claims
    proof["phase1_manifest"] = {"path": _relative(PHASE1_RUN / "partition_manifest.json"), "sha256": sha256_file(PHASE1_RUN / "partition_manifest.json")}
    proof["phase3_manifest"] = {"path": _relative(PHASE3_RUN / "partition_manifest.json"), "sha256": sha256_file(PHASE3_RUN / "partition_manifest.json")}
    _write_json(run_dir / "partition_integrity.json", proof)
    contract = _input_contract(inputs)
    _write_json(run_dir / "input_contract.json", contract)
    configs = explicit_model_configs()
    runtime = source_runtime_identity(root=ROOT, source_files=_source_files())
    specification = {
        "schema_version": "ffps.phase6_training_specification.v1",
        "specification_id": "phase6-pl-goals-controlled-from-scratch-v1",
        "created_at": utc_now(),
        "phase": 6,
        "purpose": "evaluation_only_controlled_lineage_retraining",
        "model_roles": list(MODEL_ROLE_ORDER),
        "feature_contract": {
            "ordered_feature_names": list(inputs["features"]),
            "feature_schema_sha256": contract["feature_schema_sha256"],
            "representation": "pandas.DataFrame with ordered float64 columns; exact numerical values hashed",
            "preprocessing": "none after frozen feature matrix selection",
            "missing_value_handling": "reject every NaN or Infinity before fit and validation prediction",
        },
        "target_contract": {
            "home_goals": "float64 clip(home_score, upper=6)",
            "away_goals": "float64 clip(away_score, upper=6)",
            "sample_weighting": "none",
        },
        "partition_contract": {
            "training_identity": EXPECTED_PARTITION_HASHES["training"],
            "validation_identity": EXPECTED_PARTITION_HASHES["validation"],
            "calibration_fit_excluded_identity": EXPECTED_PARTITION_HASHES["calibration_fit"],
            "final_test_excluded_identity": EXPECTED_PARTITION_HASHES["final_test"],
            "validation_use": "prediction reproduction corpus and descriptive diagnostics only; not passed to fit",
            "final_test_outcomes": "not used or evaluated",
            "partition_integrity_sha256": sha256_file(run_dir / "partition_integrity.json"),
        },
        "input_contract": {
            "path": _relative(run_dir / "input_contract.json"),
            "sha256": sha256_file(run_dir / "input_contract.json"),
            "source_dataset_sha256": contract["source_dataset"]["sha256"],
        },
        "model_protocols": {
            role: {
                "library_family": "lightgbm.LGBMRegressor" if role.endswith("lgbm") else "xgboost.XGBRegressor",
                "target": "home_goals" if role.startswith("home") else "away_goals",
                "hyperparameters": configs[role],
                "number_of_rounds_policy": f"fixed n_estimators={configs[role]['n_estimators']}; no early stopping",
                "early_stopping": False,
                "warm_start": False,
                "continuation_arguments": [],
                "serialization": "joblib.dump(compress=0, protocol=5)",
            }
            for role in MODEL_ROLE_ORDER
        },
        "randomness_parallelism": {
            "random_state": 42,
            "threads_per_estimator": 1,
            "lightgbm_deterministic": True,
            "lightgbm_force_col_wise": True,
            "execution_order": list(MODEL_ROLE_ORDER),
        },
        "reproduction_contract": {
            "clean_execution_count": 2,
            "process_isolation": "primary and reproduction commands run in separate Python processes",
            "prediction_corpus": "complete frozen Phase 1 validation partition in frozen row order",
            "prediction_atol": 1e-12,
            "prediction_rtol": 1e-12,
            "input_identity": "exact",
            "fit_procedure_identity": "exact specification/config/seed/runtime",
            "artifact_bytes": "compare truthfully; byte equality desirable but not required if semantic reproduction succeeds",
            "tolerance_change_after_fit": "forbidden",
        },
        "source_runtime": runtime,
        "prospective_decisions": [
            "LightGBM effective no-row-subsampling is explicit as subsample=1.0 and subsample_freq=0.",
            "Both libraries use one CPU thread to remove uncontrolled scheduling variation.",
            "XGBoost base_score=0.5 and all material tree/regularization defaults are explicit.",
            "No attempt is made to reproduce legacy 8000/9600 tree counts or predictions.",
        ],
        "activation_boundary": {
            "registry_access": "forbidden",
            "active_manifest_write": "forbidden",
            "output_location": _relative(run_dir),
        },
    }
    spec = Phase6TrainingSpecification.from_dict(seal_specification(specification))
    spec.save(run_dir / "training_specification.json")
    _verify_spec_sources(spec)
    result = {
        "status": "PREPARED_AND_FROZEN_BEFORE_FIRST_FIT",
        "prepared_at": utc_now(),
        "run_id": run_dir.name,
        "training_specification_sha256": spec.fingerprint,
        "training_rows": len(inputs["training_rows"]),
        "validation_rows": len(inputs["validation_rows"]),
        "protected_partitions_disjoint": True,
        "active_manifest_sha256": sha256_file(ACTIVE_MANIFEST),
        "protected_prior_phases_unchanged": True,
    }
    _write_json(run_dir / "preparation_result.json", result)
    return result


def fit_execution(run_dir: Path, execution: str) -> dict[str, Any]:
    if execution not in {"primary", "reproduction"}:
        raise Phase6EvidenceError("Execution must be primary or reproduction")
    output_dir = run_dir / ("candidates" if execution == "primary" else "reproduction")
    manifest_path = run_dir / "executions" / f"{execution}_manifest.json"
    if output_dir.exists() or manifest_path.exists():
        raise Phase6EvidenceError(f"Immutable Phase 6 {execution} execution already exists")
    before = _read_json(run_dir / "protected_before.json")
    if not _same(before, protected_snapshot()):
        raise Phase6EvidenceError("Protected state changed before controlled fit")
    spec = Phase6TrainingSpecification.load(run_dir / "training_specification.json")
    _verify_spec_sources(spec)
    inputs = _materialize_inputs()
    contract = _read_json(run_dir / "input_contract.json")
    _verify_input_contract(contract, inputs)
    spec_payload = spec.to_dict()
    records: list[dict[str, Any]] = []
    for role in MODEL_ROLE_ORDER:
        protocol = cast(Mapping[str, Any], spec_payload["model_protocols"][role])
        config = cast(Mapping[str, Any], protocol["hyperparameters"])
        home = role.startswith("home")
        target = cast(pd.Series, inputs["y_training_home"] if home else inputs["y_training_away"])
        validation_target = cast(pd.Series, inputs["y_validation_home"] if home else inputs["y_validation_away"])
        model, predictions, warm_start_proof = fit_fresh_candidate(
            role=role,
            config=config,
            fit_features=cast(pd.DataFrame, inputs["X_training"]),
            fit_target=target,
            validation_features=cast(pd.DataFrame, inputs["X_validation"]),
        )
        artifact_path = output_dir / f"{role}.joblib"
        artifact = serialize_candidate(model, artifact_path, run_dir=run_dir)
        prediction_values = [float(value) for value in predictions]
        residuals = predictions - validation_target.to_numpy(dtype=np.float64)
        records.append(
            {
                "role": role,
                "artifact": artifact,
                "prediction_values": prediction_values,
                "prediction_sha256": sha256_json(prediction_values),
                "prediction_count": len(prediction_values),
                "config_sha256": sha256_json(config),
                "randomness_sha256": sha256_json({"random_state": 42, "n_jobs": 1}),
                "warm_start_proof": warm_start_proof,
                "descriptive_validation_only": {
                    "mae": float(np.mean(np.abs(residuals))),
                    "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
                    "used_for_selection_or_retry": False,
                },
            }
        )
    manifest = {
        "schema_version": "ffps.phase6_fit_execution.v1",
        "execution": execution,
        "completed_at": utc_now(),
        "process_isolation": "fresh Python process",
        "training_specification_sha256": spec.fingerprint,
        "input_contract_sha256": sha256_file(run_dir / "input_contract.json"),
        "active_manifest_before_sha256": before["active_manifest_sha256"],
        "active_manifest_after_sha256": sha256_file(ACTIVE_MANIFEST),
        "models": records,
    }
    if manifest["active_manifest_before_sha256"] != manifest["active_manifest_after_sha256"]:
        raise Phase6EvidenceError("Active manifest changed during controlled fit")
    _write_json(manifest_path, manifest)
    return {
        "status": "FIT_COMPLETE",
        "execution": execution,
        "model_count": len(records),
        "training_specification_sha256": spec.fingerprint,
        "active_manifest_unchanged": True,
    }


def compare_and_freeze(run_dir: Path) -> dict[str, Any]:
    output = run_dir / "reproduction_results.json"
    if output.exists():
        raise Phase6EvidenceError("Reproduction comparison already exists and is immutable")
    spec = Phase6TrainingSpecification.load(run_dir / "training_specification.json")
    _verify_spec_sources(spec)
    primary = _read_json(run_dir / "executions" / "primary_manifest.json")
    repeat = _read_json(run_dir / "executions" / "reproduction_manifest.json")
    for field in ("training_specification_sha256", "input_contract_sha256"):
        if primary[field] != repeat[field]:
            raise Phase6EvidenceError(f"Execution {field} drift")
    spec_payload = spec.to_dict()
    tolerance = cast(Mapping[str, Any], spec_payload["reproduction_contract"])
    primary_by_role = {str(item["role"]): item for item in primary["models"]}
    repeat_by_role = {str(item["role"]): item for item in repeat["models"]}
    comparisons: list[dict[str, Any]] = []
    for role in MODEL_ROLE_ORDER:
        first = primary_by_role[role]
        second = repeat_by_role[role]
        if first["config_sha256"] != second["config_sha256"] or first["randomness_sha256"] != second["randomness_sha256"]:
            raise Phase6EvidenceError(f"Configuration or seed drift for {role}")
        prediction = compare_predictions(
            first["prediction_values"],
            second["prediction_values"],
            atol=float(tolerance["prediction_atol"]),
            rtol=float(tolerance["prediction_rtol"]),
        )
        if not prediction["reproduced"]:
            raise Phase6EvidenceError(f"Predeclared prediction reproduction contract failed for {role}: {prediction}")
        comparisons.append(
            {
                "role": role,
                "input_reproducible": True,
                "fit_procedure_reproducible": True,
                "prediction_reproducibility": prediction,
                "primary_artifact_sha256": first["artifact"]["sha256"],
                "reproduction_artifact_sha256": second["artifact"]["sha256"],
                "artifact_bytes_identical": first["artifact"]["sha256"] == second["artifact"]["sha256"],
                "artifact_byte_result": (
                    "ARTIFACT_BYTE_REPRODUCIBLE"
                    if first["artifact"]["sha256"] == second["artifact"]["sha256"]
                    else "ARTIFACT_BYTES_DIFFERENT_SEMANTIC_REPRODUCTION_VERIFIED"
                ),
            }
        )
    result = {
        "schema_version": "ffps.phase6_reproduction.v1",
        "training_specification_sha256": spec.fingerprint,
        "input_contract_sha256": primary["input_contract_sha256"],
        "predeclared_tolerance": {
            "atol": tolerance["prediction_atol"],
            "rtol": tolerance["prediction_rtol"],
        },
        "comparisons": comparisons,
        "all_inputs_reproducible": True,
        "all_fit_procedures_reproducible": True,
        "all_predictions_reproducible": True,
        "artifact_byte_reproducible_count": sum(bool(item["artifact_bytes_identical"]) for item in comparisons),
    }
    _write_json(output, result)

    inputs = _materialize_inputs()
    contract = _read_json(run_dir / "input_contract.json")
    _verify_input_contract(contract, inputs)
    evidence = [
        {"path": _relative(run_dir / "partition_integrity.json"), "sha256": sha256_file(run_dir / "partition_integrity.json")},
        {"path": _relative(run_dir / "input_contract.json"), "sha256": sha256_file(run_dir / "input_contract.json")},
        {"path": _relative(output), "sha256": sha256_file(output)},
    ]
    lineage_items: dict[str, dict[str, Any]] = {}
    comparison_by_role = {str(item["role"]): item for item in comparisons}
    for execution, manifest, by_role in (
        ("primary", primary, primary_by_role),
        ("reproduction", repeat, repeat_by_role),
    ):
        for role in MODEL_ROLE_ORDER:
            item = by_role[role]
            home = role.startswith("home")
            artifact_path = ROOT / str(item["artifact"]["path"])
            lineage = build_reproduced_lineage_record(
                artifact_path=artifact_path,
                role=role,
                execution=execution,
                fit_features=cast(pd.DataFrame, inputs["X_training"]),
                fit_target=cast(pd.Series, inputs["y_training_home"] if home else inputs["y_training_away"]),
                fit_row_ids=list(cast(pd.DataFrame, inputs["training_rows"])["match_hash"].astype(str)),
                fit_event_times=list(cast(pd.DataFrame, inputs["training_rows"])["date"].astype(str)),
                validation_features=cast(pd.DataFrame, inputs["X_validation"]),
                validation_target=cast(pd.Series, inputs["y_validation_home"] if home else inputs["y_validation_away"]),
                validation_row_ids=list(cast(pd.DataFrame, inputs["validation_rows"])["match_hash"].astype(str)),
                validation_event_times=list(cast(pd.DataFrame, inputs["validation_rows"])["date"].astype(str)),
                config=cast(Mapping[str, Any], spec_payload["model_protocols"][role]["hyperparameters"]),
                randomness={"random_state": 42, "n_jobs": 1},
                warm_start_proof=cast(Mapping[str, Any], item["warm_start_proof"]),
                source_dataset_path=FEATURE_MATRIX,
                source_files=_source_files(),
                training_specification_path=run_dir / "training_specification.json",
                reproduction=comparison_by_role[role],
                evidence=evidence,
                root=ROOT,
            )
            lineage_path = run_dir / "lineage" / execution / f"{role}.json"
            lineage.save(lineage_path)
            lineage_items[f"{execution}:{role}"] = {
                "path": _relative(lineage_path),
                "sha256": sha256_file(lineage_path),
                "fingerprint": lineage.fingerprint,
                "classification": lineage.to_dict()["classification"],
            }

    assessments = []
    for role in MODEL_ROLE_ORDER:
        lineage_path = ROOT / lineage_items[f"primary:{role}"]["path"]
        assessment = build_candidate_assessment(
            role=role,
            artifact_path=ROOT / str(primary_by_role[role]["artifact"]["path"]),
            core_lineage_path=lineage_path,
            training_specification_path=run_dir / "training_specification.json",
            partition_proof_path=run_dir / "partition_integrity.json",
            reproduction_path=output,
            root=ROOT,
        )
        assessment_path = run_dir / "assessments" / f"{role}.json"
        _write_json(assessment_path, assessment)
        verification = verify_candidate_assessment(assessment, root=ROOT)
        if not verification["verified"]:
            raise Phase6EvidenceError(f"Candidate assessment failed: {verification}")
        assessments.append(
            {
                "role": role,
                "path": _relative(assessment_path),
                "sha256": sha256_file(assessment_path),
                "fingerprint": assessment["assessment_fingerprint"],
                "classification": CANDIDATE_CLASSIFICATION,
            }
        )

    bundle_models = []
    for role in MODEL_ROLE_ORDER:
        primary_item = primary_by_role[role]
        bundle_models.append(
            {
                "role": role,
                "artifact": primary_item["artifact"],
                "lineage": lineage_items[f"primary:{role}"],
                "assessment": next(item for item in assessments if item["role"] == role),
                "feature_schema_sha256": contract["feature_schema_sha256"],
                "tree_count": primary_item["warm_start_proof"]["tree_count"],
            }
        )
    bundle_material = {
        "training_specification_sha256": spec.fingerprint,
        "training_partition_identity": EXPECTED_PARTITION_HASHES["training"],
        "validation_partition_identity": EXPECTED_PARTITION_HASHES["validation"],
        "models": bundle_models,
    }
    bundle_fingerprint = sha256_json(bundle_material)
    bundle = {
        "schema_version": "ffps.phase6_candidate_bundle.v1",
        "bundle_id": f"phase6-evaluation-only-pl-goals-{bundle_fingerprint[:16]}",
        "created_at": utc_now(),
        "evaluation_only": True,
        "production_activation_eligible": False,
        "candidate_classification": CANDIDATE_CLASSIFICATION,
        **bundle_material,
        "runtime_identity": spec_payload["source_runtime"]["runtime"],
        "reproduction_results_sha256": sha256_file(output),
        "bundle_aggregate_sha256": bundle_fingerprint,
    }
    _write_json(run_dir / "candidate_bundle.json", bundle)
    return {
        "status": "CANDIDATE_BUNDLE_FROZEN",
        "bundle_id": bundle["bundle_id"],
        "bundle_aggregate_sha256": bundle_fingerprint,
        "candidate_count": 4,
        "lineage_record_count": 8,
        "classification": CANDIDATE_CLASSIFICATION,
        "all_predictions_reproducible": True,
        "artifact_byte_reproducible_count": result["artifact_byte_reproducible_count"],
    }


def integrate_candidate_bundle(run_dir: Path) -> dict[str, Any]:
    integration_dir = run_dir / "integration"
    if integration_dir.exists():
        raise Phase6EvidenceError("Phase 6 integration evidence already exists")
    bundle = _read_json(run_dir / "candidate_bundle.json")
    features = _phase4_features()
    models: dict[str, Any] = {}
    for item in bundle["models"]:
        role = str(item["role"])
        artifact = cast(Mapping[str, Any], item["artifact"])
        path = ROOT / str(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise Phase6EvidenceError(f"Candidate artifact substitution before integration: {role}")
        model = joblib.load(path)
        model.meta = {
            "features": features,
            "type": "lgbm_regressor_poisson" if role.endswith("lgbm") else "xgb_regressor_count_poisson",
            "candidate_bundle_id": bundle["bundle_id"],
        }
        model._ffps_provenance = {
            "artifact_sha256": artifact["sha256"],
            "source_location": artifact["path"],
            "size_bytes": artifact["size_bytes"],
            "manifest_key": f"{bundle['bundle_id']}:{role}",
            "registry_slot": f"evaluation_only:{role}",
            "training_lineage": {
                "classification": "proven",
                "phase6_classification": CANDIDATE_CLASSIFICATION,
                "lineage_fingerprint": item["lineage"]["fingerprint"],
                "lineage_path": item["lineage"]["path"],
            },
        }
        models[role] = model
    suite = {
        "mh_goals": models["home_goals_lgbm"],
        "ma_goals": models["away_goals_lgbm"],
        "meta_goals": {"features": features},
        "mh_goals_xgb": models["home_goals_xgb"],
        "ma_goals_xgb": models["away_goals_xgb"],
        "meta_goals_xgb": {"features": features},
    }
    matrix, _ = _input_frames()
    validation_ids = _partition_sources()[0]["validation"]
    rows = matrix.loc[matrix["match_hash"].astype(str).eq(validation_ids[0])]
    if len(rows) != 1:
        raise Phase6EvidenceError("Frozen integration fixture is not uniquely available")
    fixture = rows.iloc[0]
    prediction_timestamp = pd.Timestamp(fixture["date"]).isoformat()
    data = data_provenance_for_fixture(matrix, fixture, prediction_timestamp)
    simulator = MatchSimulator(
        n_simulations=DEFAULT_N_SIMULATIONS,
        league="PL",
        seed=42,
        seed_scope="phase6_isolated_candidate_integration",
    )
    with PredictionCapture(prediction_timestamp=prediction_timestamp, data_provenance=data) as capture:
        probabilities = _calc_goals(
            fixture,
            cast(Any, suite),
            f"Phase6Candidate:{validation_ids[0]}",
            simulator=simulator,
            league="PL",
            rl_weights=None,
            use_simulator=True,
        )
        record = capture.finalize(fixture=fixture, league="PL", probabilities=probabilities, simulator=simulator)
    replay = replay_pl_1x2(record, root=ROOT)
    record = mark_output_reproduced(record, replay)
    integration_dir.mkdir(parents=True)
    record_path = integration_dir / "prediction_provenance.json"
    record.save(record_path)
    verification = verify_record(record, root=ROOT, verify_referenced_artifacts=True)
    replay_final = replay_pl_1x2(record, root=ROOT)
    if not verification["overall_phase4_trusted"] or not replay_final["reproduced"]:
        raise Phase6EvidenceError(f"Candidate Phase 4 integration verification failed: {verification} {replay_final}")
    member_hashes = {
        str(item["role"]): str(item["artifact_sha256"])
        for item in record.to_dict()["model_provenance"]["members"]
    }
    expected_hashes = {str(item["role"]): str(item["artifact"]["sha256"]) for item in bundle["models"]}
    if member_hashes != expected_hashes:
        raise Phase6EvidenceError("Phase 4 integration did not record the candidate bytes actually loaded")
    calls = record.to_dict()["feature_provenance"]["model_inputs"]
    result = {
        "status": "VERIFIED",
        "candidate_bundle_id": bundle["bundle_id"],
        "prediction_fingerprint": record.prediction_fingerprint,
        "fixture_match_hash": validation_ids[0],
        "probabilities": [probabilities["home"], probabilities["draw"], probabilities["away"]],
        "probability_sum": probabilities["home"] + probabilities["draw"] + probabilities["away"],
        "finite_and_normalized": all(math.isfinite(float(probabilities[key])) for key in ("home", "draw", "away"))
        and math.isclose(sum(float(probabilities[key]) for key in ("home", "draw", "away")), 1.0, abs_tol=1e-12),
        "candidate_artifact_hashes_recorded": member_hashes == expected_hashes,
        "lineage_classification": verification["training_lineage_classification"],
        "phase4_overall_trusted": verification["overall_phase4_trusted"],
        "independent_replay": replay_final["reproduced"],
        "no_model_calibrator_applied": all(not item["model_calibrator"]["applied"] for item in calls),
        "no_match_outcome_classifier": True,
        "bookmaker_inputs_absent": all(
            not any(token in name.lower() for token in ("odds", "bookmaker", "implied_probability"))
            for item in calls
            for name in item["ordered_feature_names"]
        ),
        "simulator_seed_scope": record.to_dict()["method_provenance"]["descriptor"]["simulation"]["seed_scope"],
        "active_manifest_sha256": sha256_file(ACTIVE_MANIFEST),
    }
    if not all(
        result[key]
        for key in (
            "finite_and_normalized",
            "candidate_artifact_hashes_recorded",
            "phase4_overall_trusted",
            "independent_replay",
            "no_model_calibrator_applied",
            "no_match_outcome_classifier",
            "bookmaker_inputs_absent",
        )
    ):
        raise Phase6EvidenceError(f"Candidate integration assertion failed: {result}")
    _write_json(integration_dir / "integration_result.json", result)
    return result


def _junit(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    failed = [
        f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    ]
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "tests": len(cases),
        "failures": len([case for case in cases if case.find("failure") is not None]),
        "errors": len([case for case in cases if case.find("error") is not None]),
        "skipped": len([case for case in cases if case.find("skipped") is not None]),
        "failed_tests": failed,
    }


def _report(run_dir: Path, tests: Mapping[str, Any]) -> str:
    bundle = _read_json(run_dir / "candidate_bundle.json")
    reproduction = _read_json(run_dir / "reproduction_results.json")
    integration = _read_json(run_dir / "integration" / "integration_result.json")
    model_lines = "\n".join(
        f"| {item['role']} | `{item['artifact']['sha256']}` | {item['tree_count']} | `{item['lineage']['fingerprint']}` |"
        for item in bundle["models"]
    )
    return f"""# Phase 6 — Controlled lineage-trusted PL goal retraining

## Decision

Phase 6 produced four new evaluation-only candidates from fresh estimator state.
Every candidate is `TRAINING_LINEAGE_VERIFIED`; the underlying frozen Phase 5
record is `LINEAGE_REPRODUCED`. No candidate was registered or activated.

## Controlled protocol

- Specification: `{bundle['training_specification_sha256']}`.
- Training partition: `{EXPECTED_PARTITION_HASHES['training']}` (1,505 fixtures).
- Validation partition: `{EXPECTED_PARTITION_HASHES['validation']}` (379 fixtures).
- Calibration-fit excluded: `{EXPECTED_PARTITION_HASHES['calibration_fit']}`.
- Final test excluded: `{EXPECTED_PARTITION_HASHES['final_test']}`.
- Warm start: forbidden; all four fits begin from new estimator objects.
- LightGBM: fixed 500 trees, seed 42, one thread, deterministic CPU column-wise mode.
- XGBoost: fixed 600 trees, seed 42, one thread, CPU histogram mode.
- Validation is not passed to fit and is not used for tuning or retry decisions.

## Candidate bundle

Bundle ID: `{bundle['bundle_id']}`  
Bundle aggregate: `{bundle['bundle_aggregate_sha256']}`

| Role | Artifact SHA-256 | Trees | Lineage fingerprint |
|---|---|---:|---|
{model_lines}

## Reproducibility

- Exact input identity: verified for all four models.
- Exact specification/config/seed/runtime identity: verified.
- Prediction reproduction: all four pass the predeclared `atol=1e-12`, `rtol=1e-12` contract.
- Byte-identical serialized artifacts: {reproduction['artifact_byte_reproducible_count']}/4.
- No tolerance, hyperparameter, or science change occurred after observing results.

## PL 1X2 integration

Phase 4 provenance verified with candidate bytes, classification `proven`, no
model calibrator, no match-outcome classifier, no bookmaker feature, and an
independently replayable normalized result. Prediction fingerprint:
`{integration['prediction_fingerprint']}`.

## Verification

- Focused Phase 6: {tests['focused']['tests']} tests, {tests['focused']['failures'] + tests['focused']['errors']} failures/errors.
- Phase 0–5 regression: {tests['regression']['tests']} tests, {tests['regression']['failures'] + tests['regression']['errors']} failures/errors.
- Full suite: {tests['full']['tests']} tests, {tests['full']['failures'] + tests['full']['errors']} failures/errors, {tests['full']['skipped']} skipped.
- New unexplained full-suite failures: {tests['new_unexplained_failure_count']}.
- Protected Phase 0–5 identities and active models/manifest: unchanged.

## Interpretation and recommendation

This establishes prospective lineage and reproducibility, not production
superiority. A later separately authorized shadow/model-quality phase may compare
the frozen candidate bundle descriptively and define promotion gates. It must not
retroactively tune or rewrite this Phase 6 specification.
"""


def _artifact_index(run_dir: Path) -> dict[str, Any]:
    excluded = {"artifact_index.json", "run_manifest.json", "offline_verification.json"}
    items = [
        {"path": path.relative_to(run_dir).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(run_dir.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.relative_to(run_dir).as_posix() not in excluded
    ]
    return {"artifacts": items, "artifact_count": len(items), "aggregate_sha256": sha256_json(items)}


def finalize(run_dir: Path, focused: Path, regression: Path, full: Path) -> dict[str, Any]:
    if (run_dir / "run_manifest.json").exists():
        raise Phase6EvidenceError("Finalized Phase 6 evidence is immutable")
    tests: dict[str, Any] = {}
    for label, source in (("focused", focused), ("regression", regression), ("full", full)):
        destination = run_dir / f"{label}_test_results.xml"
        shutil.copy2(source, destination)
        tests[label] = _junit(destination)
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
        raise Phase6EvidenceError(f"Final checks failed: unexplained={unexplained}, protected={protected['matches_before']}")
    report_path = run_dir / "PHASE6_IMPLEMENTATION_REPORT.md"
    report_path.write_text(_report(run_dir, tests), encoding="utf-8")
    index = _artifact_index(run_dir)
    _write_json(run_dir / "artifact_index.json", index)
    bundle = _read_json(run_dir / "candidate_bundle.json")
    manifest = {
        "schema_version": 1,
        "phase": 6,
        "run_id": run_dir.name,
        "status": "complete",
        "finalized_at": utc_now(),
        "candidate_classification": CANDIDATE_CLASSIFICATION,
        "candidate_count": 4,
        "lineage_record_count": 8,
        "candidate_bundle_id": bundle["bundle_id"],
        "candidate_bundle_aggregate_sha256": bundle["bundle_aggregate_sha256"],
        "training_specification_sha256": bundle["training_specification_sha256"],
        "evaluation_only": True,
        "production_activation_performed": False,
        "protected_phase0_to_phase5_unchanged": True,
        "no_new_unexplained_test_failures": True,
        "artifact_index_sha256": sha256_file(run_dir / "artifact_index.json"),
        "artifact_aggregate_sha256": index["aggregate_sha256"],
        "report_sha256": sha256_file(report_path),
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return manifest


def verify(run_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    try:
        spec = Phase6TrainingSpecification.load(run_dir / "training_specification.json")
        _verify_spec_sources(spec)
        proof = _read_json(run_dir / "partition_integrity.json")
        partitions = {
            name: cast(Sequence[str], proof["partitions"][name]["row_ids"])
            for name in EXPECTED_PARTITION_HASHES
        }
        verify_partition_sets(partitions, expected_hashes=EXPECTED_PARTITION_HASHES)
        inputs = _materialize_inputs()
        _verify_input_contract(_read_json(run_dir / "input_contract.json"), inputs)
        reproduction = _read_json(run_dir / "reproduction_results.json")
        if not all(
            reproduction.get(key)
            for key in ("all_inputs_reproducible", "all_fit_procedures_reproducible", "all_predictions_reproducible")
        ):
            errors.append("mandatory reproduction assertion is false")
        for item in reproduction["comparisons"]:
            if not item["prediction_reproducibility"]["reproduced"]:
                errors.append(f"prediction reproduction failed: {item['role']}")
        bundle = _read_json(run_dir / "candidate_bundle.json")
        material = {
            "training_specification_sha256": bundle["training_specification_sha256"],
            "training_partition_identity": bundle["training_partition_identity"],
            "validation_partition_identity": bundle["validation_partition_identity"],
            "models": bundle["models"],
        }
        if sha256_json(material) != bundle["bundle_aggregate_sha256"]:
            errors.append("candidate bundle aggregate mismatch")
        for item in bundle["models"]:
            artifact = ROOT / str(item["artifact"]["path"])
            if not artifact.is_file() or sha256_file(artifact) != item["artifact"]["sha256"]:
                errors.append(f"candidate artifact mismatch: {item['role']}")
            lineage_path = ROOT / str(item["lineage"]["path"])
            lineage = ModelTrainingLineageRecord.load(lineage_path)
            lineage_verification = verify_lineage_record(lineage, root=ROOT)
            if not lineage_verification["lineage_verified"] or lineage.fingerprint != item["lineage"]["fingerprint"]:
                errors.append(f"core lineage failed: {item['role']}: {lineage_verification['errors']}")
            assessment = _read_json(ROOT / str(item["assessment"]["path"]))
            assessment_result = verify_candidate_assessment(assessment, root=ROOT)
            if not assessment_result["verified"]:
                errors.append(f"candidate assessment failed: {item['role']}: {assessment_result['errors']}")
        integration_record = PredictionProvenanceRecord.load(run_dir / "integration" / "prediction_provenance.json")
        integration_verification = verify_record(integration_record, root=ROOT, verify_referenced_artifacts=True)
        integration_replay = replay_pl_1x2(integration_record, root=ROOT)
        if not integration_verification["overall_phase4_trusted"] or not integration_replay["reproduced"]:
            errors.append("Phase 4 candidate integration verification failed")
        before = _read_json(run_dir / "protected_before.json")
        if not _same(before, protected_snapshot()):
            errors.append("protected Phase 0-5 identities changed")
        phase5 = verify_phase5_evidence(PHASE5_RUN)
        if phase5["status"] != "VERIFIED":
            errors.append("Phase 5 offline verification failed")
        index = _read_json(run_dir / "artifact_index.json")
        for item in cast(Sequence[Mapping[str, Any]], index["artifacts"]):
            path = run_dir / str(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                errors.append(f"Phase 6 evidence artifact mismatch: {item['path']}")
        if index["aggregate_sha256"] != sha256_json(index["artifacts"]):
            errors.append("Phase 6 artifact aggregate mismatch")
        manifest = _read_json(run_dir / "run_manifest.json")
        if manifest["artifact_index_sha256"] != sha256_file(run_dir / "artifact_index.json"):
            errors.append("run manifest artifact-index mismatch")
        if manifest["report_sha256"] != sha256_file(run_dir / "PHASE6_IMPLEMENTATION_REPORT.md"):
            errors.append("run manifest report mismatch")
        if _read_json(run_dir / "test_results.json")["new_unexplained_failure_count"] != 0:
            errors.append("test evidence contains new unexplained failures")
    except Exception as exc:
        errors.append(f"offline verification exception: {type(exc).__name__}: {exc}")
    result = {
        "status": "VERIFIED" if not errors else "FAILED",
        "candidate_classification": CANDIDATE_CLASSIFICATION,
        "candidate_count": 4,
        "protected_phase0_to_phase5_unchanged": not any("protected Phase" in item for item in errors),
        "errors": errors,
    }
    if errors:
        raise Phase6EvidenceError(json.dumps(result, indent=2))
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("prepare", "compare", "integrate", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    fit = commands.add_parser("fit")
    fit.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    fit.add_argument("--execution", choices=("primary", "reproduction"), required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    final.add_argument("--focused-junit", type=Path, required=True)
    final.add_argument("--regression-junit", type=Path, required=True)
    final.add_argument("--full-junit", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        run_dir = args.run_dir.resolve()
        if args.command == "prepare":
            result = prepare(run_dir)
        elif args.command == "fit":
            result = fit_execution(run_dir, args.execution)
        elif args.command == "compare":
            result = compare_and_freeze(run_dir)
        elif args.command == "integrate":
            result = integrate_candidate_bundle(run_dir)
        elif args.command == "finalize":
            result = finalize(
                run_dir,
                args.focused_junit.resolve(),
                args.regression_junit.resolve(),
                args.full_junit.resolve(),
            )
        else:
            result = verify(run_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (Phase6EvidenceError, Phase6TrainingError) as exc:
        print(f"Phase 6 error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

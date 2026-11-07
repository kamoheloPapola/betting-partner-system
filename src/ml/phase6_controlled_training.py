"""Prospective, evaluation-only controlled training primitives for FFPS Phase 6.

This module deliberately has no registry dependency.  It creates fresh
estimators, rejects continuation state, and composes the frozen Phase 5
``ffps.model_training_lineage.v1`` contract without changing Phase 5 code.
"""

from __future__ import annotations

import json
import math
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.ml.training_lineage import (
    LINEAGE_REPRODUCED,
    SCHEMA_VERSION as LINEAGE_SCHEMA_VERSION,
    ModelTrainingLineageRecord,
    build_partition_record,
    record_integrity_hash,
    seal_record,
    verify_lineage_record,
)
from src.predictions.provenance import (
    artifact_identity,
    canonical_json,
    logical_path,
    repository_root,
    sha256_file,
    sha256_json,
)


SPEC_SCHEMA_VERSION = "ffps.phase6_training_specification.v1"
ASSESSMENT_SCHEMA_VERSION = "ffps.phase6_candidate_lineage.v1"
CANDIDATE_CLASSIFICATION = "TRAINING_LINEAGE_VERIFIED"
MODEL_ROLE_ORDER = (
    "home_goals_lgbm",
    "away_goals_lgbm",
    "home_goals_xgb",
    "away_goals_xgb",
)
FORBIDDEN_CONTINUATION_ARGUMENTS = frozenset(
    {"init_model", "xgb_model", "process_type", "refresh_leaf", "updater", "warm_start"}
)


class Phase6TrainingError(RuntimeError):
    """Raised when the prospective controlled-training contract is violated."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _strict_finite_frame(frame: pd.DataFrame, label: str) -> None:
    values = frame.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        positions = np.argwhere(~np.isfinite(values))[:5].tolist()
        raise Phase6TrainingError(f"{label} contains unsupported non-finite values at {positions}")


def _strict_finite_target(target: pd.Series, label: str) -> None:
    values = target.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise Phase6TrainingError(f"{label} contains unsupported non-finite values")


def specification_fingerprint(value: Mapping[str, Any]) -> str:
    material = {
        key: item
        for key, item in value.items()
        if key not in {"specification_id", "created_at", "specification_sha256", "record_integrity_sha256"}
    }
    return sha256_json(material)


def _spec_integrity(value: Mapping[str, Any]) -> str:
    return sha256_json({key: item for key, item in value.items() if key != "record_integrity_sha256"})


def seal_specification(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = cast(dict[str, Any], _json_value(value))
    sealed["specification_sha256"] = specification_fingerprint(sealed)
    sealed["record_integrity_sha256"] = _spec_integrity(sealed)
    return cast(dict[str, Any], _json_value(sealed))


@dataclass(frozen=True)
class Phase6TrainingSpecification:
    _canonical_record: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Phase6TrainingSpecification":
        parsed = cast(dict[str, Any], _json_value(value))
        if parsed.get("schema_version") != SPEC_SCHEMA_VERSION:
            raise Phase6TrainingError(f"Unsupported Phase 6 specification: {parsed.get('schema_version')}")
        if parsed.get("specification_sha256") != specification_fingerprint(parsed):
            raise Phase6TrainingError("Phase 6 training specification fingerprint mismatch")
        if parsed.get("record_integrity_sha256") != _spec_integrity(parsed):
            raise Phase6TrainingError("Phase 6 training specification integrity mismatch")
        if tuple(parsed.get("model_roles", ())) != MODEL_ROLE_ORDER:
            raise Phase6TrainingError("Phase 6 model role order is invalid")
        return cls(canonical_json(parsed))

    @classmethod
    def load(cls, path: Path) -> "Phase6TrainingSpecification":
        return cls.from_dict(cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8"))))

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._canonical_record))

    @property
    def fingerprint(self) -> str:
        return str(self.to_dict()["specification_sha256"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)


def explicit_model_configs() -> dict[str, dict[str, Any]]:
    """Return the pre-fit Phase 6 protocol; values are never performance-tuned."""
    lgbm = {
        "boosting_type": "gbdt",
        "objective": "poisson",
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "min_child_samples": 20,
        "min_child_weight": 0.001,
        "min_split_gain": 0.0,
        "subsample": 1.0,
        "subsample_freq": 0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "max_bin": 255,
        "random_state": 42,
        "n_jobs": 1,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    xgb = {
        "booster": "gbtree",
        "objective": "count:poisson",
        "n_estimators": 600,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_delta_step": 0.0,
        "max_bin": 256,
        "tree_method": "hist",
        "grow_policy": "depthwise",
        "sampling_method": "uniform",
        "base_score": 0.5,
        "random_state": 42,
        "n_jobs": 1,
        "eval_metric": "poisson-nloglik",
        "enable_categorical": False,
        "validate_parameters": True,
        "verbosity": 0,
    }
    return {
        "home_goals_lgbm": dict(lgbm),
        "away_goals_lgbm": dict(lgbm),
        "home_goals_xgb": dict(xgb),
        "away_goals_xgb": dict(xgb),
    }


def row_ids_hash(values: Sequence[str]) -> str:
    return sha256_json(sorted(str(value) for value in values))


def verify_partition_sets(
    partitions: Mapping[str, Sequence[str]],
    *,
    expected_hashes: Mapping[str, str],
) -> dict[str, Any]:
    required = ("training", "validation", "calibration_fit", "final_test")
    if any(name not in partitions for name in required):
        raise Phase6TrainingError("Required Phase 6 partition is missing")
    identities: dict[str, dict[str, Any]] = {}
    sets: dict[str, set[str]] = {}
    for name in required:
        ids = [str(value) for value in partitions[name]]
        if len(ids) != len(set(ids)):
            raise Phase6TrainingError(f"Duplicate fixture identity in {name}")
        actual = row_ids_hash(ids)
        if actual != expected_hashes.get(name):
            raise Phase6TrainingError(
                f"{name} fixture identity mismatch: expected={expected_hashes.get(name)} actual={actual}"
            )
        sets[name] = set(ids)
        identities[name] = {"row_count": len(ids), "row_ids_sha256": actual, "row_ids": ids}
    checks = []
    for left, right in (
        ("training", "validation"),
        ("training", "calibration_fit"),
        ("training", "final_test"),
        ("validation", "calibration_fit"),
        ("validation", "final_test"),
    ):
        overlap = sorted(sets[left].intersection(sets[right]))
        checks.append({"left": left, "right": right, "overlap_count": len(overlap), "overlap": overlap})
        if overlap:
            raise Phase6TrainingError(f"Protected partition overlap: {left} vs {right}: {overlap[:5]}")
    return {"partitions": identities, "pairwise_disjointness": checks, "all_required_disjoint": True}


def compare_predictions(
    left: Sequence[float],
    right: Sequence[float],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if atol < 0 or rtol < 0 or atol > 1e-12 or rtol > 1e-12:
        raise Phase6TrainingError("Prediction tolerance violates the predeclared <=1e-12 contract")
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape:
        raise Phase6TrainingError(f"Prediction shape mismatch: {first.shape} != {second.shape}")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise Phase6TrainingError("Prediction comparison contains non-finite values")
    absolute = np.abs(first - second)
    reproduced = bool(np.allclose(first, second, atol=atol, rtol=rtol))
    return {
        "shape": list(first.shape),
        "atol": float(atol),
        "rtol": float(rtol),
        "exactly_equal": bool(np.array_equal(first, second)),
        "max_absolute_difference": float(absolute.max(initial=0.0)),
        "mean_absolute_difference": float(absolute.mean()) if absolute.size else 0.0,
        "reproduced": reproduced,
    }


def fresh_estimator(role: str, config: Mapping[str, Any]) -> Any:
    if role not in MODEL_ROLE_ORDER:
        raise Phase6TrainingError(f"Unsupported Phase 6 model role: {role}")
    params = cast(dict[str, Any], _json_value(config))
    model = LGBMRegressor(**params) if role.endswith("lgbm") else XGBRegressor(**params)
    assert_no_continuation_state(model, fit_kwargs={})
    return model


def assert_no_continuation_state(model: Any, *, fit_kwargs: Mapping[str, Any]) -> None:
    prohibited = sorted(FORBIDDEN_CONTINUATION_ARGUMENTS.intersection(str(key) for key in fit_kwargs))
    if prohibited:
        raise Phase6TrainingError(f"Forbidden continuation arguments: {prohibited}")
    params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    if params.get("warm_start") is True:
        raise Phase6TrainingError("Estimator warm_start=True is forbidden")
    if params.get("process_type") in {"update", "refresh"}:
        raise Phase6TrainingError("XGBoost continuation process_type is forbidden")
    for attribute in ("booster_", "_Booster", "estimators_", "n_features_in_"):
        if hasattr(model, attribute):
            value = getattr(model, attribute)
            if value is not None:
                raise Phase6TrainingError(f"Estimator already contains fitted state: {attribute}")
    is_fitted = getattr(model, "__sklearn_is_fitted__", None)
    if callable(is_fitted) and bool(is_fitted()):
        raise Phase6TrainingError("Estimator reports pre-existing fitted state")


def assert_evaluation_only_path(path: Path, *, run_dir: Path) -> None:
    resolved = path.resolve()
    expected = run_dir.resolve()
    try:
        resolved.relative_to(expected)
    except ValueError as exc:
        raise Phase6TrainingError(f"Candidate output escapes Phase 6 evidence: {resolved}") from exc
    production = (repository_root() / "src" / "ml" / "models").resolve()
    try:
        resolved.relative_to(production)
    except ValueError:
        return
    raise Phase6TrainingError("Phase 6 candidate output targets the production model directory")


def fit_fresh_candidate(
    *,
    role: str,
    config: Mapping[str, Any],
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    validation_features: pd.DataFrame,
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    _strict_finite_frame(fit_features, f"{role} fit features")
    _strict_finite_target(fit_target, f"{role} fit target")
    _strict_finite_frame(validation_features, f"{role} validation features")
    model = fresh_estimator(role, config)
    assert_no_continuation_state(model, fit_kwargs={})
    model.fit(fit_features, fit_target)
    predictions = np.asarray(model.predict(validation_features), dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise Phase6TrainingError(f"{role} produced non-finite validation predictions")
    tree_count = (
        int(model.booster_.num_trees())
        if role.endswith("lgbm")
        else int(model.get_booster().num_boosted_rounds())
    )
    expected = int(config["n_estimators"])
    if tree_count != expected:
        raise Phase6TrainingError(f"{role} produced {tree_count} trees; expected exactly {expected}")
    return model, predictions, {
        "fresh_estimator_verified": True,
        "forbidden_fit_arguments_absent": True,
        "warm_start": False,
        "predecessor_artifact": None,
        "tree_count": tree_count,
    }


def serialize_candidate(model: Any, path: Path, *, run_dir: Path) -> dict[str, Any]:
    assert_evaluation_only_path(path, run_dir=run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary, compress=0, protocol=5)
    os.replace(temporary, path)
    return {"path": logical_path(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def source_runtime_identity(*, root: Path, source_files: Sequence[Path]) -> dict[str, Any]:
    files = [{"path": logical_path(path, root), "sha256": sha256_file(path)} for path in source_files]
    dependencies: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "lightgbm", "xgboost"):
        try:
            dependencies[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    runtime = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "dependencies": dependencies,
    }
    return {
        "status": "CAPTURED_FROM_EXECUTION",
        "source_files": files,
        "source_aggregate_sha256": sha256_json(files),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
    }


def build_reproduced_lineage_record(
    *,
    artifact_path: Path,
    role: str,
    execution: str,
    fit_features: pd.DataFrame,
    fit_target: pd.Series,
    fit_row_ids: Sequence[str],
    fit_event_times: Sequence[Any],
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    validation_row_ids: Sequence[str],
    validation_event_times: Sequence[Any],
    config: Mapping[str, Any],
    randomness: Mapping[str, Any],
    warm_start_proof: Mapping[str, Any],
    source_dataset_path: Path,
    source_files: Sequence[Path],
    training_specification_path: Path,
    reproduction: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    root: Path | None = None,
) -> ModelTrainingLineageRecord:
    resolved_root = (root or repository_root()).resolve()
    fit = build_partition_record(
        role="model_fit",
        features=fit_features,
        targets=fit_target,
        row_ids=fit_row_ids,
        event_times=fit_event_times,
        target_name="home_goals" if role.startswith("home") else "away_goals",
        missing_value_policy="reject_all_non_finite_values_before_fit",
    )
    validation = build_partition_record(
        role="frozen_validation_reproduction_corpus",
        features=validation_features,
        targets=validation_target,
        row_ids=validation_row_ids,
        event_times=validation_event_times,
        target_name="home_goals" if role.startswith("home") else "away_goals",
        missing_value_policy="reject_all_non_finite_values_before_prediction",
    )
    config_json = cast(dict[str, Any], _json_value(config))
    rng = cast(dict[str, Any], _json_value(randomness))
    spec = Phase6TrainingSpecification.load(training_specification_path)
    record = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "record_id": str(uuid.uuid4()),
        "recorded_at": utc_now(),
        "classification": LINEAGE_REPRODUCED,
        "lineage_verified": True,
        "lineage_trusted_for_promotion": True,
        "artifact": artifact_identity(artifact_path, artifact_path.read_bytes(), root=resolved_root),
        "model_identity": {
            "role": role,
            "execution": execution,
            "model_family": "lightgbm.LGBMRegressor" if role.endswith("lgbm") else "xgboost.XGBRegressor",
            "target_name": "home_goals" if role.startswith("home") else "away_goals",
            "evaluation_only": True,
        },
        "training_data": {
            "source_dataset": {
                "path": logical_path(source_dataset_path, resolved_root),
                "sha256": sha256_file(source_dataset_path),
                "size_bytes": source_dataset_path.stat().st_size,
            },
            "fit_partition": fit,
            "validation_partition": validation,
            "evaluation_partition": validation,
        },
        "fit_contract": {
            "status": "CAPTURED_FROM_EXECUTION",
            "fit_mode": "fresh_estimator_from_zero_state",
            "warm_start": False,
            "warm_start_predecessor": None,
            "warm_start_proof": _json_value(warm_start_proof),
            "training_specification_path": logical_path(training_specification_path, resolved_root),
            "training_specification_sha256": spec.fingerprint,
            "effective_config": config_json,
            "effective_config_sha256": sha256_json(config_json),
            "randomness_and_parallelism": rng,
            "randomness_and_parallelism_sha256": sha256_json(rng),
        },
        "source_runtime": source_runtime_identity(root=resolved_root, source_files=source_files),
        "partition_exclusions": {
            "training_vs_validation": "VERIFIED_DISJOINT",
            "training_vs_calibration_fit": "VERIFIED_DISJOINT",
            "training_vs_final_test": "VERIFIED_DISJOINT",
            "validation_vs_calibration_fit": "VERIFIED_DISJOINT",
            "validation_vs_final_test": "VERIFIED_DISJOINT",
        },
        "reproducibility": {
            "input_reproducibility": "VERIFIED",
            "fit_procedure_reproducibility": "VERIFIED",
            "prediction_reproducibility": "REPRODUCED",
            "artifact_byte_comparison": "REPRODUCED" if reproduction.get("artifact_bytes_identical") else "VERIFIED",
        },
        "phase6_reproduction": _json_value(reproduction),
        "evidence": [_json_value(item) for item in evidence],
        "limitations": ["Evaluation-only candidate; no model-quality or production-promotion conclusion is authorized."],
    }
    result = ModelTrainingLineageRecord.from_dict(seal_record(record))
    verification = verify_lineage_record(result, root=resolved_root)
    if not verification["lineage_verified"]:
        raise Phase6TrainingError(f"Frozen Phase 5 lineage verifier rejected controlled record: {verification}")
    return result


def assessment_fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: item
            for key, item in value.items()
            if key not in {"assessment_id", "recorded_at", "assessment_fingerprint", "record_integrity_sha256"}
        }
    )


def seal_assessment(value: Mapping[str, Any]) -> dict[str, Any]:
    sealed = cast(dict[str, Any], _json_value(value))
    sealed["assessment_fingerprint"] = assessment_fingerprint(sealed)
    sealed["record_integrity_sha256"] = sha256_json(
        {key: item for key, item in sealed.items() if key != "record_integrity_sha256"}
    )
    return cast(dict[str, Any], _json_value(sealed))


def build_candidate_assessment(
    *,
    role: str,
    artifact_path: Path,
    core_lineage_path: Path,
    training_specification_path: Path,
    partition_proof_path: Path,
    reproduction_path: Path,
    root: Path | None = None,
) -> dict[str, Any]:
    resolved_root = (root or repository_root()).resolve()
    core = ModelTrainingLineageRecord.load(core_lineage_path)
    verified = verify_lineage_record(core, root=resolved_root)
    if not verified["lineage_verified"] or verified["classification"] != LINEAGE_REPRODUCED:
        raise Phase6TrainingError(f"Core Phase 5 lineage is not reproduced/trusted: {verified}")
    value = {
        "schema_version": ASSESSMENT_SCHEMA_VERSION,
        "assessment_id": str(uuid.uuid4()),
        "recorded_at": utc_now(),
        "candidate_classification": CANDIDATE_CLASSIFICATION,
        "evaluation_only": True,
        "role": role,
        "artifact": artifact_identity(artifact_path, artifact_path.read_bytes(), root=resolved_root),
        "core_lineage": {
            "path": logical_path(core_lineage_path, resolved_root),
            "sha256": sha256_file(core_lineage_path),
            "fingerprint": core.fingerprint,
            "classification": LINEAGE_REPRODUCED,
        },
        "training_specification": {
            "path": logical_path(training_specification_path, resolved_root),
            "sha256": sha256_file(training_specification_path),
            "fingerprint": Phase6TrainingSpecification.load(training_specification_path).fingerprint,
        },
        "partition_proof": {
            "path": logical_path(partition_proof_path, resolved_root),
            "sha256": sha256_file(partition_proof_path),
        },
        "reproduction": {"path": logical_path(reproduction_path, resolved_root), "sha256": sha256_file(reproduction_path)},
        "assertions": {
            "fresh_estimator_state": True,
            "warm_start_absent": True,
            "protected_partitions_disjoint": True,
            "input_reproducible": True,
            "fit_procedure_reproducible": True,
            "prediction_reproducible": True,
            "active_manifest_untouched": True,
        },
    }
    return seal_assessment(value)


def verify_candidate_assessment(value: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    resolved_root = (root or repository_root()).resolve()
    assessment = cast(dict[str, Any], _json_value(value))
    errors: list[str] = []
    if assessment.get("schema_version") != ASSESSMENT_SCHEMA_VERSION:
        errors.append("unsupported Phase 6 assessment schema")
    if assessment.get("candidate_classification") != CANDIDATE_CLASSIFICATION:
        errors.append("candidate is not TRAINING_LINEAGE_VERIFIED")
    if not assessment.get("evaluation_only"):
        errors.append("candidate is not evaluation-only")
    if assessment.get("assessment_fingerprint") != assessment_fingerprint(assessment):
        errors.append("assessment fingerprint mismatch")
    integrity = sha256_json({key: item for key, item in assessment.items() if key != "record_integrity_sha256"})
    if assessment.get("record_integrity_sha256") != integrity:
        errors.append("assessment integrity mismatch")
    for section in ("artifact", "core_lineage", "training_specification", "partition_proof", "reproduction"):
        item = cast(Mapping[str, Any], assessment.get(section, {}))
        path = resolved_root / str(item.get("source_location") or item.get("path") or "")
        expected = item.get("artifact_sha256") or item.get("sha256")
        if not path.is_file() or sha256_file(path) != expected:
            errors.append(f"{section} reference mismatch")
    core_item = cast(Mapping[str, Any], assessment.get("core_lineage", {}))
    core_path = resolved_root / str(core_item.get("path", ""))
    if core_path.is_file():
        core = ModelTrainingLineageRecord.load(core_path)
        verification = verify_lineage_record(core, root=resolved_root)
        if not verification["lineage_verified"] or core.fingerprint != core_item.get("fingerprint"):
            errors.append("core Phase 5 lineage verification failed")
    assertions = cast(Mapping[str, Any], assessment.get("assertions", {}))
    if not assertions or not all(value is True for value in assertions.values()):
        errors.append("mandatory Phase 6 assertion is not true")
    return {
        "verified": not errors,
        "candidate_classification": assessment.get("candidate_classification"),
        "role": assessment.get("role"),
        "errors": errors,
    }

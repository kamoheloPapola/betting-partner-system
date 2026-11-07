"""Immutable training-lineage records for fitted model artifacts.

The record separates two questions that must never be conflated:

* is this assessment internally intact and bound to the stated artifact?; and
* is the historical training execution itself reproducible and proven?

Legacy artifacts may answer the first question yes and the second no.  That is
the fail-closed result used by Phase 5 for the four active PL goal models.
"""

from __future__ import annotations

import json
import os
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd

from src.predictions.provenance import (
    artifact_identity,
    canonical_json,
    logical_path,
    repository_root,
    sha256_file,
    sha256_json,
)


SCHEMA_VERSION = "ffps.model_training_lineage.v1"
HISTORICAL_LINEAGE_VERIFIED = "HISTORICAL_LINEAGE_VERIFIED"
LINEAGE_REPRODUCED = "LINEAGE_REPRODUCED"
TRAINING_LINEAGE_PARTIAL = "TRAINING_LINEAGE_PARTIAL"
LEGACY_TRAINING_LINEAGE_UNPROVEN = "LEGACY_TRAINING_LINEAGE_UNPROVEN"
TRUSTED_CLASSIFICATIONS = frozenset({HISTORICAL_LINEAGE_VERIFIED, LINEAGE_REPRODUCED})


class TrainingLineageError(RuntimeError):
    """Raised when a training-lineage record cannot be safely constructed."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    """Round-trip through the shared strict canonical JSON contract."""
    return json.loads(canonical_json(value))


def _fingerprint_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"record_id", "recorded_at", "training_lineage_fingerprint", "record_integrity_sha256"}
    }


def training_lineage_fingerprint(record: Mapping[str, Any]) -> str:
    return sha256_json(_fingerprint_payload(record))


def record_integrity_hash(record: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in record.items() if key != "record_integrity_sha256"})


def seal_record(record: Mapping[str, Any]) -> dict[str, Any]:
    sealed = cast(dict[str, Any], _json_value(record))
    sealed["training_lineage_fingerprint"] = training_lineage_fingerprint(sealed)
    sealed["record_integrity_sha256"] = record_integrity_hash(sealed)
    return cast(dict[str, Any], _json_value(sealed))


@dataclass(frozen=True)
class ModelTrainingLineageRecord:
    """Immutable canonical representation of one fitted artifact's lineage."""

    _canonical_record: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelTrainingLineageRecord":
        parsed = cast(dict[str, Any], _json_value(value))
        if parsed.get("schema_version") != SCHEMA_VERSION:
            raise TrainingLineageError(f"Unsupported training-lineage schema: {parsed.get('schema_version')}")
        return cls(canonical_json(parsed))

    @classmethod
    def load(cls, path: Path) -> "ModelTrainingLineageRecord":
        return cls.from_dict(cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8"))))

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._canonical_record))

    @property
    def fingerprint(self) -> str:
        return str(self.to_dict()["training_lineage_fingerprint"])

    def canonical_bytes(self) -> bytes:
        return self._canonical_record.encode("utf-8")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)


def _series_values(values: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if pd.isna(value):
            result.append(None)
        elif isinstance(value, np.generic):
            result.append(value.item())
        elif isinstance(value, pd.Timestamp):
            result.append(value.isoformat())
        else:
            result.append(value)
    return cast(list[Any], _json_value(result))


def build_partition_record(
    *,
    role: str,
    features: pd.DataFrame,
    targets: pd.Series,
    row_ids: Sequence[Any],
    event_times: Sequence[Any] | None = None,
    target_name: str,
    missing_value_policy: str,
) -> dict[str, Any]:
    """Bind ordered rows, features and targets to one exact fit/evaluation partition."""
    if len(features) != len(targets) or len(features) != len(row_ids):
        raise TrainingLineageError("Partition features, targets and row IDs have different lengths")
    if event_times is not None and len(event_times) != len(features):
        raise TrainingLineageError("Partition event times have a different length")

    names = [str(column) for column in features.columns]
    if len(names) != len(set(names)):
        raise TrainingLineageError("Duplicate ordered feature names are forbidden")
    ids = [str(value) for value in row_ids]
    if any(not value for value in ids):
        raise TrainingLineageError("Blank training row identity is forbidden")
    times = [None] * len(ids) if event_times is None else _series_values(event_times)
    target_values = _series_values(list(targets))
    rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(features.iterrows()):
        rows.append(
            {
                "row_id": ids[position],
                "event_time": times[position],
                "feature_values": _series_values(list(row)),
                "target": target_values[position],
            }
        )
    dtypes = [str(dtype) for dtype in features.dtypes]
    matrix_material = {
        "ordered_feature_names": names,
        "feature_dtypes": dtypes,
        "rows": [{"row_id": row["row_id"], "feature_values": row["feature_values"]} for row in rows],
    }
    target_material = [
        {"row_id": row["row_id"], "target": row["target"]}
        for row in rows
    ]
    return {
        "status": "CAPTURED_FROM_EXECUTION",
        "role": role,
        "row_count": len(rows),
        "ordered_feature_names": names,
        "feature_schema_sha256": sha256_json(names),
        "feature_dtypes": dtypes,
        "feature_shape": [len(rows), len(names)],
        "missing_value_policy": missing_value_policy,
        "row_ids_ordered_sha256": sha256_json(ids),
        "row_ids_set_sha256": sha256_json(sorted(set(ids))),
        "duplicate_row_id_count": len(ids) - len(set(ids)),
        "event_time_min": min((value for value in times if value is not None), default=None),
        "event_time_max": max((value for value in times if value is not None), default=None),
        "feature_matrix_sha256": sha256_json(matrix_material),
        "target_name": target_name,
        "target_vector_sha256": sha256_json(target_material),
        "rows": rows,
    }


def _source_runtime_identity(root: Path, source_files: Sequence[Path]) -> dict[str, Any]:
    files = [
        {"path": logical_path(path, root), "sha256": sha256_file(path)}
        for path in source_files
    ]
    dependencies: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "joblib", "lightgbm", "xgboost"):
        try:
            dependencies[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    runtime = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "dependencies": dependencies,
    }
    return {
        "status": "CAPTURED_FROM_EXECUTION",
        "source_files": files,
        "source_aggregate_sha256": sha256_json(files),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
    }


def checkpoint_identity(path: Path, *, root: Path | None = None) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = path.read_bytes()
    return {**artifact_identity(path, payload, root=root), "role": "warm_start_predecessor"}


def build_future_training_record(
    *,
    artifact_path: Path,
    model_role: str,
    manifest_key: str,
    registry_slot: str,
    model_family: str,
    target_name: str,
    fit_features: pd.DataFrame,
    fit_targets: pd.Series,
    fit_row_ids: Sequence[Any],
    fit_event_times: Sequence[Any],
    evaluation_features: pd.DataFrame,
    evaluation_targets: pd.Series,
    evaluation_row_ids: Sequence[Any],
    evaluation_event_times: Sequence[Any],
    effective_config: Mapping[str, Any],
    randomness: Mapping[str, Any],
    fit_mode: str,
    warm_start_predecessor: Mapping[str, Any] | None,
    source_dataset_path: Path,
    source_files: Sequence[Path],
    root: Path | None = None,
) -> ModelTrainingLineageRecord:
    resolved_root = (root or repository_root()).resolve()
    artifact = artifact_identity(artifact_path, artifact_path.read_bytes(), root=resolved_root)
    fit = build_partition_record(
        role="model_fit",
        features=fit_features,
        targets=fit_targets,
        row_ids=fit_row_ids,
        event_times=fit_event_times,
        target_name=target_name,
        missing_value_policy="pandas_numeric_values_passed_to_estimator_without_imputation",
    )
    evaluation = build_partition_record(
        role="held_out_evaluation_only",
        features=evaluation_features,
        targets=evaluation_targets,
        row_ids=evaluation_row_ids,
        event_times=evaluation_event_times,
        target_name=target_name,
        missing_value_policy="pandas_numeric_values_passed_to_estimator_without_imputation",
    )
    config = cast(dict[str, Any], _json_value(effective_config))
    rng = cast(dict[str, Any], _json_value(randomness))
    source_dataset = {
        "path": logical_path(source_dataset_path, resolved_root),
        "sha256": sha256_file(source_dataset_path),
        "size_bytes": source_dataset_path.stat().st_size,
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": str(uuid.uuid4()),
        "recorded_at": utc_now(),
        "classification": TRAINING_LINEAGE_PARTIAL,
        "lineage_verified": False,
        "lineage_trusted_for_promotion": False,
        "artifact": artifact,
        "model_identity": {
            "role": model_role,
            "manifest_key": manifest_key,
            "registry_slot": registry_slot,
            "model_family": model_family,
            "target_name": target_name,
        },
        "training_data": {
            "source_dataset": source_dataset,
            "fit_partition": fit,
            "validation_partition": {
                "status": "NOT_USED",
                "role": "none",
                "row_count": 0,
                "used_for_early_stopping": False,
                "used_for_model_selection": False,
            },
            "evaluation_partition": evaluation,
        },
        "fit_contract": {
            "status": "CAPTURED_FROM_EXECUTION",
            "fit_mode": fit_mode,
            "warm_start_predecessor": _json_value(warm_start_predecessor),
            "effective_config": config,
            "effective_config_sha256": sha256_json(config),
            "randomness_and_parallelism": rng,
            "randomness_and_parallelism_sha256": sha256_json(rng),
        },
        "source_runtime": _source_runtime_identity(resolved_root, source_files),
        "partition_exclusions": {
            "status": "NOT_SUPPLIED_TO_TRAINER",
            "phase1_overlap": "UNPROVEN",
            "phase3_calibration_overlap": "UNPROVEN",
            "phase3_final_test_overlap": "UNPROVEN",
        },
        "reproducibility": {
            "input_reproducibility": "CAPTURED",
            "fit_reproducibility": "NOT_REPLAYED",
            "prediction_reproducibility": "NOT_REPLAYED",
            "artifact_byte_reproducibility": "NOT_REPLAYED",
        },
        "limitations": [
            "Phase 1 and Phase 3 exclusion identities were not supplied to this training invocation.",
            "A second isolated fit has not established prediction or byte-for-byte reproducibility.",
        ],
        "evidence": [],
    }
    return ModelTrainingLineageRecord.from_dict(seal_record(record))


def build_legacy_unproven_record(
    *,
    artifact_path: Path,
    model_role: str,
    manifest_key: str,
    registry_slot: str,
    model_family: str,
    target_name: str,
    ordered_feature_names: Sequence[str],
    embedded_config: Mapping[str, Any],
    embedded_randomness: Mapping[str, Any],
    audit_findings: Sequence[str],
    evidence: Sequence[Mapping[str, Any]],
    root: Path | None = None,
) -> ModelTrainingLineageRecord:
    """Create an honest, mechanically verifiable assessment of a legacy artifact."""
    resolved_root = (root or repository_root()).resolve()
    artifact = artifact_identity(artifact_path, artifact_path.read_bytes(), root=resolved_root)
    names = [str(name) for name in ordered_feature_names]
    config = cast(dict[str, Any], _json_value(embedded_config))
    rng = cast(dict[str, Any], _json_value(embedded_randomness))
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_id": str(uuid.uuid4()),
        "recorded_at": utc_now(),
        "classification": LEGACY_TRAINING_LINEAGE_UNPROVEN,
        "lineage_verified": False,
        "lineage_trusted_for_promotion": False,
        "artifact": artifact,
        "model_identity": {
            "role": model_role,
            "manifest_key": manifest_key,
            "registry_slot": registry_slot,
            "model_family": model_family,
            "target_name": target_name,
        },
        "training_data": {
            "source_dataset": {"status": "UNKNOWN"},
            "fit_partition": {"status": "UNKNOWN", "row_count": None},
            "validation_partition": {"status": "UNKNOWN", "row_count": None},
            "evaluation_partition": {"status": "UNKNOWN", "row_count": None},
            "target_vector": {"status": "UNKNOWN"},
        },
        "feature_contract": {
            "status": "VERIFIED_FROM_ARTIFACT",
            "ordered_feature_names": names,
            "feature_schema_sha256": sha256_json(names),
            "training_feature_matrix": "UNKNOWN",
            "feature_dtypes": "UNKNOWN",
            "missing_value_policy": "UNKNOWN",
        },
        "fit_contract": {
            "status": "PARTIAL_FROM_ARTIFACT",
            "embedded_config": config,
            "embedded_config_sha256": sha256_json(config),
            "embedded_randomness": rng,
            "embedded_randomness_sha256": sha256_json(rng),
            "fit_invocation": "UNKNOWN",
            "warm_start_history": "UNKNOWN",
        },
        "source_runtime": {"status": "UNKNOWN", "historical_source_binding": False},
        "partition_exclusions": {
            "phase1_overlap": "UNPROVEN",
            "phase3_calibration_overlap": "UNPROVEN",
            "phase3_final_test_overlap": "UNPROVEN",
        },
        "reproducibility": {
            "input_reproducibility": "NOT_ESTABLISHED",
            "fit_reproducibility": "NOT_ESTABLISHED",
            "prediction_reproducibility": "NOT_ESTABLISHED",
            "artifact_byte_reproducibility": "NOT_ESTABLISHED",
        },
        "audit_findings": list(audit_findings),
        "evidence": [_json_value(item) for item in evidence],
    }
    return ModelTrainingLineageRecord.from_dict(seal_record(record))


def _verify_partition(partition: Mapping[str, Any], errors: list[str]) -> None:
    rows = cast(list[dict[str, Any]], partition.get("rows", []))
    ids = [str(row.get("row_id", "")) for row in rows]
    names = cast(list[str], partition.get("ordered_feature_names", []))
    dtypes = cast(list[str], partition.get("feature_dtypes", []))
    if partition.get("row_count") != len(rows):
        errors.append(f"{partition.get('role')} row count mismatch")
    if partition.get("feature_shape") != [len(rows), len(names)]:
        errors.append(f"{partition.get('role')} feature shape mismatch")
    if partition.get("duplicate_row_id_count") != len(ids) - len(set(ids)):
        errors.append(f"{partition.get('role')} duplicate row count mismatch")
    checks = {
        "feature_schema_sha256": sha256_json(names),
        "row_ids_ordered_sha256": sha256_json(ids),
        "row_ids_set_sha256": sha256_json(sorted(set(ids))),
        "feature_matrix_sha256": sha256_json(
            {
                "ordered_feature_names": names,
                "feature_dtypes": dtypes,
                "rows": [{"row_id": row["row_id"], "feature_values": row["feature_values"]} for row in rows],
            }
        ),
        "target_vector_sha256": sha256_json(
            [{"row_id": row["row_id"], "target": row["target"]} for row in rows]
        ),
    }
    for field, actual in checks.items():
        if partition.get(field) != actual:
            errors.append(f"{partition.get('role')} {field} mismatch")


def verify_lineage_record(
    value: ModelTrainingLineageRecord | Mapping[str, Any],
    *,
    root: Path | None = None,
    verify_referenced_artifact: bool = True,
) -> dict[str, Any]:
    record = value.to_dict() if isinstance(value, ModelTrainingLineageRecord) else cast(dict[str, Any], _json_value(value))
    errors: list[str] = []
    resolved_root = (root or repository_root()).resolve()
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"unsupported schema: {record.get('schema_version')}")
    try:
        uuid.UUID(str(record.get("record_id")))
    except (TypeError, ValueError):
        errors.append("record_id is not a UUID")
    if record.get("training_lineage_fingerprint") != training_lineage_fingerprint(record):
        errors.append("training lineage fingerprint mismatch")
    if record.get("record_integrity_sha256") != record_integrity_hash(record):
        errors.append("record integrity mismatch")

    classification = str(record.get("classification"))
    declared_trusted = bool(record.get("lineage_trusted_for_promotion"))
    if declared_trusted != (classification in TRUSTED_CLASSIFICATIONS):
        errors.append("classification and lineage_trusted_for_promotion disagree")
    if bool(record.get("lineage_verified")) != (classification in TRUSTED_CLASSIFICATIONS):
        errors.append("classification and lineage_verified disagree")

    artifact = cast(Mapping[str, Any], record.get("artifact", {}))
    if verify_referenced_artifact:
        artifact_path = resolved_root / str(artifact.get("source_location", ""))
        if not artifact_path.is_file():
            errors.append(f"artifact missing: {artifact_path}")
        else:
            if sha256_file(artifact_path) != artifact.get("artifact_sha256"):
                errors.append("artifact SHA-256 mismatch")
            if artifact_path.stat().st_size != artifact.get("size_bytes"):
                errors.append("artifact size mismatch")

    if classification == LEGACY_TRAINING_LINEAGE_UNPROVEN:
        feature = cast(Mapping[str, Any], record.get("feature_contract", {}))
        names = cast(Sequence[str], feature.get("ordered_feature_names", []))
        if feature.get("feature_schema_sha256") != sha256_json(names):
            errors.append("legacy embedded feature schema mismatch")
        fit = cast(Mapping[str, Any], record.get("fit_contract", {}))
        if fit.get("embedded_config_sha256") != sha256_json(fit.get("embedded_config", {})):
            errors.append("legacy embedded config mismatch")
        if fit.get("embedded_randomness_sha256") != sha256_json(fit.get("embedded_randomness", {})):
            errors.append("legacy embedded randomness mismatch")
        required_unknown = [
            cast(Mapping[str, Any], record["training_data"])["fit_partition"].get("status") == "UNKNOWN",
            fit.get("fit_invocation") == "UNKNOWN",
            fit.get("warm_start_history") == "UNKNOWN",
            cast(Mapping[str, Any], record["source_runtime"]).get("status") == "UNKNOWN",
        ]
        if not all(required_unknown):
            errors.append("legacy unproven record suppresses an unknown lineage dimension")
        for item in cast(Sequence[Mapping[str, Any]], record.get("evidence", [])):
            if "path" not in item or "sha256" not in item:
                continue
            evidence_path = resolved_root / str(item["path"])
            if not evidence_path.is_file() or sha256_file(evidence_path) != item["sha256"]:
                errors.append(f"legacy evidence reference mismatch: {item['path']}")
    else:
        data = cast(Mapping[str, Any], record.get("training_data", {}))
        fit = cast(Mapping[str, Any], data.get("fit_partition", {}))
        evaluation = cast(Mapping[str, Any], data.get("evaluation_partition", {}))
        _verify_partition(fit, errors)
        _verify_partition(evaluation, errors)
        fit_ids = {str(row["row_id"]) for row in cast(Sequence[Mapping[str, Any]], fit.get("rows", []))}
        eval_ids = {str(row["row_id"]) for row in cast(Sequence[Mapping[str, Any]], evaluation.get("rows", []))}
        if fit_ids.intersection(eval_ids):
            errors.append("fit/evaluation row identity overlap")
        contract = cast(Mapping[str, Any], record.get("fit_contract", {}))
        if contract.get("effective_config_sha256") != sha256_json(contract.get("effective_config", {})):
            errors.append("effective config mismatch")
        if contract.get("randomness_and_parallelism_sha256") != sha256_json(
            contract.get("randomness_and_parallelism", {})
        ):
            errors.append("randomness contract mismatch")
        dataset = cast(Mapping[str, Any], data.get("source_dataset", {}))
        dataset_path = resolved_root / str(dataset.get("path", ""))
        if not dataset_path.is_file():
            errors.append(f"training source dataset missing: {dataset_path}")
        else:
            if sha256_file(dataset_path) != dataset.get("sha256"):
                errors.append("training source dataset SHA-256 mismatch")
            if dataset_path.stat().st_size != dataset.get("size_bytes"):
                errors.append("training source dataset size mismatch")
        source_runtime = cast(Mapping[str, Any], record.get("source_runtime", {}))
        source_files = cast(Sequence[Mapping[str, Any]], source_runtime.get("source_files", []))
        for item in source_files:
            source_path = resolved_root / str(item.get("path", ""))
            if not source_path.is_file() or sha256_file(source_path) != item.get("sha256"):
                errors.append(f"training source file mismatch: {item.get('path')}")
        if source_runtime.get("source_aggregate_sha256") != sha256_json(source_files):
            errors.append("training source aggregate mismatch")
        if source_runtime.get("runtime_sha256") != sha256_json(source_runtime.get("runtime", {})):
            errors.append("training runtime identity mismatch")

    if classification in TRUSTED_CLASSIFICATIONS:
        exclusions = cast(Mapping[str, Any], record.get("partition_exclusions", {}))
        if not exclusions or any(value != "VERIFIED_DISJOINT" for value in exclusions.values()):
            errors.append("trusted lineage requires verified exclusion disjointness")
        reproduction = cast(Mapping[str, Any], record.get("reproducibility", {}))
        if not reproduction or any(value not in {"VERIFIED", "REPRODUCED"} for value in reproduction.values()):
            errors.append("trusted lineage requires completed reproducibility checks")

    assessment_verified = not errors
    lineage_verified = assessment_verified and classification in TRUSTED_CLASSIFICATIONS
    return {
        "assessment_verified": assessment_verified,
        "lineage_verified": lineage_verified,
        "classification": classification,
        "lineage_trusted_for_promotion": lineage_verified,
        "training_lineage_fingerprint": record.get("training_lineage_fingerprint"),
        "errors": errors,
    }


def persist_verified_record(record: ModelTrainingLineageRecord, path: Path, *, root: Path | None = None) -> dict[str, Any]:
    """Atomically persist and re-read a record; fail before registry mutation."""
    record.save(path)
    persisted = ModelTrainingLineageRecord.load(path)
    verification = verify_lineage_record(persisted, root=root)
    if not verification["assessment_verified"]:
        raise TrainingLineageError(f"Persisted training lineage did not verify: {verification['errors']}")
    return verification


def registry_lineage_metadata(record: ModelTrainingLineageRecord, path: Path, *, models_dir: Path) -> dict[str, Any]:
    payload = record.to_dict()
    classification = str(payload["classification"])
    return {
        "training_lineage_schema": SCHEMA_VERSION,
        "training_lineage_status": classification,
        "training_lineage_record": path.resolve().relative_to(models_dir.resolve()).as_posix(),
        "training_lineage_fingerprint": payload["training_lineage_fingerprint"],
        "training_lineage_record_sha256": sha256_file(path),
        "training_lineage_eligible": classification in TRUSTED_CLASSIFICATIONS,
    }

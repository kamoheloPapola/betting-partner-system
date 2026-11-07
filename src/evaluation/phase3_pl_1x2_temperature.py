"""Leakage-safe PL 1X2 temperature calibration over frozen Phase 1 predictions.

This module is evaluation-only.  It never trains or loads production models and
never mutates the Phase 1 replay.  The Phase 1 final-holdout predictions are
split at the predeclared 2025/26 season boundary: 2024/25 fits one scalar
temperature and available 2025/26 fixtures form the protocol-isolated final
test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


ROOT = Path(__file__).resolve().parents[2]
PHASE1_RUN = ROOT / "baselines" / "phase1" / "phase1-pl-1x2-79110f9a-20240801-v1"
PHASE1_PREDICTIONS = PHASE1_RUN / "fixture_predictions.jsonl"
PHASE1_PARTITIONS = PHASE1_RUN / "partition_manifest.json"
PHASE1_INDEX = PHASE1_RUN / "artifact_index.json"
PHASE1_MODEL_BUNDLE = PHASE1_RUN / "model_bundle_manifest.json"
FEATURE_MATRIX = ROOT / "data" / "features" / "feature_matrix.csv"

CALIBRATION_END_EXCLUSIVE = date(2025, 8, 1)
CLASS_ORDER = ("home", "draw", "away")
LABEL_ORDER = ("H", "D", "A")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABEL_ORDER)}
BOOKMAKER_TOKENS = (
    "bookmaker",
    "odds",
    "implied_probability",
    "implied_prob",
    "market_price",
    "opening_price",
    "closing_price",
    "consensus_price",
)
RELIABILITY_BINS = 10
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95

# Frozen before final-test metrics are inspected.  Loss deltas are calibrated -
# raw, so negative values favor calibration.
DECISION_RULE = {
    "supported": {
        "log_loss_point": "< 0",
        "log_loss_interval_upper": "<= 0",
        "brier_point": "<= 0",
        "brier_interval_upper": "<= 0.005",
        "probability_invariants": "pass",
        "reliability_pathology": "false",
    },
    "rejected": {
        "log_loss_point_material_harm": ">= 0.01",
        "brier_point_material_harm": ">= 0.005",
        "or_loss_interval_lower": "> 0",
        "probability_invariants": "pass required",
    },
    "otherwise": "INCONCLUSIVE",
    "reliability_pathology_thresholds": {
        "top_label_ece_increase": 0.03,
        "any_classwise_ece_increase": 0.05,
    },
}

REQUIRED_PROVENANCE_FIELDS = {
    "calibrator_id",
    "method",
    "market",
    "league",
    "class_order",
    "temperature",
    "log_temperature",
    "created_at",
    "raw_prediction_input_artifact",
    "calibration_fit_fixture_ids",
    "calibration_fit_fixture_ids_sha256",
    "calibration_fit_date_range",
    "calibration_fit_count",
    "final_test_fixture_ids_sha256",
    "final_test_exclusion_assertion",
    "objective",
    "optimizer",
    "random_seed",
    "source_code",
    "source_state",
    "runtime",
    "dependency_fingerprint",
    "upstream_model_bundle",
    "feature_schema",
    "simulator",
    "no_bookmaker_input_assertion",
    "evaluation_only",
    "production_eligible",
    "trust_status",
}


class Phase3TemperatureIntegrityError(RuntimeError):
    """Raised whenever the authorized Phase 3 boundary cannot be proven."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_ids_hash(values: Iterable[str]) -> str:
    return sha256_json(sorted(str(value) for value in values))


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")


def _fixture_key(fixture_date: str, league: str, home_team: str, away_team: str) -> str:
    normalize = lambda value: " ".join(str(value).strip().upper().split())
    return "|".join((fixture_date, normalize(league), normalize(home_team), normalize(away_team)))


def _phase1_artifact_hash(relative_name: str) -> str:
    index = read_json(PHASE1_INDEX)
    suffix = "/" + relative_name.replace("\\", "/")
    matches = [item for item in index["artifacts"] if str(item["path"]).replace("\\", "/").endswith(suffix)]
    if len(matches) != 1:
        raise Phase3TemperatureIntegrityError(f"Phase 1 artifact index does not uniquely identify {relative_name}")
    return str(matches[0]["sha256"])


def _load_phase1_predictions() -> list[dict[str, Any]]:
    expected = _phase1_artifact_hash("fixture_predictions.jsonl")
    actual = sha256_file(PHASE1_PREDICTIONS)
    if actual != expected:
        raise Phase3TemperatureIntegrityError(
            f"Frozen Phase 1 predictions changed: expected={expected} actual={actual}"
        )
    records: list[dict[str, Any]] = []
    with PHASE1_PREDICTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(cast(dict[str, Any], json.loads(line)))
    return records


def _feature_matrix_identities(required_ids: set[str]) -> dict[str, str]:
    identities: dict[str, str] = {}
    with FEATURE_MATRIX.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            match_hash = str(row.get("match_hash", ""))
            if match_hash not in required_ids:
                continue
            fixture_date = str(row.get("date", ""))[:10]
            identities[match_hash] = _fixture_key(
                fixture_date,
                str(row.get("league", "")),
                str(row.get("home_team", "")),
                str(row.get("away_team", "")),
            )
    missing = sorted(required_ids - set(identities))
    if missing:
        raise Phase3TemperatureIntegrityError(f"Feature matrix is missing partition identities: {missing[:5]}")
    return identities


def _compact_raw_record(record: Mapping[str, Any], *, source_artifact_sha256: str) -> dict[str, Any]:
    fixture = cast(Mapping[str, Any], record["fixture"])
    output = cast(Mapping[str, Any], record["output"])
    probabilities = cast(Mapping[str, Any], output["probabilities"])
    raw = {
        "home": float(probabilities["H"]),
        "draw": float(probabilities["D"]),
        "away": float(probabilities["A"]),
    }
    raw_hash = sha256_json({"class_order": list(CLASS_ORDER), "probabilities": raw})
    return {
        "schema_version": 1,
        "prediction_run_id": "phase1-pl-1x2-79110f9a-20240801-v1",
        "match_hash": str(fixture["match_hash"]),
        "fixture": {
            "date": str(fixture["fixture_date"]),
            "league": str(fixture["league"]),
            "home_team": str(fixture["home_team"]),
            "away_team": str(fixture["away_team"]),
        },
        "prediction_time": str(record["prediction_time"]),
        "actual_result": str(fixture["actual_1x2"]),
        "raw_probabilities": raw,
        "raw_prediction_hash": raw_hash,
        "phase1_record_hash": sha256_json(record),
        "phase1_raw_artifact_sha256": source_artifact_sha256,
        "pathway": record["pathway"],
        "rng": record["rng"],
        "ensemble": record["ensemble"],
        "temporal_evidence": {
            "strict_cutoff_passed": bool(cast(Mapping[str, Any], record["temporal_evidence"])["strict_cutoff_passed"]),
            "same_day_sources_excluded": bool(cast(Mapping[str, Any], record["temporal_evidence"])["same_day_sources_excluded"]),
            "max_source_date": str(cast(Mapping[str, Any], record["temporal_evidence"])["max_source_date"]),
            "prediction_time": str(cast(Mapping[str, Any], record["temporal_evidence"])["prediction_time"]),
        },
    }


@dataclass(frozen=True)
class Phase3PartitionContract:
    model_train: tuple[dict[str, Any], ...]
    model_validation: tuple[dict[str, Any], ...]
    calibration_fit: tuple[dict[str, Any], ...]
    final_test: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]


def build_partition_contract() -> Phase3PartitionContract:
    """Reuse Phase 1 identities and season-split its frozen raw replay."""
    phase1_manifest = read_json(PHASE1_PARTITIONS)
    source_hash = _phase1_artifact_hash("fixture_predictions.jsonl")
    phase1_records = _load_phase1_predictions()

    compact = [_compact_raw_record(record, source_artifact_sha256=source_hash) for record in phase1_records]
    calibration_fit = tuple(
        record for record in compact
        if date.fromisoformat(str(cast(Mapping[str, Any], record["fixture"])["date"])) < CALIBRATION_END_EXCLUSIVE
    )
    final_test = tuple(
        record for record in compact
        if date.fromisoformat(str(cast(Mapping[str, Any], record["fixture"])["date"])) >= CALIBRATION_END_EXCLUSIVE
    )
    if len(calibration_fit) + len(final_test) != len(compact):
        raise Phase3TemperatureIntegrityError("Not every Phase 1 prediction entered an authorized Phase 3 partition")

    train_ids = tuple(str(value) for value in phase1_manifest["partitions"]["training"]["match_hashes"])
    validation_ids = tuple(str(value) for value in phase1_manifest["partitions"]["validation"]["match_hashes"])
    identity_map = _feature_matrix_identities(set(train_ids) | set(validation_ids))
    model_train = tuple({"match_hash": value, "fixture_identity": identity_map[value]} for value in train_ids)
    model_validation = tuple({"match_hash": value, "fixture_identity": identity_map[value]} for value in validation_ids)

    partition_records: dict[str, Sequence[Mapping[str, Any]]] = {
        "model_train": model_train,
        "model_validation": model_validation,
        "calibration_fit": calibration_fit,
        "final_test": final_test,
    }
    hash_sets: dict[str, set[str]] = {}
    fixture_sets: dict[str, set[str]] = {}
    for name, rows in partition_records.items():
        hashes = [str(row["match_hash"]) for row in rows]
        if len(hashes) != len(set(hashes)):
            raise Phase3TemperatureIntegrityError(f"Duplicate match hash in {name}")
        hash_sets[name] = set(hashes)
        if name in {"model_train", "model_validation"}:
            fixture_keys = [str(row["fixture_identity"]) for row in rows]
        else:
            fixture_keys = [
                _fixture_key(
                    str(cast(Mapping[str, Any], row["fixture"])["date"]),
                    str(cast(Mapping[str, Any], row["fixture"])["league"]),
                    str(cast(Mapping[str, Any], row["fixture"])["home_team"]),
                    str(cast(Mapping[str, Any], row["fixture"])["away_team"]),
                )
                for row in rows
            ]
        if len(fixture_keys) != len(set(fixture_keys)):
            raise Phase3TemperatureIntegrityError(f"Duplicate canonical fixture identity in {name}")
        fixture_sets[name] = set(fixture_keys)

    names = tuple(partition_records)
    overlap_evidence: list[dict[str, Any]] = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            hash_overlap = sorted(hash_sets[left] & hash_sets[right])
            fixture_overlap = sorted(fixture_sets[left] & fixture_sets[right])
            overlap_evidence.append({
                "left": left,
                "right": right,
                "match_hash_overlap_count": len(hash_overlap),
                "canonical_fixture_overlap_count": len(fixture_overlap),
            })
            if hash_overlap or fixture_overlap:
                raise Phase3TemperatureIntegrityError(
                    f"Partition overlap {left}/{right}: hashes={hash_overlap[:3]} fixtures={fixture_overlap[:3]}"
                )

    def summary(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if name == "model_train":
            source = phase1_manifest["partitions"]["training"]
            dates = (source["date_min"], source["date_max"])
        elif name == "model_validation":
            source = phase1_manifest["partitions"]["validation"]
            dates = (source["date_min"], source["date_max"])
        else:
            values = sorted(str(cast(Mapping[str, Any], row["fixture"])["date"]) for row in rows)
            dates = (values[0], values[-1])
        ids = [str(row["match_hash"]) for row in rows]
        return {
            "name": name,
            "count": len(rows),
            "date_min": dates[0],
            "date_max": dates[1],
            "fixture_ids": ids,
            "fixture_ids_sha256": row_ids_hash(ids),
            "canonical_fixture_ids_sha256": row_ids_hash(fixture_sets[name]),
        }

    temporal_checks = []
    for row in (*calibration_fit, *final_test):
        temporal = cast(Mapping[str, Any], row["temporal_evidence"])
        fixture_date = str(cast(Mapping[str, Any], row["fixture"])["date"])
        passed = (
            bool(temporal["strict_cutoff_passed"])
            and bool(temporal["same_day_sources_excluded"])
            and str(temporal["max_source_date"]) < fixture_date
        )
        if not passed:
            raise Phase3TemperatureIntegrityError(f"Point-in-time evidence failed for {row['match_hash']}")
        temporal_checks.append(passed)

    exclusions_path = PHASE1_RUN / "exclusions.jsonl"
    exclusion_count = sum(1 for line in exclusions_path.read_text(encoding="utf-8").splitlines() if line.strip())
    manifest = {
        "schema_version": 1,
        "league": "PL",
        "market": "1X2",
        "class_order": list(CLASS_ORDER),
        "boundaries": {
            "model_train": "Phase 1 isolated training; actual coverage before 2023/24",
            "model_validation": "Phase 1 2023/24 model validation",
            "calibration_fit": "2024-08-01 <= date < 2025-08-01",
            "final_test": "date >= 2025-08-01 through frozen dataset end",
        },
        "partitions": {
            name: summary(name, rows) for name, rows in partition_records.items()
        },
        "pairwise_disjointness": overlap_evidence,
        "identity_policy": "match_hash plus canonical date/league/home/away identity",
        "temporal_policy": {
            "prediction_time": "date_start_utc",
            "source_event_rule": "max_source_event_date < fixture_date",
            "same_day_source_events": "excluded",
            "rows_checked": len(temporal_checks),
            "all_passed": all(temporal_checks),
        },
        "source": {
            "phase1_run": PHASE1_RUN.relative_to(ROOT).as_posix(),
            "phase1_predictions_sha256": source_hash,
            "phase1_partition_manifest_sha256": _phase1_artifact_hash("partition_manifest.json"),
            "phase1_artifact_aggregate_sha256": read_json(PHASE1_INDEX)["aggregate_sha256"],
            "feature_matrix_sha256": sha256_file(FEATURE_MATRIX),
        },
        "exclusions": {
            "phase1_exclusions_path": exclusions_path.relative_to(ROOT).as_posix(),
            "phase1_exclusions_sha256": _phase1_artifact_hash("exclusions.jsonl"),
            "phase1_exclusion_count": exclusion_count,
            "phase3_additional_exclusion_count": 0,
        },
        "historical_limitation": (
            "The final-test period is protocol-isolated from calibration fitting but was already included "
            "in Phase 1 aggregate reporting; it is not globally pristine research data."
        ),
    }
    return Phase3PartitionContract(model_train, model_validation, calibration_fit, final_test, manifest)


def _contains_bookmaker_input(value: Any, path: str = "root") -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in BOOKMAKER_TOKENS):
                return f"{path}.{key}"
            found = _contains_bookmaker_input(item, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _contains_bookmaker_input(item, f"{path}[{index}]")
            if found:
                return found
    return None


def calibration_fit_rows(contract: Phase3PartitionContract) -> list[dict[str, Any]]:
    """Expose the deliberately tiny fit interface: ID, probabilities, outcome."""
    return [
        {
            "fixture_id": str(record["match_hash"]),
            "raw_probabilities": dict(cast(Mapping[str, float], record["raw_probabilities"])),
            "actual_result": str(record["actual_result"]),
        }
        for record in contract.calibration_fit
    ]


def validate_fit_rows(
    rows: Sequence[Mapping[str, Any]],
    partition_manifest: Mapping[str, Any],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    required = {"fixture_id", "raw_probabilities", "actual_result"}
    expected_ids = set(partition_manifest["partitions"]["calibration_fit"]["fixture_ids"])
    final_ids = set(partition_manifest["partitions"]["final_test"]["fixture_ids"])
    train_ids = set(partition_manifest["partitions"]["model_train"]["fixture_ids"])
    validation_ids = set(partition_manifest["partitions"]["model_validation"]["fixture_ids"])
    if expected_ids & final_ids or expected_ids & train_ids or expected_ids & validation_ids:
        raise Phase3TemperatureIntegrityError("Partition manifest is not disjoint")

    fixture_ids: list[str] = []
    probabilities: list[list[float]] = []
    labels: list[int] = []
    for index, row in enumerate(rows):
        if set(row) != required:
            raise Phase3TemperatureIntegrityError(
                f"Fit row {index} fields must be exactly {sorted(required)}; got {sorted(row)}"
            )
        forbidden = _contains_bookmaker_input(row)
        if forbidden:
            raise Phase3TemperatureIntegrityError(f"Bookmaker input forbidden at {forbidden}")
        fixture_id = str(row["fixture_id"])
        if fixture_id in final_ids:
            raise Phase3TemperatureIntegrityError(f"Final-test fixture entered calibrator fit: {fixture_id}")
        if fixture_id not in expected_ids:
            raise Phase3TemperatureIntegrityError(f"Non-calibration fixture entered calibrator fit: {fixture_id}")
        raw = cast(Mapping[str, Any], row["raw_probabilities"])
        if tuple(raw.keys()) != CLASS_ORDER:
            raise Phase3TemperatureIntegrityError(
                f"Class order must be {CLASS_ORDER}; got {tuple(raw.keys())}"
            )
        label = str(row["actual_result"])
        if label not in LABEL_TO_INDEX:
            raise Phase3TemperatureIntegrityError(f"Unknown 1X2 label: {label}")
        fixture_ids.append(fixture_id)
        probabilities.append([float(raw[name]) for name in CLASS_ORDER])
        labels.append(LABEL_TO_INDEX[label])

    if len(fixture_ids) != len(set(fixture_ids)):
        raise Phase3TemperatureIntegrityError("Duplicate fixture ID in calibration fit")
    if set(fixture_ids) != expected_ids:
        missing = sorted(expected_ids - set(fixture_ids))
        extra = sorted(set(fixture_ids) - expected_ids)
        raise Phase3TemperatureIntegrityError(f"Calibration fit must use its exact frozen partition; missing={missing[:3]} extra={extra[:3]}")
    matrix = np.asarray(probabilities, dtype=float)
    validate_probability_matrix(matrix)
    return fixture_ids, matrix, np.asarray(labels, dtype=int)


def validate_probability_matrix(probabilities: np.ndarray, *, tolerance: float = 1e-12) -> None:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(CLASS_ORDER):
        raise Phase3TemperatureIntegrityError(f"Expected N x {len(CLASS_ORDER)} probability matrix")
    if not np.all(np.isfinite(matrix)):
        raise Phase3TemperatureIntegrityError("Probability matrix contains NaN/Inf")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise Phase3TemperatureIntegrityError("Probability outside [0, 1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=tolerance, rtol=0.0):
        raise Phase3TemperatureIntegrityError("Probability rows are not normalized")


@dataclass(frozen=True)
class MulticlassTemperatureCalibrator:
    temperature: float
    class_order: tuple[str, str, str] = CLASS_ORDER

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise Phase3TemperatureIntegrityError("Temperature must be finite and positive")
        if tuple(self.class_order) != CLASS_ORDER:
            raise Phase3TemperatureIntegrityError(f"Serialized class order must be {CLASS_ORDER}")

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        matrix = np.asarray(probabilities, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        validate_probability_matrix(matrix, tolerance=1e-10)
        log_probabilities = np.log(np.clip(matrix, np.finfo(float).tiny, 1.0))
        scaled = log_probabilities / self.temperature
        calibrated = np.exp(scaled - logsumexp(scaled, axis=1, keepdims=True))
        validate_probability_matrix(calibrated, tolerance=1e-12)
        return calibrated


def fit_temperature_calibrator(
    rows: Sequence[Mapping[str, Any]],
    partition_manifest: Mapping[str, Any],
) -> tuple[MulticlassTemperatureCalibrator, dict[str, Any]]:
    fixture_ids, probabilities, labels = validate_fit_rows(rows, partition_manifest)
    log_probabilities = np.log(np.clip(probabilities, np.finfo(float).tiny, 1.0))

    def objective(vector: np.ndarray) -> float:
        log_temperature = float(vector[0])
        inverse_temperature = math.exp(-log_temperature) if -745.0 < log_temperature < 745.0 else (
            0.0 if log_temperature >= 745.0 else math.inf
        )
        if not math.isfinite(inverse_temperature):
            return float("inf")
        logits = log_probabilities * inverse_temperature
        losses = logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]
        result = float(np.mean(losses))
        return result if math.isfinite(result) else float("inf")

    result = minimize(
        objective,
        x0=np.asarray([0.0], dtype=float),
        method="BFGS",
        options={"gtol": 1e-12, "maxiter": 1000},
    )
    # scipy can report precision loss after converging numerically.  Accept the
    # finite stationary result only when a symmetric local check confirms it.
    fitted_log_temperature = float(result.x[0])
    epsilon = 1e-7
    center = objective(np.asarray([fitted_log_temperature]))
    left = objective(np.asarray([fitted_log_temperature - epsilon]))
    right = objective(np.asarray([fitted_log_temperature + epsilon]))
    if not math.isfinite(center) or center > min(left, right) + 1e-10:
        raise Phase3TemperatureIntegrityError(f"Temperature optimization failed: {result.message}")
    temperature = math.exp(fitted_log_temperature)
    calibrator = MulticlassTemperatureCalibrator(temperature=temperature)
    diagnostics = {
        "objective": "mean multiclass negative log likelihood on calibration_fit only",
        "optimizer": {
            "method": "BFGS over unconstrained log(T)",
            "bounds": None,
            "initial_log_temperature": 0.0,
            "reported_success": bool(result.success),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "fitted_objective": center,
            "local_minimum_check": {"epsilon": epsilon, "left": left, "center": center, "right": right},
        },
        "fit_fixture_count": len(fixture_ids),
        "fit_fixture_ids_sha256": row_ids_hash(fixture_ids),
        "label_counts": {
            label: int(np.sum(labels == index)) for index, label in enumerate(LABEL_ORDER)
        },
    }
    return calibrator, diagnostics


def _reliability(values: np.ndarray, outcomes: np.ndarray, *, bins: int = RELIABILITY_BINS) -> tuple[float, list[dict[str, Any]]]:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower = float(boundaries[index])
        upper = float(boundaries[index + 1])
        mask = (values >= lower) & ((values < upper) if index < bins - 1 else (values <= upper))
        count = int(mask.sum())
        predicted = float(values[mask].mean()) if count else None
        observed = float(outcomes[mask].mean()) if count else None
        gap = float(abs(predicted - observed)) if count else None
        if count:
            ece += (count / len(values)) * cast(float, gap)
        rows.append({
            "bin": index + 1,
            "lower_inclusive": lower,
            "upper_exclusive": upper if index < bins - 1 else None,
            "upper_inclusive": upper if index == bins - 1 else None,
            "count": count,
            "predicted_probability": predicted,
            "observed_frequency": observed,
            "absolute_gap": gap,
        })
    return float(ece), rows


def multiclass_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(probabilities, dtype=float)
    validate_probability_matrix(matrix, tolerance=1e-10)
    y = np.asarray(labels, dtype=int)
    if len(y) != len(matrix) or not np.all((0 <= y) & (y < len(CLASS_ORDER))):
        raise Phase3TemperatureIntegrityError("Invalid multiclass labels")
    clipped = np.clip(matrix, np.finfo(float).tiny, 1.0)
    predictions = np.argmax(matrix, axis=1)
    one_hot = np.eye(len(CLASS_ORDER), dtype=float)[y]
    confidence = np.max(matrix, axis=1)
    correct = (predictions == y).astype(float)
    top_ece, top_reliability = _reliability(confidence, correct)
    classwise: dict[str, Any] = {}
    for index, name in enumerate(CLASS_ORDER):
        ece, reliability = _reliability(matrix[:, index], (y == index).astype(float))
        classwise[name] = {"ece": ece, "reliability": reliability}
    return {
        "count": len(y),
        "class_order": list(CLASS_ORDER),
        "accuracy": float(np.mean(correct)),
        "log_loss": float(-np.mean(np.log(clipped[np.arange(len(y)), y]))),
        "multiclass_brier": float(np.mean(np.sum((matrix - one_hot) ** 2, axis=1))),
        "top_label_ece": top_ece,
        "top_label_reliability": top_reliability,
        "classwise": classwise,
        "average_entropy_nats": float(np.mean(-np.sum(clipped * np.log(clipped), axis=1))),
        "mean_max_probability": float(np.mean(confidence)),
        "binning": {
            "method": "10 equal-width bins on [0,1]",
            "bins": RELIABILITY_BINS,
            "last_bin_right_inclusive": True,
        },
    }


def paired_bootstrap(
    labels: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=int)
    raw_matrix = np.asarray(raw, dtype=float)
    calibrated_matrix = np.asarray(calibrated, dtype=float)
    if raw_matrix.shape != calibrated_matrix.shape or len(y) != len(raw_matrix):
        raise Phase3TemperatureIntegrityError("Paired bootstrap inputs do not align")
    one_hot = np.eye(len(CLASS_ORDER), dtype=float)[y]
    raw_ll = -np.log(np.clip(raw_matrix[np.arange(len(y)), y], np.finfo(float).tiny, 1.0))
    calibrated_ll = -np.log(np.clip(calibrated_matrix[np.arange(len(y)), y], np.finfo(float).tiny, 1.0))
    raw_brier = np.sum((raw_matrix - one_hot) ** 2, axis=1)
    calibrated_brier = np.sum((calibrated_matrix - one_hot) ** 2, axis=1)
    rng = np.random.default_rng(seed)
    log_loss_deltas = np.empty(samples, dtype=float)
    brier_deltas = np.empty(samples, dtype=float)
    ece_deltas = np.empty(samples, dtype=float)
    for index in range(samples):
        sample = rng.integers(0, len(y), size=len(y))
        log_loss_deltas[index] = float(np.mean(calibrated_ll[sample] - raw_ll[sample]))
        brier_deltas[index] = float(np.mean(calibrated_brier[sample] - raw_brier[sample]))
        raw_metrics = multiclass_metrics(y[sample], raw_matrix[sample])
        calibrated_metrics = multiclass_metrics(y[sample], calibrated_matrix[sample])
        ece_deltas[index] = float(calibrated_metrics["top_label_ece"] - raw_metrics["top_label_ece"])
    alpha = (1.0 - confidence_level) / 2.0

    def report(point: float, values: np.ndarray) -> dict[str, Any]:
        return {
            "point_delta_calibrated_minus_raw": point,
            "interval_lower": float(np.quantile(values, alpha)),
            "interval_upper": float(np.quantile(values, 1.0 - alpha)),
            "probability_delta_below_zero": float(np.mean(values < 0.0)),
        }

    raw_metrics = multiclass_metrics(y, raw_matrix)
    calibrated_metrics = multiclass_metrics(y, calibrated_matrix)
    return {
        "procedure": "paired fixture-level bootstrap with replacement; same sampled indices for raw and calibrated",
        "seed": seed,
        "samples": samples,
        "confidence_level": confidence_level,
        "delta_semantics": "calibrated - raw; negative loss delta indicates improvement",
        "log_loss": report(
            float(calibrated_metrics["log_loss"] - raw_metrics["log_loss"]), log_loss_deltas
        ),
        "multiclass_brier": report(
            float(calibrated_metrics["multiclass_brier"] - raw_metrics["multiclass_brier"]), brier_deltas
        ),
        "top_label_ece": report(
            float(calibrated_metrics["top_label_ece"] - raw_metrics["top_label_ece"]), ece_deltas
        ),
    }


def calibration_decision(
    raw_metrics: Mapping[str, Any],
    calibrated_metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    log_loss = cast(Mapping[str, float], comparison["log_loss"])
    brier = cast(Mapping[str, float], comparison["multiclass_brier"])
    top_ece_increase = float(calibrated_metrics["top_label_ece"] - raw_metrics["top_label_ece"])
    class_increases = {
        name: float(
            cast(Mapping[str, Any], calibrated_metrics["classwise"])[name]["ece"]
            - cast(Mapping[str, Any], raw_metrics["classwise"])[name]["ece"]
        )
        for name in CLASS_ORDER
    }
    pathology = (
        top_ece_increase > DECISION_RULE["reliability_pathology_thresholds"]["top_label_ece_increase"]
        or any(
            increase > DECISION_RULE["reliability_pathology_thresholds"]["any_classwise_ece_increase"]
            for increase in class_increases.values()
        )
    )
    supported = (
        log_loss["point_delta_calibrated_minus_raw"] < 0.0
        and log_loss["interval_upper"] <= 0.0
        and brier["point_delta_calibrated_minus_raw"] <= 0.0
        and brier["interval_upper"] <= 0.005
        and not pathology
    )
    rejected = (
        log_loss["point_delta_calibrated_minus_raw"] >= 0.01
        or brier["point_delta_calibrated_minus_raw"] >= 0.005
        or log_loss["interval_lower"] > 0.0
        or brier["interval_lower"] > 0.0
        or pathology
    )
    decision = "SUPPORTED FOR FURTHER VALIDATION" if supported else ("REJECTED" if rejected else "INCONCLUSIVE")
    return {
        "decision": decision,
        "rule": DECISION_RULE,
        "reliability_pathology": pathology,
        "top_label_ece_increase": top_ece_increase,
        "classwise_ece_increase": class_increases,
        "production_promotion": False,
        "interpretation": (
            "Research recommendation only. Final-test observations were not used to fit, select, or tune the calibrator."
        ),
    }


def probabilities_and_labels(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray([
        [float(cast(Mapping[str, Any], row["raw_probabilities"])[name]) for name in CLASS_ORDER]
        for row in records
    ], dtype=float)
    labels = np.asarray([LABEL_TO_INDEX[str(row["actual_result"])] for row in records], dtype=int)
    validate_probability_matrix(probabilities, tolerance=1e-10)
    return probabilities, labels


def source_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    diff = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, check=True, capture_output=True).stdout
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def runtime_fingerprint() -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
        "dependency_fingerprint": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
    }


def build_calibrator_payload(
    calibrator: MulticlassTemperatureCalibrator,
    fit_diagnostics: Mapping[str, Any],
    contract: Phase3PartitionContract,
    *,
    calibrator_id: str,
    source_code_path: Path,
) -> dict[str, Any]:
    calibration = contract.manifest["partitions"]["calibration_fit"]
    final_test = contract.manifest["partitions"]["final_test"]
    bundle = read_json(PHASE1_MODEL_BUNDLE)
    runtime = runtime_fingerprint()
    payload = {
        "schema_version": 1,
        "calibrator_id": calibrator_id,
        "method": "multiclass_temperature_scaling",
        "market": "1X2",
        "league": "PL",
        "class_order": list(CLASS_ORDER),
        "temperature": calibrator.temperature,
        "log_temperature": math.log(calibrator.temperature),
        "created_at": utc_now(),
        "raw_prediction_input_artifact": {
            "path": PHASE1_PREDICTIONS.relative_to(ROOT).as_posix(),
            "sha256": contract.manifest["source"]["phase1_predictions_sha256"],
        },
        "calibration_fit_fixture_ids": calibration["fixture_ids"],
        "calibration_fit_fixture_ids_sha256": calibration["fixture_ids_sha256"],
        "calibration_fit_date_range": [calibration["date_min"], calibration["date_max"]],
        "calibration_fit_count": calibration["count"],
        "final_test_fixture_ids_sha256": final_test["fixture_ids_sha256"],
        "final_test_exclusion_assertion": True,
        "objective": fit_diagnostics["objective"],
        "optimizer": fit_diagnostics["optimizer"],
        "random_seed": None,
        "source_code": {
            "path": source_code_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(source_code_path),
            "revision": source_state()["commit"],
        },
        "source_state": source_state(),
        "runtime": {
            "python": runtime["python"],
            "platform": runtime["platform"],
            "implementation": runtime["implementation"],
        },
        "dependency_fingerprint": runtime["dependency_fingerprint"],
        "upstream_model_bundle": {
            "path": PHASE1_MODEL_BUNDLE.relative_to(ROOT).as_posix(),
            "manifest_sha256": _phase1_artifact_hash("model_bundle_manifest.json"),
            "artifact_aggregate_sha256": bundle["artifacts_aggregate_sha256"],
            "evaluation_only": bundle["evaluation_only"],
        },
        "feature_schema": {
            "class": "Phase 1 goal-model feature order",
            "sha256": bundle["feature_order_sha256"],
            "bookmaker_features": False,
        },
        "simulator": {
            "implementation": "src.simulation.match_simulator.MatchSimulator.simulate",
            "n_simulations": 50_000,
            "raw_predictions_reused_from_phase1": True,
            "fixture_specific_seed": True,
        },
        "no_bookmaker_input_assertion": True,
        "evaluation_only": True,
        "production_eligible": False,
        "trust_status": "PHASE3_TRUSTED_EVALUATION_ONLY",
        "fit_diagnostics": dict(fit_diagnostics),
    }
    validate_trusted_calibrator_payload(payload, contract.manifest)
    return payload


def validate_trusted_calibrator_payload(
    payload: Mapping[str, Any],
    partition_manifest: Mapping[str, Any],
) -> None:
    missing = sorted(REQUIRED_PROVENANCE_FIELDS - set(payload))
    if missing:
        raise Phase3TemperatureIntegrityError(f"Trusted calibrator provenance missing fields: {missing}")
    if payload["trust_status"] != "PHASE3_TRUSTED_EVALUATION_ONLY":
        raise Phase3TemperatureIntegrityError("Legacy calibrator cannot be marked trusted")
    if payload["method"] != "multiclass_temperature_scaling" or payload["market"] != "1X2" or payload["league"] != "PL":
        raise Phase3TemperatureIntegrityError("Calibrator scope is not authorized")
    if tuple(payload["class_order"]) != CLASS_ORDER:
        raise Phase3TemperatureIntegrityError("Calibrator class order changed")
    if not payload["evaluation_only"] or payload["production_eligible"]:
        raise Phase3TemperatureIntegrityError("Phase 3 calibrator must remain evaluation-only")
    if not payload["no_bookmaker_input_assertion"] or _contains_bookmaker_input({"fit": payload.get("fit_inputs", {})}):
        raise Phase3TemperatureIntegrityError("Bookmaker independence is not proven")
    calibration = partition_manifest["partitions"]["calibration_fit"]
    final_test = partition_manifest["partitions"]["final_test"]
    ids = [str(value) for value in payload["calibration_fit_fixture_ids"]]
    if row_ids_hash(ids) != payload["calibration_fit_fixture_ids_sha256"]:
        raise Phase3TemperatureIntegrityError("Calibrator fit identity hash does not match its identities")
    if set(ids) != set(calibration["fixture_ids"]):
        raise Phase3TemperatureIntegrityError("Calibrator was not fit on the exact calibration partition")
    if set(ids) & set(final_test["fixture_ids"]):
        raise Phase3TemperatureIntegrityError("Calibrator fit identities overlap final test")
    if payload["final_test_fixture_ids_sha256"] != final_test["fixture_ids_sha256"]:
        raise Phase3TemperatureIntegrityError("Calibrator final-test exclusion target changed")
    if not payload["final_test_exclusion_assertion"]:
        raise Phase3TemperatureIntegrityError("Calibrator does not assert final-test exclusion")
    MulticlassTemperatureCalibrator(
        temperature=float(payload["temperature"]),
        class_order=tuple(payload["class_order"]),
    )


def load_calibrator_artifact(
    artifact_path: Path,
    manifest_path: Path,
    partition_manifest: Mapping[str, Any],
) -> tuple[MulticlassTemperatureCalibrator, dict[str, Any]]:
    manifest = read_json(manifest_path)
    if manifest.get("artifact_sha256") != sha256_file(artifact_path):
        raise Phase3TemperatureIntegrityError("Calibrator artifact hash mismatch")
    payload = read_json(artifact_path)
    validate_trusted_calibrator_payload(payload, partition_manifest)
    if manifest.get("calibrator_id") != payload["calibrator_id"]:
        raise Phase3TemperatureIntegrityError("Calibrator manifest ID mismatch")
    calibrator = MulticlassTemperatureCalibrator(
        temperature=float(payload["temperature"]),
        class_order=tuple(payload["class_order"]),
    )
    return calibrator, payload


def legacy_calibrator_inventory() -> dict[str, Any]:
    """Inventory without loading or mutating unsafe pickle/joblib artifacts."""
    manifest = read_json(ROOT / "src" / "ml" / "models" / "manifest.json")
    pointers: list[dict[str, Any]] = []
    for key, value in manifest.items():
        if not isinstance(value, Mapping):
            continue
        for field, filename in value.items():
            if "calibrator" not in str(field).lower() or "filename" not in str(field).lower() or not filename:
                continue
            candidate = ROOT / "src" / "ml" / "models" / str(filename)
            pointers.append({
                "manifest_key": str(key),
                "field": str(field),
                "path": candidate.relative_to(ROOT).as_posix(),
                "exists": candidate.is_file(),
                "sha256": sha256_file(candidate) if candidate.is_file() else None,
                "status": "LEGACY_CALIBRATION_VALIDITY_UNPROVEN",
                "reason": "Manifest metadata does not prove calibration-fit/final-test identity disjointness.",
            })
    standalone = []
    for path in sorted((ROOT / "src" / "ml" / "models").rglob("*calibrat*")):
        if not path.is_file():
            continue
        standalone.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
            "status": "LEGACY_CALIBRATION_VALIDITY_UNPROVEN",
        })
    workflows = [
        {
            "path": "src/models/train_probability_models.py",
            "symbols": ["_fit_and_save_goals_calibrators", "_fit_and_save_corners_calibrators"],
            "fit_population": "nominal X_test/y_test",
            "fit_test_relationship": "same nominal test partition can fit calibrators and report evaluation",
            "status": "LEGACY_CALIBRATION_VALIDITY_UNPROVEN",
        },
        {
            "path": "src/ml/training/orchestrator.py",
            "symbols": ["_fit_goals_calibrators"],
            "fit_population": "time_series_split_train_val_test(...).test_df",
            "fit_test_relationship": "test_df directly fits saved calibrators",
            "status": "LEGACY_CALIBRATION_VALIDITY_UNPROVEN",
        },
        {
            "path": "src/ml/trainer.py",
            "symbols": ["_fit_posthoc_calibrator"],
            "fit_population": "validation, with earliest-training fallback",
            "fit_test_relationship": "safer split label, but artifact lacks immutable fixture-level final-test exclusion provenance",
            "status": "LEGACY_CALIBRATION_VALIDITY_UNPROVEN",
        },
        {
            "path": "scripts/train_calibrators.py",
            "symbols": ["train_calibrators"],
            "fit_population": "backtest rows loaded without fixture-level partition provenance",
            "fit_test_relationship": "cannot prove calibration-fit/final-test disjointness from saved artifact",
            "status": "LEGACY_CALIBRATION_VALIDITY_UNPROVEN",
        },
    ]
    prior_phase3 = ROOT / "baselines" / "phase3" / "phase3-pl-calibration-79110f9a-20260811-v1"
    return {
        "schema_version": 1,
        "policy": "Unproven artifacts remain untouched and are ineligible as trusted Phase 3 evidence.",
        "classification": "LEGACY_CALIBRATION_VALIDITY_UNPROVEN",
        "manifest_pointer_count": len(pointers),
        "manifest_pointers": pointers,
        "standalone_artifact_count": len(standalone),
        "standalone_artifacts": standalone,
        "workflows": workflows,
        "prior_phase3_bundle": {
            "path": prior_phase3.relative_to(ROOT).as_posix(),
            "exists": prior_phase3.is_dir(),
            "classification": "LEGACY_DIAGNOSTIC_OUTSIDE_AUTHORIZED_PL_1X2_TEMPERATURE_SCOPE",
            "mutated": False,
        },
    }


def protected_snapshot() -> dict[str, Any]:
    """Fingerprint all earlier phases and production-protected identities."""
    from scripts.phase0_baseline import data_inventory, model_inventory
    from src.evaluation.phase3_calibration import aggregate_directory

    models = model_inventory()
    data = data_inventory()
    phase1_index = read_json(PHASE1_INDEX)
    phase2_run = ROOT / "baselines" / "phase2" / "phase2-count-markets-79110f9a-20260811-v1"
    phase2_closure = ROOT / "baselines" / "phase2" / "closures" / "phase2-count-markets-79110f9a-20260811-closure-v1"
    prior_phase3 = ROOT / "baselines" / "phase3" / "phase3-pl-calibration-79110f9a-20260811-v1"
    return {
        "captured_at": utc_now(),
        "phase0_evidence_aggregate_sha256": aggregate_directory(ROOT / "baselines" / "phase0"),
        "phase1_evidence_aggregate_sha256": aggregate_directory(PHASE1_RUN),
        "phase2_evidence_aggregate_sha256": aggregate_directory(phase2_run),
        "phase2_closure_aggregate_sha256": aggregate_directory(phase2_closure),
        "prior_phase3_legacy_aggregate_sha256": aggregate_directory(prior_phase3),
        "phase0_golden_sha256": sha256_file(ROOT / "baselines" / "phase0" / "golden_predictions.json"),
        "phase1_artifact_aggregate_sha256": phase1_index["aggregate_sha256"],
        "phase2_controlled_golden_sha256": sha256_file(phase2_run / "controlled_goldens.json"),
        "active_manifest_sha256": models["manifest"]["sha256"],
        "active_model_aggregate_sha256": models["active_artifacts_aggregate_sha256"],
        "protected_data_aggregate_sha256": data["protected_aggregate_sha256"],
        "feature_order_aggregate_sha256": models["active_feature_order_aggregate_sha256"],
    }

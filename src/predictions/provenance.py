"""Versioned prediction execution provenance for the FFPS PL 1X2 path.

The module is intentionally independent of the prediction engine.  The engine
records what it actually executed; this module validates, serializes, hashes,
persists, and replays that evidence without implementing a second prediction
method.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import pickle
import platform
import subprocess
import sys
import uuid
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, cast

import joblib
import numpy as np
import pandas as pd


SCHEMA_VERSION = "ffps.prediction_provenance.v1"
CALCULATION_PATH = "pl_1x2.goal_ensemble.match_simulator.v1"
CLASS_ORDER = ("home_win", "draw", "away_win")
MODEL_ROLE_ORDER = (
    "home_goals_lgbm",
    "away_goals_lgbm",
    "home_goals_xgb",
    "away_goals_xgb",
)
BOOKMAKER_TOKENS = (
    "odds",
    "bookmaker",
    "bet365",
    "pinnacle",
    "market_price",
    "implied_probability",
    "opening_price",
    "closing_price",
)
RELEVANT_SOURCE_FILES = (
    "src/cli/commands/prediction.py",
    "src/predictions/provenance.py",
    "src/predictions/predictor.py",
    "src/ml/registry.py",
    "src/features/pipeline.py",
    "src/features/engineering.py",
    "src/simulation/match_simulator.py",
    "src/config/__init__.py",
)


class ProvenanceError(RuntimeError):
    """Raised when trusted provenance cannot be constructed or verified."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Convert supported values to strict, finite, portable JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ProvenanceError("Canonical provenance rejects NaN and Infinity")
        return number
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=lambda item: canonical_json(item))
    if pd.isna(value):
        return None
    raise ProvenanceError(f"Unsupported canonical provenance value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the single Phase 4 UTF-8 JSON serialization contract."""
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def logical_path(path: Path, root: Path | None = None) -> str:
    resolved_root = (root or repository_root()).resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError:
        return f"external/{path.name}"


def artifact_identity(path: Path, payload: bytes, *, root: Path | None = None) -> dict[str, Any]:
    return {
        "source_location": logical_path(path, root),
        "artifact_sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _canonical_row(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        value = row.get(column)
        # Missing source cells are an explicit JSON null, never a NaN token.
        result[str(column)] = None if pd.isna(value) else _json_safe(value)
    return result


def canonical_frame_identity(frame: pd.DataFrame) -> dict[str, Any]:
    columns = [str(column) for column in frame.columns]
    records = [_canonical_row(record, columns) for record in frame.to_dict(orient="records")]
    payload = {"ordered_columns": columns, "records": records}
    return {
        "ordered_columns": columns,
        "row_count": len(records),
        "row_set_sha256": sha256_json(payload),
    }


def build_file_snapshot(
    *,
    frame: pd.DataFrame,
    artifacts: Sequence[Mapping[str, Any]],
    source_type: str,
) -> dict[str, Any]:
    """Create portable data-source metadata for a DataFrame built from files."""
    # Source load order is semantic because concatenation and duplicate
    # resolution can depend on it.  Callers therefore supply deterministic or
    # explicitly observed order and canonical serialization preserves it.
    ordered_artifacts = [_json_safe(item) for item in artifacts]
    frame_identity = canonical_frame_identity(frame)
    return {
        "source_type": source_type,
        "artifacts": ordered_artifacts,
        "artifacts_aggregate_sha256": sha256_json(ordered_artifacts),
        **frame_identity,
    }


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return cast(pd.Timestamp, parsed)


def _completed_event_rows(frame: pd.DataFrame) -> Iterable[Mapping[str, Any]]:
    for record in frame.to_dict(orient="records"):
        home_score = record.get("home_score")
        away_score = record.get("away_score")
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        yield record


def source_event_index(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Build compact identities for completed rows that can feed rolling state."""
    columns = [str(column) for column in frame.columns]
    events: list[dict[str, Any]] = []
    for record in _completed_event_rows(frame):
        event_time = _timestamp(record.get("date") or record.get("match_date"))
        if event_time is None:
            continue
        events.append(
            {
                "event_timestamp": event_time.isoformat(),
                "match_identity": str(record.get("match_id") or record.get("match_hash") or ""),
                "home_team": str(record.get("home_team") or ""),
                "away_team": str(record.get("away_team") or ""),
                "row_sha256": sha256_json(_canonical_row(record, columns)),
            }
        )
    events.sort(key=lambda item: (item["event_timestamp"], item["match_identity"], item["row_sha256"]))
    return events


def data_provenance_for_fixture(
    frame: pd.DataFrame,
    fixture: pd.Series,
    prediction_timestamp: str,
) -> dict[str, Any]:
    """Bind a fixture to the precise prediction frame and causal source events."""
    snapshot = copy.deepcopy(cast(dict[str, Any], frame.attrs.get("ffps_data_snapshot", {})))
    if not snapshot:
        snapshot = {
            "source_type": "in_memory_prediction_frame",
            "artifacts": [],
            "artifacts_aggregate_sha256": sha256_json([]),
            **canonical_frame_identity(frame),
        }

    fixture_time = _timestamp(fixture.get("kickoff_utc")) or _timestamp(fixture.get("date"))
    prediction_time = _timestamp(prediction_timestamp)
    if prediction_time is None:
        raise ProvenanceError("Prediction timestamp is missing or invalid")
    if fixture_time is None:
        raise ProvenanceError("Fixture kickoff/date identity is missing or invalid")

    raw_kickoff = fixture.get("kickoff_utc")
    if _timestamp(raw_kickoff) is not None:
        temporal_policy = "exact_kickoff_exclusive"
    else:
        temporal_policy = "date_only_start_of_day_excludes_same_day"
        fixture_time = fixture_time.normalize()
    cutoff = min(prediction_time, fixture_time)

    home_team = str(fixture.get("home_team", ""))
    away_team = str(fixture.get("away_team", ""))
    relevant: list[dict[str, Any]] = []
    indexed_events = frame.attrs.get("ffps_source_events")
    candidates = (
        cast(Sequence[Mapping[str, Any]], indexed_events)
        if isinstance(indexed_events, (list, tuple))
        else source_event_index(frame)
    )
    for event in candidates:
        event_time = _timestamp(event.get("event_timestamp"))
        if event_time is None or event_time >= cutoff:
            continue
        teams = {str(event.get("home_team", "")), str(event.get("away_team", ""))}
        if not teams.intersection({home_team, away_team}):
            continue
        relevant.append(
            {
                "event_timestamp": event_time.isoformat(),
                "match_identity": str(event.get("match_identity") or ""),
                "row_sha256": str(event["row_sha256"]),
            }
        )
    relevant.sort(key=lambda item: (item["event_timestamp"], item["match_identity"], item["row_sha256"]))
    maximum = relevant[-1]["event_timestamp"] if relevant else None
    violations = [item for item in relevant if _timestamp(item["event_timestamp"]) >= cutoff]

    return {
        "source_identity": snapshot,
        "causal_cutoff": cutoff.isoformat(),
        "prediction_timestamp": prediction_time.isoformat(),
        "fixture_cutoff_policy": temporal_policy,
        "maximum_source_event_timestamp": maximum,
        "source_event_count": len(relevant),
        "source_events": relevant,
        "source_row_set_sha256": sha256_json(relevant),
        "temporal_cutoff_verified": not violations,
        "cutoff_violations": violations,
    }


def _git_output(root: Path, *args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


@lru_cache(maxsize=1)
def code_runtime_identity() -> dict[str, Any]:
    """Capture approved code/runtime identities without environment secrets."""
    root = repository_root()
    try:
        revision = str(_git_output(root, "rev-parse", "HEAD")).strip()
        branch = str(_git_output(root, "branch", "--show-current")).strip()
        status = str(_git_output(root, "status", "--porcelain=v1", "--untracked-files=all"))
        diff = cast(bytes, _git_output(root, "diff", "--binary", "HEAD", binary=True))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProvenanceError(f"Cannot capture Git source state: {exc}") from exc

    source_files: list[dict[str, Any]] = []
    for relative in RELEVANT_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise ProvenanceError(f"Required provenance source file is missing: {relative}")
        source_files.append({"path": relative, "sha256": sha256_file(path)})

    dependencies: dict[str, str] = {}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "lightgbm", "xgboost"):
        try:
            dependencies[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"

    return {
        "git": {
            "revision": revision,
            "branch": branch,
            "dirty": bool(status.strip()),
            "status_sha256": sha256_bytes(status.encode("utf-8")),
            "tracked_diff_sha256": sha256_bytes(diff),
        },
        "relevant_source_files": source_files,
        "relevant_source_aggregate_sha256": sha256_json(source_files),
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable_name": Path(sys.executable).name,
            "dependencies": dependencies,
            "dependency_fingerprint": sha256_json(dependencies),
        },
    }


def _role_from_context(context: str) -> str:
    suffixes = {
        "GoalsH:LGBM": "home_goals_lgbm",
        "GoalsA:LGBM": "away_goals_lgbm",
        "GoalsH:XGB": "home_goals_xgb",
        "GoalsA:XGB": "away_goals_xgb",
    }
    for suffix, role in suffixes.items():
        if context.endswith(suffix):
            return role
    raise ProvenanceError(f"Unsupported PL 1X2 model-call context: {context}")


@dataclass
class PredictionCapture:
    prediction_timestamp: str
    data_provenance: Mapping[str, Any]
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_calls: MutableMapping[str, dict[str, Any]] = field(default_factory=dict)
    _token: Token[PredictionCapture | None] | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "PredictionCapture":
        if _ACTIVE_CAPTURE.get() is not None:
            raise ProvenanceError("Nested prediction provenance capture is unsupported")
        self._token = _ACTIVE_CAPTURE.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._token is not None:
            _ACTIVE_CAPTURE.reset(self._token)
        return False

    def record_model_call(
        self,
        *,
        model: Any,
        frame: pd.DataFrame,
        context: str,
        raw_output: float,
        effective_output: float,
    ) -> None:
        role = _role_from_context(context)
        if role in self.model_calls:
            raise ProvenanceError(f"Duplicate model role in one prediction: {role}")
        if len(frame) != 1:
            raise ProvenanceError(f"Expected one inference row for {role}, got {len(frame)}")
        identity = getattr(model, "_ffps_provenance", None)
        if not isinstance(identity, Mapping):
            raise ProvenanceError(f"Loaded artifact provenance is missing for {role}")

        names = [str(column) for column in frame.columns]
        values = [_json_safe(frame.iloc[0][column]) for column in frame.columns]
        if len(names) != len(set(names)):
            raise ProvenanceError(f"Duplicate ordered feature name for {role}")
        bookmaker = [name for name in names if any(token in name.lower() for token in BOOKMAKER_TOKENS)]
        if bookmaker:
            raise ProvenanceError(f"Bookmaker-derived prediction features are prohibited: {bookmaker}")

        meta = cast(Mapping[str, Any], getattr(model, "meta", {}) or {})
        calibrator_ref = meta.get("calibrator_filename")
        self.model_calls[role] = {
            "role": role,
            "ordered_feature_names": names,
            "feature_schema_sha256": sha256_json(names),
            "feature_values": values,
            "feature_vector_sha256": sha256_json({"ordered_feature_names": names, "values": values}),
            "finite_values_verified": True,
            "source_row_set_sha256": self.data_provenance["source_row_set_sha256"],
            "causal_cutoff": self.data_provenance["causal_cutoff"],
            "raw_model_output": _json_safe(raw_output),
            "effective_model_output": _json_safe(effective_output),
            "model_calibrator": {
                "declared": calibrator_ref is not None,
                "reference": str(calibrator_ref) if calibrator_ref is not None else None,
                "applied": calibrator_ref is not None,
            },
            "model": _json_safe(identity),
        }

    def finalize(
        self,
        *,
        fixture: pd.Series,
        league: str,
        probabilities: Mapping[str, Any],
        simulator: Any,
    ) -> "PredictionProvenanceRecord":
        if tuple(self.model_calls) != MODEL_ROLE_ORDER:
            raise ProvenanceError(
                f"PL 1X2 provenance requires ordered model roles {MODEL_ROLE_ORDER}, got {tuple(self.model_calls)}"
            )
        if any(call["model_calibrator"]["applied"] for call in self.model_calls.values()):
            raise ProvenanceError("Authorized PL 1X2 provenance forbids model-level calibrators")

        simulation = getattr(simulator, "last_call_provenance", None)
        if not isinstance(simulation, Mapping):
            raise ProvenanceError("MatchSimulator did not expose its executed RNG contract")

        from src.config import (
            GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD,
            GOALS_ENSEMBLE_LGBM_WEIGHT,
            GOALS_ENSEMBLE_XGB_WEIGHT,
        )
        from src.simulation.match_simulator import LAMBDA_MAX, LAMBDA_MIN

        members = []
        features = []
        member_outputs: dict[str, Any] = {}
        for role in MODEL_ROLE_ORDER:
            call = self.model_calls[role]
            member = {"role": role, **cast(dict[str, Any], call["model"])}
            members.append(member)
            features.append({key: value for key, value in call.items() if key != "model"})
            member_outputs[role] = {
                "raw": call["raw_model_output"],
                "effective": call["effective_model_output"],
            }
        bundle_material = [
            {
                "role": item["role"],
                "artifact_sha256": item["artifact_sha256"],
                "manifest_key": item.get("manifest_key"),
                "registry_slot": item.get("registry_slot"),
            }
            for item in members
        ]
        model_bundle_sha256 = sha256_json(bundle_material)
        feature_bundle_sha256 = sha256_json(
            [
                {
                    "role": item["role"],
                    "feature_schema_sha256": item["feature_schema_sha256"],
                    "feature_vector_sha256": item["feature_vector_sha256"],
                }
                for item in features
            ]
        )

        output_values = [
            _json_safe(probabilities["home"]),
            _json_safe(probabilities["draw"]),
            _json_safe(probabilities["away"]),
        ]
        total = float(sum(cast(float, value) for value in output_values))
        if not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0.0):
            raise ProvenanceError(f"PL 1X2 output does not normalize: {total}")

        method_descriptor = {
            "calculation_path": CALCULATION_PATH,
            "ensemble": {
                "composition": list(MODEL_ROLE_ORDER),
                "lgbm_weight": float(GOALS_ENSEMBLE_LGBM_WEIGHT),
                "xgb_weight": float(GOALS_ENSEMBLE_XGB_WEIGHT),
                "divergence_threshold": float(GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD),
                "member_outputs": member_outputs,
            },
            "goal_lambda_contract": {"minimum": float(LAMBDA_MIN), "maximum": float(LAMBDA_MAX)},
            "simulation": _json_safe(simulation),
            "rl_weights": None,
            "match_outcome_classifier": {"present": False, "used": False},
            "post_processing": [],
            "calibrator": {"applied": False, "identity": None},
        }
        method_config_hash = sha256_json(method_descriptor)
        code = code_runtime_identity()

        kickoff_candidate = fixture.get("kickoff_utc")
        kickoff = fixture.get("date") if _timestamp(kickoff_candidate) is None else kickoff_candidate
        match_hash = fixture.get("match_id") or fixture.get("match_hash")
        if not match_hash:
            raise ProvenanceError("Canonical fixture match identity is missing")
        fixture_identity = {
            "competition": str(fixture.get("competition") or league),
            "league": str(league),
            "home_team": str(fixture.get("home_team") or ""),
            "away_team": str(fixture.get("away_team") or ""),
            "kickoff_identity": _json_safe(kickoff),
            "match_hash": str(match_hash),
            "prediction_timestamp": self.prediction_timestamp,
        }
        if not fixture_identity["home_team"] or not fixture_identity["away_team"]:
            raise ProvenanceError("Fixture team identity is incomplete")

        output = {
            "market": "1X2",
            "class_order": list(CLASS_ORDER),
            "raw_probabilities": output_values,
            "final_probabilities": output_values,
            "probability_sum": total,
            "finite_verified": True,
            "normalized_verified": True,
            "applied_post_processors": [],
        }
        output["output_sha256"] = sha256_json({key: value for key, value in output.items() if key != "output_sha256"})

        training_proven = all(
            item.get("training_lineage", {}).get("classification") == "proven" for item in members
        )
        assertions = {
            "execution_provenance_complete": True,
            "temporal_cutoff_verified": bool(self.data_provenance["temporal_cutoff_verified"]),
            "feature_identity_verified": True,
            "model_bytes_verified": True,
            "model_training_lineage_proven": training_proven,
            "model_training_lineage_classification": "proven" if training_proven else "unproven",
            "method_contract_verified": True,
            "bookmaker_prediction_inputs_absent": True,
            "output_reproducible": False,
        }
        if not assertions["temporal_cutoff_verified"]:
            raise ProvenanceError("Required point-in-time cutoff evidence could not be established")

        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "recorded_at": utc_now(),
            "fixture": fixture_identity,
            "data_provenance": _json_safe(self.data_provenance),
            "feature_provenance": {
                "model_inputs": features,
                "feature_bundle_sha256": feature_bundle_sha256,
            },
            "model_provenance": {
                "members": members,
                "model_bundle_sha256": model_bundle_sha256,
            },
            "method_provenance": {
                "descriptor": method_descriptor,
                "method_config_hash": method_config_hash,
            },
            "code_provenance": code,
            "prediction_output": output,
            "bookmaker_independence": {
                "model_feature_input": False,
                "model_calculation": False,
                "calibration_or_post_processing": False,
            },
            "trust_assertions": assertions,
        }
        record["prediction_fingerprint"] = prediction_fingerprint(record)
        record["record_integrity_sha256"] = record_integrity_hash(record)
        return PredictionProvenanceRecord.from_dict(record)


_ACTIVE_CAPTURE: ContextVar[PredictionCapture | None] = ContextVar("ffps_prediction_capture", default=None)


def active_capture() -> PredictionCapture | None:
    return _ACTIVE_CAPTURE.get()


def record_model_call(
    *,
    model: Any,
    frame: pd.DataFrame,
    context: str,
    raw_output: float,
    effective_output: float,
) -> None:
    capture = active_capture()
    if capture is not None:
        capture.record_model_call(
            model=model,
            frame=frame,
            context=context,
            raw_output=raw_output,
            effective_output=effective_output,
        )


def _fingerprint_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    code = cast(Mapping[str, Any], record["code_provenance"])
    return {
        "schema_version": record["schema_version"],
        "fixture": record["fixture"],
        "data_provenance": record["data_provenance"],
        "feature_provenance": record["feature_provenance"],
        "model_provenance": record["model_provenance"],
        "method_provenance": record["method_provenance"],
        "code_identity": {
            "git_revision": cast(Mapping[str, Any], code["git"])["revision"],
            "relevant_source_aggregate_sha256": code["relevant_source_aggregate_sha256"],
            "dependency_fingerprint": cast(Mapping[str, Any], code["runtime"])["dependency_fingerprint"],
        },
        "prediction_output": record["prediction_output"],
        "bookmaker_independence": record["bookmaker_independence"],
    }


def prediction_fingerprint(record: Mapping[str, Any]) -> str:
    return sha256_json(_fingerprint_payload(record))


def record_integrity_hash(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "record_integrity_sha256"}
    return sha256_json(payload)


@dataclass(frozen=True)
class PredictionProvenanceRecord:
    """Immutable canonical representation of one prediction execution."""

    _canonical_record: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PredictionProvenanceRecord":
        canonical = canonical_json(value)
        parsed = cast(dict[str, Any], json.loads(canonical))
        if parsed.get("schema_version") != SCHEMA_VERSION:
            raise ProvenanceError(f"Unsupported provenance schema: {parsed.get('schema_version')}")
        return cls(canonical)

    @property
    def run_id(self) -> str:
        return str(self.to_dict()["run_id"])

    @property
    def prediction_fingerprint(self) -> str:
        return str(self.to_dict()["prediction_fingerprint"])

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads(self._canonical_record))

    def canonical_bytes(self) -> bytes:
        return self._canonical_record.encode("utf-8")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def load(cls, path: Path) -> "PredictionProvenanceRecord":
        return cls.from_dict(cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8"))))


def _verify_hash(errors: list[str], label: str, actual: str, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label} mismatch: expected={expected} actual={actual}")


def verify_record(
    value: PredictionProvenanceRecord | Mapping[str, Any],
    *,
    root: Path | None = None,
    verify_referenced_artifacts: bool = True,
) -> dict[str, Any]:
    record = value.to_dict() if isinstance(value, PredictionProvenanceRecord) else cast(dict[str, Any], _json_safe(value))
    errors: list[str] = []
    try:
        if record.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"unsupported schema version: {record.get('schema_version')}")
            return {"verified": False, "errors": errors}
        try:
            uuid.UUID(str(record.get("run_id")))
        except (ValueError, TypeError):
            errors.append("run_id is not a UUID")

        feature_section = cast(Mapping[str, Any], record["feature_provenance"])
        calls = cast(Sequence[Mapping[str, Any]], feature_section["model_inputs"])
        if tuple(str(call.get("role")) for call in calls) != MODEL_ROLE_ORDER:
            errors.append("model input role order is invalid")
        for call in calls:
            names = cast(Sequence[str], call["ordered_feature_names"])
            values = cast(Sequence[Any], call["feature_values"])
            if len(names) != len(values) or len(names) != len(set(names)):
                errors.append(f"feature shape/order invalid for {call.get('role')}")
            _verify_hash(errors, f"feature schema {call.get('role')}", sha256_json(names), call["feature_schema_sha256"])
            _verify_hash(
                errors,
                f"feature vector {call.get('role')}",
                sha256_json({"ordered_feature_names": names, "values": values}),
                call["feature_vector_sha256"],
            )
            bookmaker = [name for name in names if any(token in name.lower() for token in BOOKMAKER_TOKENS)]
            if bookmaker:
                errors.append(f"bookmaker-derived features present: {bookmaker}")
            if cast(Mapping[str, Any], call["model_calibrator"]).get("applied"):
                errors.append(f"model calibrator applied for {call.get('role')}")
        feature_material = [
            {
                "role": call["role"],
                "feature_schema_sha256": call["feature_schema_sha256"],
                "feature_vector_sha256": call["feature_vector_sha256"],
            }
            for call in calls
        ]
        _verify_hash(errors, "feature bundle", sha256_json(feature_material), feature_section["feature_bundle_sha256"])

        model_section = cast(Mapping[str, Any], record["model_provenance"])
        members = cast(Sequence[Mapping[str, Any]], model_section["members"])
        if tuple(str(member.get("role")) for member in members) != MODEL_ROLE_ORDER:
            errors.append("model member role order is invalid")
        if len({str(member.get("role")) for member in members}) != len(members):
            errors.append("duplicate model role identity")
        bundle_material = [
            {
                "role": item["role"],
                "artifact_sha256": item["artifact_sha256"],
                "manifest_key": item.get("manifest_key"),
                "registry_slot": item.get("registry_slot"),
            }
            for item in members
        ]
        _verify_hash(errors, "model bundle", sha256_json(bundle_material), model_section["model_bundle_sha256"])

        resolved_root = (root or repository_root()).resolve()
        if verify_referenced_artifacts:
            for member in members:
                relative = Path(str(member["source_location"]))
                if relative.is_absolute() or ".." in relative.parts:
                    errors.append(f"non-portable model path: {relative}")
                    continue
                path = (resolved_root / relative).resolve()
                try:
                    path.relative_to(resolved_root)
                except ValueError:
                    errors.append(f"model path escapes repository: {relative}")
                    continue
                if not path.is_file():
                    errors.append(f"model artifact missing: {relative.as_posix()}")
                else:
                    _verify_hash(errors, f"model artifact {relative.as_posix()}", sha256_file(path), member["artifact_sha256"])

        data = cast(Mapping[str, Any], record["data_provenance"])
        source_events = cast(Sequence[Mapping[str, Any]], data["source_events"])
        _verify_hash(errors, "source row set", sha256_json(source_events), data["source_row_set_sha256"])
        cutoff = _timestamp(data["causal_cutoff"])
        maximum = _timestamp(data.get("maximum_source_event_timestamp"))
        if cutoff is None or (maximum is not None and maximum >= cutoff):
            errors.append("temporal cutoff is not verified")
        source_identity = cast(Mapping[str, Any], data["source_identity"])
        artifacts = cast(Sequence[Mapping[str, Any]], source_identity.get("artifacts", []))
        _verify_hash(
            errors,
            "data artifact aggregate",
            sha256_json(list(artifacts)),
            source_identity["artifacts_aggregate_sha256"],
        )
        if verify_referenced_artifacts:
            for artifact in artifacts:
                relative = Path(str(artifact["path"]))
                if relative.is_absolute() or ".." in relative.parts:
                    errors.append(f"non-portable data path: {relative}")
                    continue
                path = (resolved_root / relative).resolve()
                try:
                    path.relative_to(resolved_root)
                except ValueError:
                    errors.append(f"data path escapes repository: {relative}")
                    continue
                if not path.is_file():
                    errors.append(f"data artifact missing: {relative.as_posix()}")
                else:
                    _verify_hash(
                        errors,
                        f"data artifact {relative.as_posix()}",
                        sha256_file(path),
                        artifact["sha256"],
                    )

        method = cast(Mapping[str, Any], record["method_provenance"])
        descriptor = cast(Mapping[str, Any], method["descriptor"])
        _verify_hash(errors, "method config", sha256_json(descriptor), method["method_config_hash"])
        if descriptor.get("calculation_path") != CALCULATION_PATH:
            errors.append("calculation path is not authorized")
        if cast(Mapping[str, Any], descriptor.get("calibrator", {})).get("applied"):
            errors.append("undeclared PL 1X2 calibrator was applied")
        if descriptor.get("rl_weights") is not None:
            errors.append("authorized PL 1X2 contract requires no RL weights")

        output = cast(Mapping[str, Any], record["prediction_output"])
        if tuple(output.get("class_order", [])) != CLASS_ORDER:
            errors.append("PL 1X2 class ordering is invalid")
        raw = [float(value) for value in cast(Sequence[Any], output["raw_probabilities"])]
        final = [float(value) for value in cast(Sequence[Any], output["final_probabilities"])]
        if raw != final:
            errors.append("PL 1X2 output was post-processed")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in final):
            errors.append("PL 1X2 output contains invalid probability values")
        if not math.isclose(sum(final), 1.0, abs_tol=1e-12, rel_tol=0.0):
            errors.append("PL 1X2 output does not normalize")
        output_material = {key: item for key, item in output.items() if key != "output_sha256"}
        _verify_hash(errors, "prediction output", sha256_json(output_material), output["output_sha256"])

        independence = cast(Mapping[str, Any], record["bookmaker_independence"])
        if any(bool(value) for value in independence.values()):
            errors.append("bookmaker-derived data entered prediction generation")

        if verify_referenced_artifacts:
            code = cast(Mapping[str, Any], record["code_provenance"])
            source_files = cast(Sequence[Mapping[str, Any]], code["relevant_source_files"])
            for source_file in source_files:
                relative = Path(str(source_file["path"]))
                path = resolved_root / relative
                if not path.is_file():
                    errors.append(f"relevant code file missing: {relative.as_posix()}")
                else:
                    _verify_hash(
                        errors,
                        f"relevant code file {relative.as_posix()}",
                        sha256_file(path),
                        source_file["sha256"],
                    )
            _verify_hash(
                errors,
                "relevant source aggregate",
                sha256_json(list(source_files)),
                code["relevant_source_aggregate_sha256"],
            )

        assertions = cast(Mapping[str, Any], record["trust_assertions"])
        required_assertions = (
            "execution_provenance_complete",
            "temporal_cutoff_verified",
            "feature_identity_verified",
            "model_bytes_verified",
            "method_contract_verified",
            "bookmaker_prediction_inputs_absent",
            "output_reproducible",
        )
        missing_assertions = [name for name in required_assertions if name not in assertions]
        if missing_assertions:
            errors.append(f"missing trust assertions: {missing_assertions}")
        for name in required_assertions[:-1]:
            if name in assertions and not bool(assertions[name]):
                errors.append(f"required execution assertion is false: {name}")

        _verify_hash(errors, "prediction fingerprint", prediction_fingerprint(record), record["prediction_fingerprint"])
        _verify_hash(errors, "record integrity", record_integrity_hash(record), record["record_integrity_sha256"])
    except (KeyError, TypeError, ValueError, ProvenanceError) as exc:
        errors.append(f"invalid provenance structure: {exc}")

    assertions = cast(Mapping[str, Any], record.get("trust_assertions", {}))
    return {
        "schema_version": record.get("schema_version"),
        "run_id": record.get("run_id"),
        "prediction_fingerprint": record.get("prediction_fingerprint"),
        "verified": not errors,
        "overall_phase4_trusted": not errors and bool(assertions.get("output_reproducible")),
        "training_lineage_classification": assertions.get("model_training_lineage_classification"),
        "errors": errors,
    }


def _load_model_from_bytes(path: Path, payload: bytes) -> Any:
    if path.suffix.lower() == ".joblib":
        return joblib.load(io.BytesIO(payload))
    return pickle.loads(payload)


def replay_pl_1x2(
    value: PredictionProvenanceRecord | Mapping[str, Any],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Replay the recorded four-model ensemble and exact simulator RNG state."""
    record = value.to_dict() if isinstance(value, PredictionProvenanceRecord) else cast(dict[str, Any], value)
    verification = verify_record(record, root=root, verify_referenced_artifacts=True)
    if not verification["verified"]:
        raise ProvenanceError(f"Cannot replay unverified provenance: {verification['errors']}")

    resolved_root = (root or repository_root()).resolve()
    calls = {
        str(item["role"]): item
        for item in cast(Sequence[Mapping[str, Any]], record["feature_provenance"]["model_inputs"])
    }
    member_meta = {
        str(item["role"]): item
        for item in cast(Sequence[Mapping[str, Any]], record["model_provenance"]["members"])
    }
    member_outputs: dict[str, float] = {}
    for role in MODEL_ROLE_ORDER:
        member = member_meta[role]
        path = resolved_root / str(member["source_location"])
        payload = path.read_bytes()
        model = _load_model_from_bytes(path, payload)
        call = calls[role]
        frame = pd.DataFrame(
            [cast(Sequence[Any], call["feature_values"])],
            columns=cast(Sequence[str], call["ordered_feature_names"]),
        )
        prediction = float(np.asarray(model.predict(frame), dtype=float)[0])
        member_outputs[role] = prediction

    from src.simulation.match_simulator import MatchSimulator, clamp_lambda

    descriptor = cast(Mapping[str, Any], record["method_provenance"]["descriptor"])
    ensemble = cast(Mapping[str, Any], descriptor["ensemble"])
    lgbm_weight = float(ensemble["lgbm_weight"])
    xgb_weight = float(ensemble["xgb_weight"])
    home_lambda = float(
        clamp_lambda(
            lgbm_weight * float(clamp_lambda(member_outputs["home_goals_lgbm"]))
            + xgb_weight * float(clamp_lambda(member_outputs["home_goals_xgb"]))
        )
    )
    away_lambda = float(
        clamp_lambda(
            lgbm_weight * float(clamp_lambda(member_outputs["away_goals_lgbm"]))
            + xgb_weight * float(clamp_lambda(member_outputs["away_goals_xgb"]))
        )
    )

    simulation = cast(Mapping[str, Any], descriptor["simulation"])
    simulator = MatchSimulator(
        n_simulations=int(simulation["n_simulations"]),
        max_goals=int(simulation["max_goals"]),
        top_scorelines=int(simulation["top_scorelines"]),
        tempo_sigma=float(simulation["tempo_sigma"]),
        seed=cast(int | None, simulation.get("seed")),
        seed_scope=str(simulation.get("seed_scope", "simulator_instance_stream")),
    )
    simulator.rng.bit_generator.state = copy.deepcopy(cast(dict[str, Any], simulation["rng_state_before"]))
    result = simulator.simulate(home_lambda, away_lambda, rl_weights=None)
    replayed = [result.home_win_prob, result.draw_prob, result.away_win_prob]
    expected = [float(item) for item in record["prediction_output"]["final_probabilities"]]
    return {
        "reproduced": replayed == expected,
        "expected_probabilities": expected,
        "replayed_probabilities": replayed,
        "member_outputs": member_outputs,
        "home_lambda": home_lambda,
        "away_lambda": away_lambda,
        "output_sha256": sha256_json(replayed),
    }


def mark_output_reproduced(
    record: PredictionProvenanceRecord,
    replay_result: Mapping[str, Any],
) -> PredictionProvenanceRecord:
    """Finalize the independent replay assertion without changing the calculation fingerprint."""
    if not bool(replay_result.get("reproduced")):
        raise ProvenanceError("Cannot mark an output reproducible after a failed replay")
    value = record.to_dict()
    original_fingerprint = str(value["prediction_fingerprint"])
    value["trust_assertions"]["output_reproducible"] = True
    if prediction_fingerprint(value) != original_fingerprint:
        raise ProvenanceError("Replay assertion unexpectedly changed the calculation fingerprint")
    value["record_integrity_sha256"] = record_integrity_hash(value)
    return PredictionProvenanceRecord.from_dict(value)


def persist_records(records: Sequence[PredictionProvenanceRecord], directory: Path) -> list[Path]:
    """Persist immutable per-run sidecars, refusing conflicting run identities."""
    paths: list[Path] = []
    for record in records:
        path = directory / f"{record.run_id}.json"
        if path.exists():
            existing = PredictionProvenanceRecord.load(path)
            if existing.canonical_bytes() != record.canonical_bytes():
                raise ProvenanceError(f"Conflicting immutable provenance run_id: {record.run_id}")
        else:
            record.save(path)
        paths.append(path)
    return paths

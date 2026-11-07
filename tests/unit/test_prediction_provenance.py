from __future__ import annotations

import copy
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.predictions.provenance import (
    CLASS_ORDER,
    MODEL_ROLE_ORDER,
    PredictionCapture,
    PredictionProvenanceRecord,
    ProvenanceError,
    canonical_json,
    prediction_fingerprint,
    replay_pl_1x2,
    sha256_bytes,
    sha256_json,
    verify_record,
)
from src.simulation.match_simulator import MatchSimulator


class ConstantRegressor:
    def __init__(self, value: float, identity: dict[str, object]) -> None:
        self.value = value
        self.meta = {"features": ["form", "rest"], "version": "test"}
        self._ffps_provenance = identity

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.value] * len(frame), dtype=float)


def _identity(role: str, artifact_hash: str, source: str) -> dict[str, object]:
    return {
        "source_location": source,
        "artifact_sha256": artifact_hash,
        "size_bytes": 1,
        "logical_name": role,
        "requested_name": role,
        "model_type": "constant_regressor",
        "model_version": "test",
        "configured_filename": Path(source).name,
        "manifest_key": f"{role}_vtest_PL",
        "registry_slot": f"{role}_PL",
        "manifest_sha256": "1" * 64,
        "metadata_sha256": "2" * 64,
        "training_lineage": {
            "classification": "unproven",
            "reference": None,
            "reason": "test legacy artifact",
        },
        "execution_traceable": True,
    }


def _data_provenance() -> dict[str, object]:
    return {
        "source_identity": {
            "source_type": "test_fixture",
            "artifacts": [],
            "artifacts_aggregate_sha256": sha256_json([]),
            "ordered_columns": ["form", "rest"],
            "row_count": 1,
            "row_set_sha256": sha256_json({"form": 1.1, "rest": 6.0}),
        },
        "causal_cutoff": "2026-08-30T12:00:00+00:00",
        "prediction_timestamp": "2026-08-30T12:00:00+00:00",
        "fixture_cutoff_policy": "exact_kickoff_exclusive",
        "maximum_source_event_timestamp": "2026-08-20T12:00:00+00:00",
        "source_event_count": 1,
        "source_events": [
            {
                "event_timestamp": "2026-08-20T12:00:00+00:00",
                "match_identity": "source-1",
                "row_sha256": "3" * 64,
            }
        ],
        "source_row_set_sha256": sha256_json(
            [
                {
                    "event_timestamp": "2026-08-20T12:00:00+00:00",
                    "match_identity": "source-1",
                    "row_sha256": "3" * 64,
                }
            ]
        ),
        "temporal_cutoff_verified": True,
        "cutoff_violations": [],
    }


def _build_record(
    tmp_path: Path,
    *,
    values: tuple[float, float, float, float] = (1.4, 1.1, 1.6, 1.2),
    seed: int = 42,
    run_id: str | None = None,
) -> PredictionProvenanceRecord:
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    contexts = (
        "fixture:GoalsH:LGBM",
        "fixture:GoalsA:LGBM",
        "fixture:GoalsH:XGB",
        "fixture:GoalsA:XGB",
    )
    capture = PredictionCapture(
        prediction_timestamp="2026-08-30T12:00:00+00:00",
        data_provenance=_data_provenance(),
        **({"run_id": run_id} if run_id else {}),
    )
    models: list[ConstantRegressor] = []
    with capture:
        for role, context, value in zip(MODEL_ROLE_ORDER, contexts, values):
            path = model_dir / f"{role}.joblib"
            # The identity is external to model prediction state; save the final
            # executable then bind exactly those final bytes to the live object.
            model_for_disk = ConstantRegressor(value, {})
            joblib.dump(model_for_disk, path)
            payload = path.read_bytes()
            model = ConstantRegressor(
                value,
                _identity(role, sha256_bytes(payload), f"models/{path.name}"),
            )
            models.append(model)
            frame = pd.DataFrame([[1.1, 6.0]], columns=["form", "rest"])
            raw = float(model.predict(frame)[0])
            capture.record_model_call(
                model=model,
                frame=frame,
                context=context,
                raw_output=raw,
                effective_output=raw,
            )

    home_lambda = 0.6 * values[0] + 0.4 * values[2]
    away_lambda = 0.6 * values[1] + 0.4 * values[3]
    simulator = MatchSimulator(n_simulations=1_000, league="PL", seed=seed)
    result = simulator.simulate(home_lambda, away_lambda)
    fixture = pd.Series(
        {
            "competition": "PREMIER LEAGUE",
            "league": "PL",
            "home_team": "HOME FC",
            "away_team": "AWAY FC",
            "date": "2026-08-31T15:00:00+00:00",
            "kickoff_utc": "2026-08-31T15:00:00+00:00",
            "match_id": "fixture-1",
        }
    )
    record = capture.finalize(
        fixture=fixture,
        league="PL",
        probabilities={
            "home": result.home_win_prob,
            "draw": result.draw_prob,
            "away": result.away_win_prob,
        },
        simulator=simulator,
    )
    repository = Path(__file__).resolve().parents[2]
    for source in record.to_dict()["code_provenance"]["relevant_source_files"]:
        relative = Path(source["path"])
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, target)
    return record


def test_canonical_serialization_is_deterministic_and_rejects_nonfinite() -> None:
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})
    with pytest.raises(ProvenanceError, match="NaN and Infinity"):
        canonical_json({"value": float("nan")})


def test_fingerprint_determinism_separate_run_identity_and_save_load(tmp_path: Path) -> None:
    first = _build_record(tmp_path, run_id="11111111-1111-4111-8111-111111111111")
    second = _build_record(tmp_path, run_id="22222222-2222-4222-8222-222222222222")
    assert first.run_id != second.run_id
    assert first.prediction_fingerprint == second.prediction_fingerprint

    path = tmp_path / "record.json"
    first.save(path)
    loaded = PredictionProvenanceRecord.load(path)
    assert loaded.canonical_bytes() == first.canonical_bytes()
    assert verify_record(loaded, root=tmp_path)["verified"] is True


def test_replay_reproduces_exact_output_and_capture_changes_no_probability(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    replay = replay_pl_1x2(record, root=tmp_path)
    assert replay["reproduced"] is True
    assert replay["replayed_probabilities"] == record.to_dict()["prediction_output"]["final_probabilities"]


def test_tamper_class_order_and_missing_required_fields_fail(tmp_path: Path) -> None:
    record = _build_record(tmp_path).to_dict()
    tampered = copy.deepcopy(record)
    tampered["fixture"]["home_team"] = "TAMPERED"
    assert verify_record(tampered, root=tmp_path)["verified"] is False

    reordered = copy.deepcopy(record)
    reordered["prediction_output"]["class_order"] = list(reversed(CLASS_ORDER))
    report = verify_record(reordered, root=tmp_path)
    assert report["verified"] is False
    assert any("class ordering" in error for error in report["errors"])

    missing = copy.deepcopy(record)
    del missing["model_provenance"]
    assert verify_record(missing, root=tmp_path)["verified"] is False


def test_referenced_model_artifact_tamper_is_detected(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    assert verify_record(record, root=tmp_path)["verified"] is True
    artifact = tmp_path / record.to_dict()["model_provenance"]["members"][0]["source_location"]
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    report = verify_record(record, root=tmp_path)
    assert report["verified"] is False
    assert any("model artifact" in error for error in report["errors"])


def test_effective_model_feature_and_method_changes_change_fingerprint(tmp_path: Path) -> None:
    record = _build_record(tmp_path).to_dict()

    model_changed = copy.deepcopy(record)
    model_changed["model_provenance"]["members"][0]["artifact_sha256"] = "f" * 64
    assert prediction_fingerprint(model_changed) != record["prediction_fingerprint"]

    feature_changed = copy.deepcopy(record)
    feature_changed["feature_provenance"]["model_inputs"][0]["feature_values"][0] = 9.0
    assert prediction_fingerprint(feature_changed) != record["prediction_fingerprint"]

    method_changed = copy.deepcopy(record)
    method_changed["method_provenance"]["descriptor"]["simulation"]["seed"] = 999
    assert prediction_fingerprint(method_changed) != record["prediction_fingerprint"]


def test_legacy_training_lineage_is_unproven_not_silently_trusted(tmp_path: Path) -> None:
    record = _build_record(tmp_path).to_dict()
    assert record["trust_assertions"]["execution_provenance_complete"] is True
    assert record["trust_assertions"]["model_training_lineage_proven"] is False
    assert record["trust_assertions"]["model_training_lineage_classification"] == "unproven"
    assert all(
        member["training_lineage"]["classification"] == "unproven"
        for member in record["model_provenance"]["members"]
    )


def test_bookmaker_feature_is_rejected(tmp_path: Path) -> None:
    role = MODEL_ROLE_ORDER[0]
    model = ConstantRegressor(1.0, _identity(role, "a" * 64, "models/test.joblib"))
    capture = PredictionCapture(
        prediction_timestamp="2026-08-30T12:00:00+00:00",
        data_provenance=_data_provenance(),
    )
    with pytest.raises(ProvenanceError, match="Bookmaker-derived"):
        capture.record_model_call(
            model=model,
            frame=pd.DataFrame([[1.8]], columns=["bet365_odds"]),
            context="fixture:GoalsH:LGBM",
            raw_output=1.0,
            effective_output=1.0,
        )

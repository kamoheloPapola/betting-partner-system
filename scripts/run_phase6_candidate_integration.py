"""Run the frozen Phase 6 bundle through Phase 4 without changing training state."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, cast

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_phase4_prediction_provenance import _input_frames
from scripts.run_phase6_controlled_retraining import (
    ACTIVE_MANIFEST,
    CANDIDATE_CLASSIFICATION,
    DEFAULT_RUN_DIR,
    Phase6EvidenceError,
    _materialize_inputs,
    _partition_sources,
    _phase4_features,
    _read_json,
    _write_json,
)
from src.cli.commands.prediction import _calc_goals
from src.predictions.provenance import (
    PredictionCapture,
    data_provenance_for_fixture,
    mark_output_reproduced,
    replay_pl_1x2,
    sha256_file,
    verify_record,
)
from src.simulation.match_simulator import DEFAULT_N_SIMULATIONS, MatchSimulator


def _descriptive_calibration_scores(run_dir: Path) -> dict[str, float]:
    """Derive guard metadata from the already-frozen validation predictions.

    This is post-fit descriptive metadata.  It is never used for selection,
    retrying, hyperparameter changes, early stopping, or calibration fitting.
    """
    manifest = _read_json(run_dir / "executions" / "primary_manifest.json")
    inputs = _materialize_inputs()
    scores: dict[str, float] = {}
    for item in manifest["models"]:
        role = str(item["role"])
        target = cast(
            pd.Series,
            inputs["y_validation_home"] if role.startswith("home") else inputs["y_validation_away"],
        ).to_numpy(dtype=np.float64)
        prediction = np.asarray(item["prediction_values"], dtype=np.float64)
        event = (target > 0.0).astype(np.float64)
        probability = np.clip(1.0 - np.exp(-np.maximum(0.05, prediction)), 1e-6, 1.0 - 1e-6)
        scores[role] = float(np.mean(np.square(event - probability)))
    return scores


def integrate(run_dir: Path) -> dict[str, Any]:
    integration_dir = run_dir / "integration"
    if integration_dir.exists():
        raise Phase6EvidenceError("Phase 6 integration evidence already exists")
    bundle = _read_json(run_dir / "candidate_bundle.json")
    features = _phase4_features()
    calibration_scores = _descriptive_calibration_scores(run_dir)
    models: dict[str, Any] = {}
    for item in bundle["models"]:
        role = str(item["role"])
        artifact = cast(Mapping[str, Any], item["artifact"])
        path = ROOT / str(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise Phase6EvidenceError(f"Candidate artifact substitution before integration: {role}")
        model = joblib.load(path)
        model.meta = {
            "name": f"phase6:{role}",
            "version": bundle["bundle_id"],
            "features": features,
            "type": "lgbm_regressor_poisson" if role.endswith("lgbm") else "xgb_regressor_count_poisson",
            "metrics": {"calibration_score": calibration_scores[role]},
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
        "descriptive_guard_calibration_scores": calibration_scores,
        "guard_metadata_used_for_selection_or_retry": False,
        "integration_harness": {
            "path": "scripts/run_phase6_candidate_integration.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "active_manifest_sha256": sha256_file(ACTIVE_MANIFEST),
    }
    required = (
        "finite_and_normalized",
        "candidate_artifact_hashes_recorded",
        "phase4_overall_trusted",
        "independent_replay",
        "no_model_calibrator_applied",
        "no_match_outcome_classifier",
        "bookmaker_inputs_absent",
    )
    if not all(result[key] for key in required):
        raise Phase6EvidenceError(f"Candidate integration assertion failed: {result}")
    _write_json(integration_dir / "integration_result.json", result)
    return result


if __name__ == "__main__":
    try:
        print(json.dumps(integrate(DEFAULT_RUN_DIR), indent=2, sort_keys=True))
    except Phase6EvidenceError as exc:
        print(f"Phase 6 integration error: {exc}", file=sys.stderr)
        raise SystemExit(1)

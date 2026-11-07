"""Integrity tests for the authorized Phase 3 PL 1X2 temperature experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.phase3_pl_1x2_temperature import (
    CLASS_ORDER,
    MulticlassTemperatureCalibrator,
    Phase3TemperatureIntegrityError,
    build_calibrator_payload,
    build_partition_contract,
    calibration_fit_rows,
    fit_temperature_calibrator,
    load_calibrator_artifact,
    multiclass_metrics,
    probabilities_and_labels,
    row_ids_hash,
    sha256_file,
    validate_fit_rows,
    validate_trusted_calibrator_payload,
    write_json,
)


@pytest.fixture(scope="module")
def contract():
    return build_partition_contract()


@pytest.fixture(scope="module")
def fitted(contract):
    return fit_temperature_calibrator(calibration_fit_rows(contract), contract.manifest)


def test_season_partitions_are_exact_and_identity_disjoint(contract) -> None:
    partitions = contract.manifest["partitions"]
    assert {
        name: (item["count"], item["date_min"], item["date_max"])
        for name, item in partitions.items()
    } == {
        "model_train": (1505, "2019-08-17", "2023-05-28"),
        "model_validation": (379, "2023-08-11", "2024-05-19"),
        "calibration_fit": (379, "2024-08-16", "2025-05-25"),
        "final_test": (179, "2025-08-15", "2025-12-28"),
    }
    assert all(item["match_hash_overlap_count"] == 0 for item in contract.manifest["pairwise_disjointness"])
    assert all(item["canonical_fixture_overlap_count"] == 0 for item in contract.manifest["pairwise_disjointness"])


def test_frozen_phase1_live_path_and_temporal_evidence_are_retained(contract) -> None:
    for record in (*contract.calibration_fit, *contract.final_test):
        assert record["pathway"]["goal_function"] == "src.cli.commands.prediction._calc_goals"
        assert record["pathway"]["simulator"] == "src.simulation.match_simulator.MatchSimulator.simulate"
        assert record["pathway"]["goal_calibrators_loaded"] is False
        assert record["pathway"]["model_level_calibrators_present"] is False
        assert record["rng"]["n_simulations"] == 50_000
        assert record["temporal_evidence"]["strict_cutoff_passed"] is True
        assert record["temporal_evidence"]["same_day_sources_excluded"] is True
        assert record["temporal_evidence"]["max_source_date"] < record["fixture"]["date"]


def test_final_test_fixture_passed_to_fit_fails_closed(contract) -> None:
    rows = calibration_fit_rows(contract)
    rows[0] = dict(rows[0], fixture_id=contract.final_test[0]["match_hash"])
    with pytest.raises(Phase3TemperatureIntegrityError, match="Final-test fixture"):
        validate_fit_rows(rows, contract.manifest)


def test_partition_overlap_fails_closed(contract) -> None:
    manifest = copy.deepcopy(contract.manifest)
    calibration_id = manifest["partitions"]["calibration_fit"]["fixture_ids"][0]
    manifest["partitions"]["final_test"]["fixture_ids"].append(calibration_id)
    with pytest.raises(Phase3TemperatureIntegrityError, match="Partition manifest is not disjoint"):
        validate_fit_rows(calibration_fit_rows(contract), manifest)


def test_bookmaker_input_fails_closed(contract) -> None:
    rows = calibration_fit_rows(contract)
    rows[0]["raw_probabilities"]["bookmaker_odds"] = 2.0
    with pytest.raises(Phase3TemperatureIntegrityError, match="Bookmaker input forbidden"):
        validate_fit_rows(rows, contract.manifest)


def test_final_test_outcome_cannot_enter_fit_data(contract) -> None:
    rows = calibration_fit_rows(contract)
    final_record = contract.final_test[0]
    rows[0] = {
        "fixture_id": final_record["match_hash"],
        "raw_probabilities": final_record["raw_probabilities"],
        "actual_result": final_record["actual_result"],
    }
    with pytest.raises(Phase3TemperatureIntegrityError, match="Final-test fixture"):
        fit_temperature_calibrator(rows, contract.manifest)


def test_reordered_classes_fail_closed(contract) -> None:
    rows = calibration_fit_rows(contract)
    raw = rows[0]["raw_probabilities"]
    rows[0]["raw_probabilities"] = {
        "draw": raw["draw"],
        "home": raw["home"],
        "away": raw["away"],
    }
    with pytest.raises(Phase3TemperatureIntegrityError, match="Class order"):
        validate_fit_rows(rows, contract.manifest)


def test_fit_reproducibility_and_positive_temperature(contract, fitted) -> None:
    first, first_diagnostics = fitted
    second, second_diagnostics = fit_temperature_calibrator(calibration_fit_rows(contract), contract.manifest)
    assert first.temperature > 0.0
    assert second.temperature == pytest.approx(first.temperature, abs=1e-12)
    assert second_diagnostics["fit_fixture_ids_sha256"] == first_diagnostics["fit_fixture_ids_sha256"]


def test_probability_normalization_and_numerical_extremes(fitted) -> None:
    calibrator, _ = fitted
    extreme = np.asarray([
        [1.0 - 2e-300, 1e-300, 1e-300],
        [1e-300, 0.5, 0.5 - 1e-300],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
    ])
    calibrated = calibrator.transform(extreme)
    assert np.all(np.isfinite(calibrated))
    assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)


def test_temperature_preserves_argmax_and_raw_probabilities(fitted, contract) -> None:
    calibrator, _ = fitted
    raw, _ = probabilities_and_labels(contract.final_test)
    before = raw.copy()
    calibrated = calibrator.transform(raw)
    assert np.array_equal(raw, before)
    assert np.array_equal(np.argmax(raw, axis=1), np.argmax(calibrated, axis=1))


def test_artifact_round_trip_reproduces_outputs(tmp_path: Path, contract, fitted) -> None:
    calibrator, diagnostics = fitted
    payload = build_calibrator_payload(
        calibrator,
        diagnostics,
        contract,
        calibrator_id="phase3-test-temperature",
        source_code_path=Path(__file__).resolve().parents[2] / "src" / "evaluation" / "phase3_pl_1x2_temperature.py",
    )
    artifact_path = tmp_path / "calibrator.json"
    manifest_path = tmp_path / "calibrator_manifest.json"
    write_json(artifact_path, payload)
    write_json(manifest_path, {
        "calibrator_id": payload["calibrator_id"],
        "artifact_sha256": sha256_file(artifact_path),
    })
    loaded, loaded_payload = load_calibrator_artifact(artifact_path, manifest_path, contract.manifest)
    raw, _ = probabilities_and_labels(contract.final_test)
    assert loaded_payload["class_order"] == list(CLASS_ORDER)
    assert loaded.transform(raw) == pytest.approx(calibrator.transform(raw), abs=1e-15)


def test_missing_partition_provenance_cannot_be_marked_trusted(contract) -> None:
    legacy = {
        "trust_status": "PHASE3_TRUSTED_EVALUATION_ONLY",
        "method": "multiclass_temperature_scaling",
    }
    with pytest.raises(Phase3TemperatureIntegrityError, match="provenance missing"):
        validate_trusted_calibrator_payload(legacy, contract.manifest)


def test_metrics_use_explicit_home_draw_away_semantics(contract) -> None:
    raw, labels = probabilities_and_labels(contract.calibration_fit)
    metrics = multiclass_metrics(labels, raw)
    assert metrics["class_order"] == ["home", "draw", "away"]
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["binning"]["method"] == "10 equal-width bins on [0,1]"


def test_identity_hash_changes_when_fit_identity_changes(contract) -> None:
    original = contract.manifest["partitions"]["calibration_fit"]["fixture_ids"]
    changed = list(original)
    changed[-1] = contract.final_test[-1]["match_hash"]
    assert row_ids_hash(changed) != contract.manifest["partitions"]["calibration_fit"]["fixture_ids_sha256"]


def test_invalid_serialized_temperature_or_class_order_fails_closed() -> None:
    with pytest.raises(Phase3TemperatureIntegrityError, match="positive"):
        MulticlassTemperatureCalibrator(temperature=0.0)
    with pytest.raises(Phase3TemperatureIntegrityError, match="class order"):
        MulticlassTemperatureCalibrator(temperature=1.0, class_order=("away", "draw", "home"))


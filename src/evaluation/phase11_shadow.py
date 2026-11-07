"""Phase 11 adapter into the existing Phase 10 prospective machinery."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation import phase10_current_data as phase10
from src.evaluation import phase10_shadow
from src.evaluation import phase11_current_source as data
from src.evaluation.league_prospective import validate_pair
from src.evaluation.phase7p_prospective import ImmutableCohortStore
from src.ml.bundle_activation import persist_new


def sealed(value: dict[str, Any], key: str) -> None:
    data.check(value.get(key) == data.sha256_json({k: v for k, v in value.items() if k != key}),
               "EVIDENCE_FINGERPRINT_DRIFT")


def reference(ref: dict[str, Any]) -> Path:
    path = (data.ROOT / ref.get("path", "")).resolve()
    data.check(path.is_relative_to(data.ROOT), "REFERENCE_PATH_ESCAPE")
    data.check(path.exists() and data.sha256_file(path) == ref.get("sha256"), "ARTIFACT_DRIFT")
    data.check(path.stat().st_size == ref.get("size_bytes"), "ARTIFACT_SIZE_DRIFT")
    return path


def load_qualification(path: str | Path):
    qualification_path = Path(path).resolve()
    data.check(qualification_path.is_relative_to(data.ROOT / "baselines/phase11"),
               "QUALIFICATION_PATH_ESCAPE")
    q = data.read(qualification_path)
    data.check(q.get("schema_version") == "ffps.phase11_qualification.v1", "WRONG_QUALIFICATION_SCHEMA")
    sealed(q, "qualification_sha256")
    data.check(q.get("league") in data.LEAGUES, "WRONG_LEAGUE")
    assessment = q.get("assessment", {})
    data.check(assessment.get("qualified") and not assessment.get("blockers"),
               "CURRENT_DATA_UNQUALIFIED")
    data.check(q.get("feature_readiness") and not q.get("feature_failures") and q.get("feature_proofs"),
               "CURRENT_FEATURES_UNQUALIFIED")
    data.check(all(all(item["fields"].values()) for item in assessment["cross_provider_overlap"]),
               "ALTERNATIVE_PROVIDER_DISAGREEMENT")
    index = data.read(reference(q["source_index"]))
    data.check(index.get("league") == q["league"], "CROSS_LEAGUE_SOURCE")
    reproduction = data.feature_reproduction(q["league"], q["ordered_features"])
    data.check(reproduction == q["historical_feature_reproduction"] and reproduction["reproduced"],
               "FEATURE_REPRODUCTION_MISMATCH")
    first = assessment["upcoming"][0]
    matrix, fixture = data.target_matrix(assessment, first, as_of=q["as_of"], source_index=index)
    proof = phase10.validate_features(matrix, fixture, q["ordered_features"], q["as_of"])
    data.check(proof == q["feature_proofs"][0], "FEATURE_PROOF_REPLAY_MISMATCH")
    controller = phase10_shadow.ShadowController(q)
    return q, index, controller


def freeze(path: str | Path, league: str, qualification_path: str | Path) -> dict[str, Any]:
    run = data.open_run(path)
    data.check(league in data.LEAGUES, "WRONG_LEAGUE")
    qualification_path = Path(qualification_path).resolve()
    q, _, controller = load_qualification(qualification_path)
    data.check(q["league"] == league, "CROSS_LEAGUE_QUALIFICATION")
    contract_path = qualification_path.parent / "current_source_contract.json"
    contract = data.read(contract_path)
    sealed(contract, "contract_sha256")
    data.check(reference(contract["qualification"]) == qualification_path,
               "CONTRACT_QUALIFICATION_MISMATCH")
    frozen_at = phase10.instant(data.now())
    future = [event for event in q["assessment"]["upcoming"]
              if phase10.instant(event["kickoff_at"]) > frozen_at + timedelta(seconds=1)]
    data.check(future, "NO_UPCOMING_CAUSAL_FIXTURES")
    season = str(q["assessment"]["season"])
    end = phase10.instant(q["assessment"]["season_evidence"][season]["result_set_last"] +
                          "T00:00:00Z") + timedelta(days=1)
    directory = run / league
    protocol = {
        "schema_version": "ffps.league_prospective_protocol.v1", "league": league,
        "phase": 11, "frozen_at": frozen_at.isoformat(),
        "start_at": (frozen_at + timedelta(seconds=1)).isoformat(), "end_at": end.isoformat(),
        "season": q["assessment"]["season"],
        "qualification": data.identity(qualification_path),
        "current_data_contract": data.identity(contract_path),
        "candidate_bundle_id": controller.candidate_id,
        "active_bundle_id": controller.legacy_id,
        "feature_schema_sha256": q["feature_schema_sha256"],
        "evaluation_specification_sha256": controller.owner.frozen["evaluation_specification"]["specification_sha256"],
        "decision_implementation_sha256": data.sha256_file(data.ROOT / "src/evaluation/phase7_shadow.py"),
        "cohort_path": (directory / "cohort").relative_to(data.ROOT).as_posix(),
        "bootstrap": {"resamples": 10000, "confidence_level": 0.95, "seed": 42},
        "eligible_fixture_policy": "all current-season league fixtures with TIMED or SCHEDULED status and complete causal features",
        "first_collection_policy": "earliest uncollected fixture that remains genuinely future at capture",
        "common_rng": "SHA256([league, fixture_id]) first 8 hex; identical RNG state for paired models",
        "promotion_authorized": False,
    }
    protocol["protocol_sha256"] = data.sha256_json(protocol)
    persist_new(directory / "prospective_protocol.json", protocol)
    return protocol


def load_protocol(path: str | Path, league: str):
    run = data.run_path(path)
    protocol_path = run / league / "prospective_protocol.json"
    protocol = data.read(protocol_path)
    sealed(protocol, "protocol_sha256")
    data.check(protocol.get("league") == league and league in data.LEAGUES,
               "CROSS_LEAGUE_PROTOCOL")
    data.check(phase10.instant(protocol["frozen_at"]) < phase10.instant(protocol["start_at"]) <
               phase10.instant(protocol["end_at"]), "INVALID_HORIZON")
    q, index, controller = load_qualification(reference(protocol["qualification"]))
    data.check(protocol["active_bundle_id"] == controller.legacy_id and
               protocol["candidate_bundle_id"] == controller.candidate_id, "BUNDLE_SCOPE_MISMATCH")
    data.check(protocol["feature_schema_sha256"] == q["feature_schema_sha256"],
               "FEATURE_SCHEMA_MISMATCH")
    data.check(protocol["evaluation_specification_sha256"] ==
               controller.owner.frozen["evaluation_specification"]["specification_sha256"],
               "CROSS_LEAGUE_EVALUATION_SPECIFICATION")
    data.check(protocol["decision_implementation_sha256"] ==
               data.sha256_file(data.ROOT / "src/evaluation/phase7_shadow.py"),
               "DECISION_IMPLEMENTATION_DRIFT")
    data.check(protocol["bootstrap"] == {"resamples": 10000, "confidence_level": 0.95,
                                         "seed": 42},
               "INVALID_FROZEN_UNCERTAINTY_PROTOCOL")
    reference(protocol["current_data_contract"])
    cohort = (data.ROOT / protocol["cohort_path"]).resolve()
    data.check(cohort == protocol_path.parent / "cohort", "COHORT_SCOPE_MISMATCH")
    return protocol, cohort, controller, q, index


def collect_one(path: str | Path, league: str) -> dict[str, Any]:
    data.open_run(path)
    protocol, cohort, controller, q, index = load_protocol(path, league)
    active, candidate = [controller.owner.load(controller.bundle(bundle_id))
                         for bundle_id in (controller.legacy_id, controller.candidate_id)]
    report = {"schema_version": "ffps.phase11_collection_report.v1", "league": league,
              "started_at": data.now(), "admitted": [], "rejected": [],
              "already_present": []}
    with ImmutableCohortStore(cohort).batch_lock("phase11_collection"):
        for target in q["assessment"]["upcoming"]:
            destination = phase10_shadow.pair_path(cohort, league, target["fixture_id"])
            if destination.exists():
                pair = data.read(destination)
                validate_pair(pair, controller=controller, protocol=protocol)
                report["already_present"].append(target["fixture_id"])
                continue
            capture_at = data.now()
            try:
                data.check(phase10.instant(protocol["start_at"]) <= phase10.instant(capture_at) <
                           phase10.instant(target["kickoff_at"]) < phase10.instant(protocol["end_at"]),
                           "OUTSIDE_FIXED_HORIZON")
                matrix, fixture = data.target_matrix(
                    q["assessment"], target, as_of=capture_at, source_index=index)
                phase10.validate_features(matrix, fixture, q["ordered_features"], capture_at)
                fixture = fixture.copy()
                fixture["kickoff_utc"] = target["kickoff_at"]
                prediction_at = data.now()
                records = [phase10_shadow.capture(matrix, fixture, suite, league, prediction_at)
                           for suite in (active, candidate)]
                pair = {"league": league, "protocol_sha256": protocol["protocol_sha256"],
                        "fixture_id": target["fixture_id"], "created_at": data.now(),
                        "kickoff_at": target["kickoff_at"],
                        "qualification": protocol["qualification"],
                        "active": records[0], "candidate": records[1]}
                pair["pair_sha256"] = data.sha256_json(pair)
                validate_pair(pair, controller=controller, protocol=protocol)
                data.check(phase10.instant(pair["created_at"]) < phase10.instant(target["kickoff_at"]),
                           "LATE_PREDICTION")
                persist_new(destination, pair)
                report["admitted"].append(target["fixture_id"])
                break
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                report["rejected"].append({"fixture_id": target["fixture_id"],
                                           "code": str(exc), "error_type": type(exc).__name__})
    report["finished_at"] = data.now()
    report["report_sha256"] = data.sha256_json(report)
    persist_new(cohort / "admission_reports" / (report["report_sha256"] + ".json"), report)
    return report


def verify_cohort(path: str | Path, league: str):
    """Replay every Phase 11 pair from archived current-source bytes."""
    protocol, cohort, controller, q, index = load_protocol(path, league)
    pairs = [data.read(item) for item in sorted((cohort / "predictions").glob("*.json"))]
    settlements = [data.read(item) for item in sorted((cohort / "settlements").glob("*.json"))]
    data.check(len({item["fixture_id"] for item in pairs}) == len(pairs),
               "DUPLICATE_OBSERVATION")
    data.check(not settlements, "PHASE11_SETTLEMENT_NOT_SUPPORTED_BY_FROZEN_SOURCE_CAPTURE")
    targets = {item["fixture_id"]: item for item in q["assessment"]["upcoming"]}
    for pair in pairs:
        sealed(pair, "pair_sha256")
        validate_pair(pair, controller=controller, protocol=protocol)
        data.check(pair["qualification"] == protocol["qualification"],
                   "PAIR_QUALIFICATION_MISMATCH")
        data.check(pair["fixture_id"] in targets, "UNKNOWN_TARGET_FIXTURE")
        data.check(phase10.instant(protocol["start_at"]) <= phase10.instant(pair["created_at"]) <
                   phase10.instant(pair["kickoff_at"]) < phase10.instant(protocol["end_at"]),
                   "INVALID_PROSPECTIVE_ORDER")
        matrix, fixture = data.target_matrix(
            q["assessment"], targets[pair["fixture_id"]], as_of=pair["created_at"],
            source_index=index)
        phase10.validate_features(matrix, fixture, q["ordered_features"], pair["created_at"])
        expected = fixture[q["ordered_features"]].to_numpy(float)
        for record in (pair["active"], pair["candidate"]):
            for features in record["feature_provenance"]["model_inputs"]:
                data.check(np.array_equal(np.asarray(features["feature_values"]), expected),
                           "CAUSAL_FEATURE_REPLAY_MISMATCH")
    return protocol, pairs, settlements

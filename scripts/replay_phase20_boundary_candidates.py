"""Read-only Phase 12 scientific replay for the pending Phase 20 boundary review.

This produces a review certificate outside protected evidence. It neither
approves a boundary nor changes Phase 20 settlement or historical journals.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import phase11_evidence, phase12_evidence, phase12_prospective as p
from src.evaluation import phase20_boundary as boundary

PROPOSAL = ROOT / "docs/phase20-boundary-authorization-proposal-v6.json"
DESTINATION = ROOT / "docs/phase20-boundary-scientific-replay-20260923.json"
PHASE12 = ROOT / boundary.PHASE12_RUN


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def refs_identity(path: Path) -> dict:
    return p.identity(path)


def main() -> None:
    proposal = p.read(PROPOSAL)
    p.frozen.sealed(proposal, "record_sha256")
    require(proposal["decision_state"] == "PENDING" and len(proposal["additions"]) == 183,
            "PROPOSAL_SCOPE_DRIFT")
    candidates = {row["path"] for row in proposal["additions"]
                  if row["classification"] == "UNAUTHORIZED"}
    require(len(candidates) == 167, "CANDIDATE_COUNT_DRIFT")
    for row in proposal["additions"]:
        require(refs_identity(ROOT / row["path"]) == {key: row[key]
                for key in ("path", "sha256", "size_bytes")}, "CANDIDATE_BYTES_DRIFT")

    print("Replaying frozen PD cohort", flush=True)
    pd = p.load_cohort(PHASE12, "PD")
    print("Replaying frozen BL1 cohort", flush=True)
    bl1 = p.load_cohort(PHASE12, "BL1")
    spec = p.specification(PHASE12)
    checkpoint_counts = {}
    for league, context in (("PD", pd), ("BL1", bl1)):
        checks = sorted((PHASE12 / league / "checkpoints").glob("*.json"))
        for file in checks:
            phase12_evidence.verify_checkpoint(p.read(file), context,
                                               len(spec["leagues"][league]["prefix"]))
        checkpoint_counts[league] = len(checks)
        print(f"Verified {league} checkpoints: {len(checks)}", flush=True)

    event117 = p.read(PHASE12 / "PD/journal/00000117.json")
    event118 = p.read(PHASE12 / "PD/journal/00000118.json")
    p.frozen.sealed(event117, "record_sha256")
    p.frozen.sealed(event118, "record_sha256")
    require(event117["kind"] == "incident" and
            event117["payload"]["code"] == "SOURCE_REFRESH_STARTED" and
            event117["payload"]["protocol_violation"] is False and
            event118["kind"] == "pair", "PD_JOURNAL_KIND_DRIFT")
    require(pd[6][-2:] == [event117, event118], "PD_JOURNAL_TAIL_DRIFT")
    source_run = ROOT / proposal["phase12_source_runs"][0]
    source_spec_path = source_run / "run_specification.json"
    require(event117["payload"]["source"] == refs_identity(source_spec_path),
            "PD_SOURCE_RUN_LINK_DRIFT")
    source_spec = p.read(source_spec_path)
    p.frozen.sealed(source_spec, "record_sha256")
    require(source_spec["league"] == "PD" and
            source_spec["continuation"].replace("\\", "/") == boundary.PHASE12_RUN and
            source_spec["protocol_sha256"] == pd[0]["protocol_sha256"] and
            source_spec["promotion_authorized"] is False,
            "PD_SOURCE_RUN_SCOPE_DRIFT")

    pair = event118["payload"]
    qualification_path = p.frozen.reference(pair["qualification"])
    require(qualification_path.is_relative_to(source_run / "PD/assessments/assessment-001"),
            "PD_REFRESH_QUALIFICATION_SCOPE")
    q, index, _ = p.frozen.load_qualification(qualification_path)
    source_replay = phase11_evidence.replay_source_evidence(q)
    require(source_replay["raw_sources_replayed"] == 76, "PD_SOURCE_REPLAY_COVERAGE")
    assessment_dir = qualification_path.parent
    source_paths = {source_spec_path.relative_to(ROOT).as_posix()}
    source_paths.update((source_run / "PD" / name).relative_to(ROOT).as_posix()
                        for name in ("source_index_current.json", "fotmob_index_current.json"))
    source_paths.update((assessment_dir / name).relative_to(ROOT).as_posix()
                        for name in ("qualification.json", "qualified_source_index.json",
                                     "current_source_contract.json", "provider_overlap_evidence.json",
                                     "season_boundary_evidence.json", "team_mapping_evidence.json"))
    for metadata in index["sources"]:
        raw_path = p.frozen.reference(metadata["raw"])
        metadata_path = raw_path.with_suffix(".json")
        require(p.read(metadata_path) == metadata, "PD_SOURCE_METADATA_DRIFT")
        source_paths.update((raw_path.relative_to(ROOT).as_posix(),
                             metadata_path.relative_to(ROOT).as_posix()))
    require(len(source_paths) == 161 and source_paths <= candidates,
            "PD_SOURCE_PACKAGE_INVENTORY_DRIFT")
    contract = p.read(assessment_dir / "current_source_contract.json")
    p.frozen.sealed(contract, "contract_sha256")
    require(contract["qualification"] == refs_identity(qualification_path) and
            contract["source_index"] == q["source_index"] and
            contract["feature_schema_sha256"] == q["feature_schema_sha256"] and
            contract["promotion_authorized"] is False,
            "PD_SOURCE_CONTRACT_DRIFT")
    require(p.read(assessment_dir / "season_boundary_evidence.json") ==
            q["assessment"]["season_evidence"], "PD_SEASON_EVIDENCE_DRIFT")
    require(p.read(assessment_dir / "team_mapping_evidence.json") == {
                "league": "PD", "mappings": q["assessment"]["team_mappings"],
                "new_mappings": q["assessment"]["mapping_evidence"],
                "fuzzy_matching_used": False}, "PD_MAPPING_EVIDENCE_DRIFT")
    overlap = q["assessment"]["cross_provider_overlap"]
    require(p.read(assessment_dir / "provider_overlap_evidence.json") == {
                "league": "PD", "football_data_co_uk_vs_fotmob": overlap,
                "rows": len(overlap), "conflicts": [],
                "required_fields": ["home_goals", "away_goals", "home_corners", "away_corners"]},
            "PD_PROVIDER_OVERLAP_EVIDENCE_DRIFT")

    retrieval_json = (PHASE12 / "PD/retrievals/ff957a9a5fcb424ab7d4ce2a3ffa7b53/matches.json")
    retrieval = p.read(retrieval_json)
    require(retrieval == p.read(PHASE12 / "PD/checkpoints/9cd860b35814fe92841e6d42e4fefdc9ec3015f388903f1acb746c7cb9af8de0.json")
            ["discovery"]["source"], "PD_RETRIEVAL_DISCOVERY_DRIFT")
    events = p.current_events(retrieval, pd[0], pd[2])
    retrieval_paths = {retrieval_json.relative_to(ROOT).as_posix(),
                       p.frozen.reference(retrieval["raw"]).relative_to(ROOT).as_posix()}
    require(len(retrieval_paths) == 2 and retrieval_paths <= candidates,
            "PD_RETRIEVAL_PACKAGE_INVENTORY_DRIFT")

    pd_checkpoint = "PD/checkpoints/9cd860b35814fe92841e6d42e4fefdc9ec3015f388903f1acb746c7cb9af8de0.json"
    bl1_checkpoint = "BL1/checkpoints/ccad36e37b1e0be5361ca42e98f8324d475863c583f07cbb6aebc5a2eff6b2fa.json"
    remaining = candidates - source_paths - retrieval_paths
    expected_remaining = {f"{boundary.PHASE12_RUN}/{name}" for name in
                          (pd_checkpoint, bl1_checkpoint, "PD/journal/00000117.json",
                           "PD/journal/00000118.json")}
    require(remaining == expected_remaining, "CANDIDATE_PACKAGE_COVERAGE_DRIFT")

    pair_index = p.read(ROOT / proposal["phase19_pair_index"]["path"])
    p.frozen.sealed(pair_index, "record_sha256")
    require(len(pair_index["pairs"]) == 1172 and
            all(row["pair_sha256"] != pair["pair_sha256"] and
                row["event_sha256"] != event118["record_sha256"]
                for row in pair_index["pairs"]), "PD_PHASE19_POPULATION_CONTAMINATION")
    require(proposal["phase19_pair_policy"] == "FROZEN_HEADS_ONLY", "PHASE19_POLICY_DRIFT")

    certificate = p.seal({
        "schema_version": "ffps.phase20_boundary_scientific_replay.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision_state": "REVIEW_ONLY", "approval_granted": False,
        "proposal_v6": refs_identity(PROPOSAL),
        "replay_script": refs_identity(Path(__file__)),
        "phase12_verifiers": [refs_identity(ROOT / name) for name in (
            "src/evaluation/phase12_prospective.py", "src/evaluation/phase12_evidence.py",
            "src/evaluation/phase11_evidence.py")],
        "packages": [
            {"name": "PD_SOURCE_ACQUISITION", "outcome": "REPLAY_VERIFIED",
             "candidate_files": len(source_paths), "source_replay": source_replay},
            {"name": "PD_RETRIEVAL_JOURNAL_CHECKPOINT", "outcome": "REPLAY_VERIFIED",
             "candidate_files": len(retrieval_paths) + 3,
             "normalized_retrieval_events": len(events), "cohort_pairs": len(pd[3]),
             "cohort_settlements": len(pd[4]), "journal_events": len(pd[6]),
             "checkpoints_verified": checkpoint_counts["PD"],
             "pair118_sha256": pair["pair_sha256"], "phase19_included": False},
            {"name": "BL1_CHECKPOINT", "outcome": "REPLAY_VERIFIED",
             "candidate_files": 1, "cohort_pairs": len(bl1[3]),
             "cohort_settlements": len(bl1[4]), "journal_events": len(bl1[6]),
             "checkpoints_verified": checkpoint_counts["BL1"]},
        ],
        "candidate_count": len(candidates),
        "candidate_outcomes": dict(Counter({"REPLAY_VERIFIED": len(candidates)})),
        "historical_quarantine_scientific_clearance": False,
        "terminated_incomplete_scientific_clearance": False,
        "phase19_pair_count": len(pair_index["pairs"]),
        "phase19_pair_policy": "FROZEN_HEADS_ONLY",
    })
    p.persist_new(DESTINATION, certificate)
    print(json.dumps({"certificate": refs_identity(DESTINATION),
                      "candidate_files": len(candidates),
                      "outcome": "REPLAY_VERIFIED"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

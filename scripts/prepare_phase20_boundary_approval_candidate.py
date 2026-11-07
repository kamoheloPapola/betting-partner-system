"""Prepare a sealed, still-PENDING boundary candidate from terminal replay.

This script never creates an APPROVED decision or an approval hash.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import phase12_prospective as p

SOURCE = ROOT / "docs/phase20-boundary-authorization-proposal-v6.json"
REPLAY = ROOT / "docs/phase20-boundary-scientific-replay-20260923.json"
DESTINATION = ROOT / "docs/phase20-boundary-approval-candidate-v7.json"


def main() -> None:
    source = p.read(SOURCE)
    replay = p.read(REPLAY)
    p.frozen.sealed(source, "record_sha256")
    p.frozen.sealed(replay, "record_sha256")
    if (source["decision_state"] != "PENDING" or
            replay["proposal_v6"] != p.identity(SOURCE) or
            replay["decision_state"] != "REVIEW_ONLY" or
            replay["approval_granted"] is not False or
            replay["candidate_count"] != 167 or
            replay["candidate_outcomes"] != {"REPLAY_VERIFIED": 167} or
            {item["name"]: (item["outcome"], item["candidate_files"])
             for item in replay["packages"]} != {
                 "PD_SOURCE_ACQUISITION": ("REPLAY_VERIFIED", 161),
                 "PD_RETRIEVAL_JOURNAL_CHECKPOINT": ("REPLAY_VERIFIED", 5),
                 "BL1_CHECKPOINT": ("REPLAY_VERIFIED", 1)}):
        raise ValueError("SCIENTIFIC_REPLAY_NOT_TERMINAL")
    candidate = {key: value for key, value in source.items() if key != "record_sha256"}
    candidate["decision_state"] = "PENDING"
    candidate["decision_reason"] = (
        "All 167 candidate files replayed as exact historical Phase 12 evidence; "
        "the scientific approval decision and external hash remain outstanding.")
    candidate["quarantined_may_coexist"] = True
    candidate["terminated_may_coexist"] = True
    candidate["scientific_replay_evidence"] = p.identity(REPLAY)
    for row in candidate["additions"]:
        if row["classification"] == "UNAUTHORIZED":
            row["classification"] = "AUTHORIZED_APPEND_ONLY_EVIDENCE"
            row["validation_profile"] = row["namespace"]
    counts = Counter(row["classification"] for row in candidate["additions"])
    if counts != Counter({"AUTHORIZED_APPEND_ONLY_EVIDENCE": 167,
                          "QUARANTINED_HISTORICAL_EVIDENCE": 12,
                          "TERMINATED_INCOMPLETE_EVIDENCE": 4}) or not all(
            row["scientific_clearance"] is False for row in candidate["additions"]):
        raise ValueError("CANDIDATE_CLASSIFICATION_DRIFT")
    p.persist_new(DESTINATION, p.seal(candidate))
    print(json.dumps({"candidate": p.identity(DESTINATION),
                      "decision_state": "PENDING", "counts": dict(counts),
                      "approval_granted": False}, sort_keys=True))


if __name__ == "__main__":
    main()

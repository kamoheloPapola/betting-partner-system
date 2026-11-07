"""Issue the user-authorized, exact v7 Phase 20 boundary approval artifact.

This never computes an expected approval hash or invokes the verifier.
The whole-file hash must be measured independently after publication.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import phase12_prospective as p

V7 = ROOT / "docs/phase20-boundary-approval-candidate-v7.json"
CERTIFICATE = ROOT / "docs/phase20-boundary-scientific-replay-20260923.json"
APPROVAL_BRIEF = Path(
    r"C:\Users\papol\.codex\attachments\92cd7181-ccf7-41b4-acad-892413e65ce0\Pasted text.txt")
DESTINATION = ROOT / "docs/phase20-boundary-approved-v1.json"
EXPECTED = {
    "candidate": "b8dddc14f72f8b26aa3a1a9fa9e54f86b8ee3f23705f90ac94f25aba559835a4",
    "certificate": "a16e8492fa243e344625a7396abdbc916b0c03c15789ae7fd5c35d2575c9b4f9",
    "approval_brief": "477cc983c709fcfe10f27b9f6e84993da31e121cc0e4dc7656a2ad99cfc327d4",
}


def main() -> None:
    if {"candidate": p.source.sha256_file(V7),
        "certificate": p.source.sha256_file(CERTIFICATE),
        "approval_brief": p.source.sha256_file(APPROVAL_BRIEF)} != EXPECTED:
        raise ValueError("EXACT_APPROVAL_INPUT_DRIFT")
    candidate = p.read(V7)
    replay = p.read(CERTIFICATE)
    p.frozen.sealed(candidate, "record_sha256")
    p.frozen.sealed(replay, "record_sha256")
    if (candidate["decision_state"] != "PENDING" or
            candidate["scientific_replay_evidence"] != p.identity(CERTIFICATE) or
            candidate["quarantined_may_coexist"] is not True or
            candidate["terminated_may_coexist"] is not True or
            candidate["frozen_prefix_policy"] != "EXACT_BYTES" or
            candidate["phase19_pair_policy"] != "FROZEN_HEADS_ONLY" or
            candidate["unlisted_additions_policy"] != "REJECT" or
            replay["candidate_outcomes"] != {"REPLAY_VERIFIED": 167} or
            replay["phase19_pair_count"] != 1172 or
            len(candidate["additions"]) != 183):
        raise ValueError("APPROVAL_INPUT_CONTRACT_DRIFT")
    counts = Counter(row["classification"] for row in candidate["additions"])
    if counts != Counter({"AUTHORIZED_APPEND_ONLY_EVIDENCE": 167,
                          "QUARANTINED_HISTORICAL_EVIDENCE": 12,
                          "TERMINATED_INCOMPLETE_EVIDENCE": 4}) or not all(
            row["scientific_clearance"] is False for row in candidate["additions"]):
        raise ValueError("APPROVAL_CLASSIFICATION_DRIFT")
    for row in candidate["additions"]:
        if p.identity(ROOT / row["path"]) != {key: row[key]
                                               for key in ("path", "sha256", "size_bytes")}:
            raise ValueError("APPROVAL_EVIDENCE_DRIFT:" + row["path"])
    approved = deepcopy({key: value for key, value in candidate.items()
                         if key != "record_sha256"})
    approved["decision_state"] = "APPROVED"
    approved["decision_reason"] = (
        "The user explicitly approved the exact replay-verified v7 boundary. "
        "Approval permits coexistence of only the 183 bound additions; "
        "historical scientific clearance remains false.")
    approved["approval_basis"] = {
        "reviewed_candidate": p.identity(V7),
        "scientific_replay": p.identity(CERTIFICATE),
        "user_approval_brief_sha256": EXPECTED["approval_brief"],
    }
    p.persist_new(DESTINATION, p.seal(approved))
    print(json.dumps({"approved_contract_path": DESTINATION.relative_to(ROOT).as_posix(),
                      "decision_state": "APPROVED", "additions": dict(counts),
                      "expected_hash_minted_by_verifier": False}, sort_keys=True))


if __name__ == "__main__":
    main()

"""Seal a non-authorizing Phase 20 boundary review proposal from one inventory.

The proposal classifies every captured addition but grants zero exceptions.
An approved authorization must be a new file with its own external hash anchor.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import phase12_prospective as p
from src.evaluation import phase20_boundary as boundary


INVENTORY = ROOT / "docs/phase20-protected-boundary-inventory-after-writer-freeze.json"
DESTINATION = ROOT / "docs/phase20-boundary-authorization-proposal-v6.json"
SOURCE_RUN = "baselines/phase11/phase12-acquisition-8e664a6cd54640d8bc0ad9eaa2a8b4ad"
REPORT33 = boundary.PHASE12_RUN + "/operational/report-000033"
REPORT34 = boundary.PHASE12_RUN + "/operational/report-000034"
FL1_INCIDENT = boundary.PHASE12_RUN + "/FL1/journal/00000102.json"
SA_CHECKPOINT = (boundary.PHASE12_RUN +
    "/SA/checkpoints/97da214474eb9f08045f7dcad7012dcbcf744d37b5d9e64312ad11e51a571dc5.json")


def main() -> None:
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if inventory["changed"] or inventory["missing"] or inventory["new_during_scan"] or inventory["removed_during_scan"]:
        raise ValueError("BOUNDARY_INVENTORY_NOT_STABLE")
    phase20_auth = p.read(boundary.RUN / "settlement_authorization.json")
    p.frozen.sealed(phase20_auth, "record_sha256")
    before = p.read(boundary.RUN / "protected_before.json")
    if len(before) != inventory["snapshot_protected_count"]:
        raise ValueError("BOUNDARY_SNAPSHOT_COUNT_MISMATCH")
    starts = {}
    for league in ("BL1", "SA", "FL1", "PD"):
        prefix = f"{boundary.PHASE12_RUN}/{league}/journal/"
        names = sorted(name for name in before if name.startswith(prefix) and name.endswith(".json"))
        if names:
            last = names[-1]
            event = p.read(ROOT / last)
            p.frozen.sealed(event, "record_sha256")
            if p.source.sha256_file(ROOT / last) != before[last]:
                raise ValueError("BOUNDARY_START_HEAD_CHANGED:" + league)
            starts[league] = {"sequence": event["sequence"], "record_sha256": event["record_sha256"]}
        else:
            starts[league] = {"sequence": 0, "record_sha256": None}
    rows = []
    for item in inventory["additions"]:
        name = item["path"]
        if name == FL1_INCIDENT or name.startswith(REPORT33 + "/"):
            classification = "QUARANTINED_HISTORICAL_EVIDENCE"
        elif name == SA_CHECKPOINT or name.startswith(REPORT34 + "/"):
            classification = "TERMINATED_INCOMPLETE_EVIDENCE"
        else:
            classification = "UNAUTHORIZED"  # Candidate evidence has no Phase 20 authorization.
        rows.append({"path": item["path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"],
                     "classification": classification,
                     "namespace": boundary._namespace(item["path"], boundary.PHASE12_RUN),
                     "scientific_clearance": False, "validation_profile": "PENDING_SCIENTIFIC_AUTHORIZATION"})
    if len(rows) != len({row["path"] for row in rows}) or len(rows) != inventory["added_at_start"]:
        raise ValueError("BOUNDARY_MANIFEST_INCOMPLETE")
    report33_paths = sorted(row["path"] for row in rows if row["path"].startswith(REPORT33 + "/"))
    report34_files = sorted(row["path"] for row in rows
                            if row["classification"] == "TERMINATED_INCOMPLETE_EVIDENCE")
    if len(report33_paths) != 11 or len(report34_files) != 4 or len(rows) != 183:
        raise ValueError("BOUNDARY_TERMINAL_INVENTORY_MISMATCH")
    proposal = p.seal({
        "schema_version": boundary.SCHEMA,
        "decision_state": "PENDING",
        "decision_reason": "Historical quarantine and timeout provenance are classified; replayable candidates lack scientific authorization.",
        "frozen_prefix_policy": "EXACT_BYTES",
        "unlisted_additions_policy": "REJECT",
        "phase19_pair_policy": "FROZEN_HEADS_ONLY",
        "quarantined_may_coexist": False,
        "terminated_may_coexist": False,
        "historical_quarantines": [{
            "incident": p.identity(ROOT / FL1_INCIDENT), "incident_code": "ARTIFACT_DRIFT",
            "report": p.identity(ROOT / REPORT33 / "report.json"),
            "failed_outcome": p.identity(ROOT / REPORT33 / "FL1/outcome.json"),
            "report_paths": report33_paths,
        }],
        "terminated_invocations": [{
            "report_number": "report-000034",
            "task_name": "\\FFPS Phase 12 Prospective Collection",
            "instance_id": "{6ec2d1e4-76cf-4241-897f-7fe9f860131e}",
            "engine_pid": 25984,
            "task_events": p.identity(ROOT / "docs/phase12-report34-scheduler-events.json"),
            "task_definition": p.identity(ROOT / "docs/phase12-report34-task-definition.xml"),
            "freeze_record": p.identity(ROOT / "docs/phase12-historical-writer-freeze-20260923.md"),
            "intent": p.identity(ROOT / REPORT34 / "intent.json"),
            "claim": p.identity(ROOT / REPORT34 / "SA/claim.json"),
            "outcome": p.identity(ROOT / REPORT34 / "SA/outcome.json"),
            "checkpoint": p.identity(ROOT / SA_CHECKPOINT),
            "files": report34_files,
        }],
        "phase20_authorization": p.identity(boundary.RUN / "settlement_authorization.json"),
        "protected_before": p.identity(boundary.RUN / "protected_before.json"),
        "inventory_evidence": p.identity(INVENTORY),
        "required_verifier": p.identity(ROOT / "src/evaluation/phase20_boundary.py"),
        "phase12_operations_configuration": p.identity(
            ROOT / boundary.PHASE12_RUN / "operational/configuration.json"),
        "phase19_heads": phase20_auth["phase19_heads"],
        "phase19_pair_counts": phase20_auth["pair_counts"],
        "phase19_pair_index": phase20_auth["phase19_pair_index"],
        "phase12_run_id": boundary.PHASE12_RUN,
        "phase12_run_specification": p.identity(ROOT / boundary.PHASE12_RUN / "run_specification.json"),
        "phase12_source_runs": [SOURCE_RUN],
        "starting_journal_heads": starts,
        "permitted_event_kinds": {"BL1": [], "SA": [], "FL1": ["incident"], "PD": ["incident", "pair"]},
        "permitted_sequences": {"BL1": [], "SA": [], "FL1": [102], "PD": [117, 118]},
        "permitted_schemas": {
            "source_acquisition": [None, "ffps.phase11_current_source_contract.v1",
                                   "ffps.phase11_fotmob_index.v1", "ffps.phase11_qualification.v1",
                                   "ffps.phase11_qualified_source_index.v1",
                                   "ffps.phase11_raw_source.v1", "ffps.phase11_source_index.v1",
                                   "ffps.phase12_source_refresh.v1"],
            "journal": [None], "checkpoint": [None],
            "retrieval": [None, "ffps.phase11_raw_source.v1"],
            "report": [None, "ffps.phase12_operational_report.v1"],
        },
        "additions": rows,
    })
    p.persist_new(DESTINATION, proposal)
    print(json.dumps({"proposal": p.identity(DESTINATION), "decision_state": "PENDING",
                      "added": len(rows), "authorization_granted": False}, sort_keys=True))


if __name__ == "__main__":
    main()

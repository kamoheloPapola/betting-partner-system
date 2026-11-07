"""Read-only classification aid for additions outside Phase 20's frozen snapshot.

This is an investigative inventory, not an integrity policy or authorization gate.
It never treats a valid seal or a known path as Phase 20 authorization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import phase12_prospective as p12
from src.evaluation import phase20_settlement as p20


RUN = p20.RUN
PHASE12 = "baselines/phase12/phase12-prospective-20260908-v1/"
ACQUISITION = re.compile(r"baselines/phase11/phase12-acquisition-[0-9a-f]{32}/")
SEALS = ("record_sha256", "index_sha256", "contract_sha256", "qualification_sha256")


def category(name: str) -> tuple[str, str, str]:
    """Return namespace, producer, and available *Phase 12* contract."""
    if ACQUISITION.match(name):
        return ("phase12_source_acquisition", "phase12_prospective.refresh / phase11_current_source",
                "Phase 12 source-refresh evidence; Phase 20 continuation is not declared")
    if name.startswith(PHASE12):
        tail = name[len(PHASE12):]
        if re.fullmatch(r"(?:BL1|SA|FL1|PD)/journal/\d{8}\.json", tail):
            return ("phase12_journal", "phase12_prospective.append",
                    "Phase 12 ordered continuation journal; Phase 20 continuation is not declared")
        if re.fullmatch(r"(?:BL1|SA|FL1|PD)/checkpoints/[0-9a-f]{64}\.json", tail):
            return ("phase12_checkpoint", "phase12_prospective.write_checkpoint",
                    "Phase 12 sealed checkpoint; Phase 20 continuation is not declared")
        if re.fullmatch(r"(?:BL1|SA|FL1|PD)/retrievals/[0-9a-f]{32}/matches\.(?:json|raw)", tail):
            return ("phase12_result_retrieval", "phase12_operations.retrieve",
                    "Phase 12 archived result response; Phase 20 continuation is not declared")
        if re.fullmatch(r"operational/report-\d{6}/.+\.json", tail):
            return ("phase12_operational_report", "phase12_operations.run / seal_report",
                    "Phase 12 scheduled-operation report; Phase 20 continuation is not declared")
    return ("unknown", "unattributed", "no identified append-only contract")


def references(value, found: list[dict]) -> None:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            found.append(value)
        for child in value.values():
            references(child, found)
    elif isinstance(value, list):
        for child in value:
            references(child, found)


def build() -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    before = p12.read(RUN / "protected_before.json")
    current = {}
    for base in ("baselines", "data", "src/ml/models"):
        for path in (ROOT / base).rglob("*"):
            if path.is_file() and not path.is_relative_to(RUN) and path.suffix not in (".pyc", ".lock", ".tmp"):
                current[path.relative_to(ROOT).as_posix()] = path
    added = sorted(set(current) - set(before))
    missing = sorted(set(before) - set(current))
    changed = sorted(name for name in set(before) & set(current)
                     if p12.source.sha256_file(current[name]) != before[name])

    all_rows = []
    incoming = defaultdict(list)
    journal_checks = {}
    for league in ("BL1", "SA", "FL1", "PD"):
        try:
            journal_checks[league] = {"valid": True, "count": len(p12.journal(ROOT / PHASE12, league))}
        except Exception as exc:
            journal_checks[league] = {"valid": False, "error": str(exc)}

    for name in added:
        path = current[name]
        namespace, producer, contract = category(name)
        row = {"path": name, "sha256": p12.source.sha256_file(path),
               "size_bytes": path.stat().st_size, "namespace": namespace,
               "creating_subsystem": producer, "contractual_basis": contract,
               "phase20_authorized": False, "checks": {}}
        if path.suffix == ".json":
            try:
                value = p12.read(path)
                if isinstance(value, dict):
                    for key in SEALS:
                        if key in value:
                            row["checks"][key] = (p12.digest({k: v for k, v in value.items() if k != key}) == value[key])
                    for key in ("schema", "schema_version", "status", "league", "sequence", "kind", "created_at"):
                        if key in value and not isinstance(value[key], (dict, list)):
                            row[key] = value[key]
                    if namespace == "phase12_journal":
                        row["event_code"] = value.get("payload", {}).get("code")
                        row["protocol_violation"] = value.get("payload", {}).get("protocol_violation")
                        row["source_ref_present"] = value.get("payload", {}).get("source") is not None
                        row["checks"]["league_chain_and_checkpoints"] = journal_checks.get(value.get("league"), {}).get("valid", False)
                    if name.endswith("/FL1/outcome.json"):
                        row["error"] = value.get("error")
                        row["replay"] = value.get("replay")
                    if name.endswith("/report.json"):
                        row["degraded"] = value.get("degraded")
                    refs = []
                    references(value, refs)
                    for ref in refs:
                        target = ref["path"]
                        if target in current:
                            target_path = current[target]
                            valid = (p12.source.sha256_file(target_path) == ref["sha256"] and
                                     ("size_bytes" not in ref or target_path.stat().st_size == ref["size_bytes"]))
                            incoming[target].append({"from": name, "valid": valid})
                    if path.name == "matches.json" and namespace == "phase12_result_retrieval":
                        row["retrieval_status"] = value.get("status")
            except Exception as exc:
                row["checks"]["json_parse"] = False
                row["error"] = str(exc)
        all_rows.append(row)

    rows_by_path = {r["path"]: r for r in all_rows}
    for row in all_rows:
        refs = incoming[row["path"]]
        row["incoming_references"] = len(refs)
        row["checks"]["incoming_references_valid"] = all(ref["valid"] for ref in refs)
        if row["namespace"] in ("phase12_source_acquisition", "phase12_result_retrieval") and row["path"].endswith(".raw"):
            metadata = rows_by_path.get(row["path"][:-4] + ".json")
            # Raw archives need a matching metadata identity, not only a matching filename.
            row["checks"]["raw_metadata_reference"] = bool(metadata and any(
                ref["from"] == metadata["path"] and ref["valid"] for ref in refs))
        if row["namespace"] == "unknown":
            row["evidence_state"] = "unattributed_namespace"
        elif row["path"].endswith("/FL1/journal/00000102.json"):
            row["evidence_state"] = "historical_protocol_violation_unresolved"
        elif row["path"].endswith("/report-000033/FL1/outcome.json"):
            row["evidence_state"] = "historical_failed_replay_unresolved"
        elif "/operational/report-000034/" in row["path"]:
            row["evidence_state"] = "unfinished_operational_report"
        elif any(value is False for value in row["checks"].values()):
            row["evidence_state"] = "structural_check_failed"
        else:
            row["evidence_state"] = "candidate_phase12_append_only_evidence"

    end_set = set()
    for base in ("baselines", "data", "src/ml/models"):
        for path in (ROOT / base).rglob("*"):
            if path.is_file() and not path.is_relative_to(RUN) and path.suffix not in (".pyc", ".lock", ".tmp"):
                end_set.add(path.relative_to(ROOT).as_posix())
    return {"schema": "investigative.phase20_boundary_inventory.v1",
            "started_at": started_at, "finished_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_protected_count": len(before), "changed": changed, "missing": missing,
            "added_at_start": len(added), "new_during_scan": sorted(end_set - set(current)),
            "removed_during_scan": sorted(set(current) - end_set),
            "journal_checks": journal_checks,
            "namespace_counts": dict(Counter(r["namespace"] for r in all_rows)),
            "evidence_state_counts": dict(Counter(r["evidence_state"] for r in all_rows)),
            "phase20_verified": False if (added or changed or missing or end_set - set(current)) else True,
            "authorization_note": "No added file is authorized by the frozen Phase 20 boundary. Candidate means only that a Phase 12 producer/contract and basic structure were identified.",
            "additions": all_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write inventory JSON outside protected baselines")
    args = parser.parse_args()
    result = build()
    if args.output:
        destination = args.output.resolve()
        if destination.is_relative_to(ROOT / "baselines") or destination.is_relative_to(ROOT / "data") or destination.is_relative_to(ROOT / "src/ml/models"):
            parser.error("output must be outside protected trees")
        destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "additions"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

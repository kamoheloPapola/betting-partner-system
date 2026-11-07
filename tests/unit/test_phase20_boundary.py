"""Fail-closed contract checks independent of the live settlement gate."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from src.evaluation import phase12_prospective as p
from src.evaluation import phase20_boundary as b


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path):
    root = tmp_path
    run = root / "baselines/phase20/phase20-controlled-settlement-20260923-v1"
    spec = root / b.PHASE12_RUN / "run_specification.json"
    _write(spec, p.seal({"schema_version": "test.phase12_run", "leagues": {}}))
    config = root / b.PHASE12_RUN / "operational/configuration.json"
    _write(config, p.seal({"implementation": []}))
    verifier = root / "src/evaluation/phase20_boundary.py"
    verifier.parent.mkdir(parents=True, exist_ok=True)
    verifier.write_bytes(b"test verifier identity")
    before = {path.relative_to(root).as_posix(): p.source.sha256_file(path) for path in (spec, config)}
    snapshot = run / "protected_before.json"
    _write(snapshot, before)
    heads = {league: f"phase19-{league}" for league in ("BL1", "SA", "FL1", "PD", "PL")}
    counts = {league: 1 for league in heads}
    pair_index = run / "phase19_pair_index.json"
    _write(pair_index, {"frozen_pairs": True})
    settlement = run / "settlement_authorization.json"
    _write(settlement, p.seal({"protected_before": b._identity(root, snapshot),
                               "phase19_heads": heads, "pair_counts": counts,
                               "phase19_pair_index": b._identity(root, pair_index),
                               "pair_creation_authorized": False}))
    inventory = root / "docs/inventory.json"
    _write(inventory, {"investigative_only": True})
    event_file = root / b.PHASE12_RUN / "BL1/journal/00000001.json"
    event = p.seal({"sequence": 1, "previous": None, "league": "BL1", "kind": "incident",
                    "created_at": "2026-09-23T00:00:00+00:00",
                    "payload": {"code": "SOURCE_UNAVAILABLE", "protocol_violation": False, "source": None}})
    _write(event_file, event)
    auth_path = root / "docs/authorization.json"
    contract = {"schema_version": b.SCHEMA, "decision_state": "APPROVED",
                "frozen_prefix_policy": "EXACT_BYTES", "unlisted_additions_policy": "REJECT",
                "phase19_pair_policy": "FROZEN_HEADS_ONLY", "quarantined_may_coexist": False,
                "terminated_may_coexist": False,
                "historical_quarantines": [], "terminated_invocations": [],
                "phase20_authorization": b._identity(root, settlement),
                "protected_before": b._identity(root, snapshot),
                "inventory_evidence": b._identity(root, inventory),
                "required_verifier": b._identity(root, verifier),
                "phase12_operations_configuration": b._identity(root, config),
                "phase19_heads": heads, "phase19_pair_counts": counts,
                "phase19_pair_index": b._identity(root, pair_index),
                "phase12_run_id": b.PHASE12_RUN,
                "phase12_run_specification": b._identity(root, spec),
                "phase12_source_runs": ["baselines/phase11/phase12-acquisition-" + "a" * 32],
                "starting_journal_heads": {league: {"sequence": 0, "record_sha256": None}
                                           for league in ("BL1", "SA", "FL1", "PD")},
                "permitted_event_kinds": {league: (["incident"] if league == "BL1" else [])
                                          for league in ("BL1", "SA", "FL1", "PD")},
                "permitted_sequences": {league: ([1] if league == "BL1" else [])
                                        for league in ("BL1", "SA", "FL1", "PD")},
                "permitted_schemas": {namespace: [None] for namespace in
                                      ("source_acquisition", "journal", "checkpoint", "retrieval", "report")},
                "additions": [{**b._identity(root, event_file),
                               "classification": "AUTHORIZED_APPEND_ONLY_EVIDENCE",
                               "namespace": "journal", "scientific_clearance": False,
                               "validation_profile": "journal"}]}
    _write(auth_path, p.seal(contract))
    return root, run, auth_path, contract, event_file, spec


def _check(root, run, auth_path):
    return b.verify(root, run, auth_path, approved_sha256=p.source.sha256_file(auth_path))


def _update_contract(path, contract):
    _write(path, p.seal(contract))


def _terminated_fixture(tmp_path):
    root, run, auth, contract, _, _ = _fixture(tmp_path)
    start = datetime(2026, 9, 23, 11, 50, 3, tzinfo=timezone.utc)
    end = start + timedelta(hours=6, seconds=5)
    task_name = "\\FFPS Phase 12 Prospective Collection"
    instance = "{test-phase12-instance}"
    action = str(root / "venv/Scripts/pythonw.exe")
    events = []
    for index, (event_id, when) in enumerate(((107, start), (100, start), (200, start),
                                               (201, end), (329, end)), 1):
        event = ET.Element("Event")
        system = ET.SubElement(event, "System")
        ET.SubElement(system, "Provider", Name="Microsoft-Windows-TaskScheduler")
        ET.SubElement(system, "EventID").text = str(event_id)
        ET.SubElement(system, "EventRecordID").text = str(index)
        ET.SubElement(system, "TimeCreated", SystemTime=when.isoformat())
        data = ET.SubElement(event, "EventData")
        values = {"TaskName": task_name,
                  "InstanceId" if event_id in (107, 100) else "TaskInstanceId": instance}
        if event_id in (200, 201):
            values.update(ActionName=action, EnginePID="25984")
        for key, value in values.items():
            ET.SubElement(data, "Data", Name=key).text = value
        events.append({"event_id": event_id, "record_id": index,
                       "time_utc": when.isoformat(), "provider": "Microsoft-Windows-TaskScheduler",
                       "message": "terminated due to exceeding the time allocated for execution" if event_id == 329 else "task event",
                       "xml": ET.tostring(event, encoding="unicode")})
    events_path = root / "docs/scheduler-events.json"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.write_text(json.dumps(events), encoding="utf-8")
    task = root / "docs/task-definition.xml"
    task.write_text('<Task><Settings><Enabled>false</Enabled><ExecutionTimeLimit>PT6H</ExecutionTimeLimit>'
                    '</Settings><Actions><Exec><Command>' + action + '</Command><Arguments>"' +
                    str(root / "scripts/run_phase12_operations.py") + '" run</Arguments><WorkingDirectory>' +
                    str(root) + '</WorkingDirectory></Exec></Actions></Task>', encoding="utf-8")
    freeze = root / "docs/freeze.txt"
    freeze.write_text("task terminated after six hours", encoding="utf-8")
    directory = root / b.PHASE12_RUN / "operational/report-000034"
    intent = directory / "intent.json"
    claim = directory / "SA/claim.json"
    outcome = directory / "SA/outcome.json"
    _write(intent, p.seal({"key": "20260923T11", "created_at": (start + timedelta(seconds=30)).isoformat(),
                           "previous_report": None}))
    claim_value = p.seal({"league": "SA", "as_of": (start + timedelta(hours=5, minutes=30)).isoformat(),
                          "journal_count": 0, "journal_head": None, "pairs": 0, "settled": 0})
    _write(claim, claim_value)
    checkpoint_value = p.seal({"league": "SA", "as_of": (start + timedelta(hours=5, minutes=45)).isoformat(),
                               "journal_count": 0, "journal_head": None, "pairs": 0, "settled": 0,
                               "replay": "VERIFIED", "scientific_state": "PROTOCOL_VIOLATION_BLOCKED"})
    checkpoint = root / b.PHASE12_RUN / "SA/checkpoints" / (checkpoint_value["record_sha256"] + ".json")
    _write(checkpoint, checkpoint_value)
    _write(outcome, p.seal({"league": "SA", "before": claim_value, "checkpoint": checkpoint_value,
                            "newly_admitted": 0, "newly_settled": 0, "new_incidents": [], "error": None}))
    files = [intent, claim, outcome, checkpoint]
    contract["additions"].extend({**b._identity(root, file),
                                  "classification": "TERMINATED_INCOMPLETE_EVIDENCE",
                                  "namespace": "checkpoint" if file == checkpoint else "report",
                                  "scientific_clearance": False,
                                  "validation_profile": "terminated_invocation"} for file in files)
    contract["terminated_invocations"] = [{"report_number": "report-000034",
        "task_name": task_name, "instance_id": instance, "engine_pid": 25984,
        "task_events": b._identity(root, events_path), "task_definition": b._identity(root, task),
        "freeze_record": b._identity(root, freeze), "intent": b._identity(root, intent),
        "claim": b._identity(root, claim), "outcome": b._identity(root, outcome),
        "checkpoint": b._identity(root, checkpoint),
        "files": sorted(file.relative_to(root).as_posix() for file in files)}]
    contract["terminated_may_coexist"] = True
    _update_contract(auth, contract)
    return root, run, auth, contract, files


def test_exact_approved_append_only_incident_verifies(tmp_path):
    root, run, auth, _, _, _ = _fixture(tmp_path)
    result = _check(root, run, auth)
    assert result["verified"] is True
    assert result["protected_existing"] == 2
    assert result["authorized_append_only_additions"] == 1
    assert b.verify(root, run, auth)["verified"] is False  # external approval hash is required


def test_wrong_hash_and_modified_approved_contract_fail(tmp_path):
    root, run, auth, contract, _, _ = _fixture(tmp_path)
    approved_hash = p.source.sha256_file(auth)
    assert b.verify(root, run, auth, approved_sha256="0" * 64)["verified"] is False
    contract["decision_reason"] = "modified after approval"
    _update_contract(auth, contract)
    assert b.verify(root, run, auth, approved_sha256=approved_hash)["verified"] is False


def test_authorized_addition_byte_change_fails(tmp_path):
    root, run, auth, _, event_file, _ = _fixture(tmp_path)
    approved_hash = p.source.sha256_file(auth)
    event_file.write_bytes(event_file.read_bytes() + b"\n")
    result = b.verify(root, run, auth, approved_sha256=approved_hash)
    assert result["verified"] is False
    assert any("BOUNDARY_ADDITION_DRIFT" in error for error in result["errors"])


def test_missing_approved_contract_fails(tmp_path):
    root, run, auth, _, _, _ = _fixture(tmp_path)
    approved_hash = p.source.sha256_file(auth)
    auth.unlink()
    result = b.verify(root, run, auth, approved_sha256=approved_hash)
    assert result["verified"] is False
    assert any("BOUNDARY_AUTHORIZATION_MISSING" in error for error in result["errors"])


def test_approved_verifier_source_drift_fails(tmp_path):
    root, run, auth, contract, _, _ = _fixture(tmp_path)
    verifier = root / contract["required_verifier"]["path"]
    verifier.write_bytes(b"changed verifier")
    result = _check(root, run, auth)
    assert result["verified"] is False
    assert any("BOUNDARY_REFERENCE_DRIFT" in error for error in result["errors"])


def test_missing_authorized_addition_fails(tmp_path):
    root, run, auth, _, event_file, _ = _fixture(tmp_path)
    event_file.unlink()
    assert _check(root, run, auth)["verified"] is False


def test_approved_pair_fails_when_frozen_scientific_replay_fails(tmp_path, monkeypatch):
    root, run, auth, contract, event_file, _ = _fixture(tmp_path)
    event = json.loads(event_file.read_text())
    event["kind"] = "pair"
    event["payload"] = {"fixture_id": "BL1_fixture", "pair_sha256": "pair-hash"}
    _write(event_file, p.seal({k: v for k, v in event.items() if k != "record_sha256"}))
    contract["additions"][0].update(b._identity(root, event_file))
    contract["permitted_event_kinds"]["BL1"] = ["pair"]
    _update_contract(auth, contract)
    monkeypatch.setattr(b, "ROOT", root)
    monkeypatch.setattr(p, "load_cohort", lambda *_: (_ for _ in ()).throw(
        ValueError("FROZEN_SCIENTIFIC_REPLAY_FAILED")))
    result = _check(root, run, auth)
    assert result["verified"] is False
    assert any("FROZEN_SCIENTIFIC_REPLAY_FAILED" in error for error in result["errors"])


@pytest.mark.parametrize("operation", ["change", "remove"])
def test_frozen_prefix_must_exist_with_exact_bytes(tmp_path, operation):
    root, run, auth, _, _, spec = _fixture(tmp_path)
    if operation == "change":
        spec.write_bytes(b"changed")
    else:
        spec.unlink()
    assert _check(root, run, auth)["verified"] is False


@pytest.mark.parametrize("name", ["baselines/phase12/unlisted.json", "baselines/other/unknown.json"])
def test_arbitrary_or_unknown_addition_fails(tmp_path, name):
    root, run, auth, _, _, _ = _fixture(tmp_path)
    _write(root / name, {"unlisted": True})
    result = _check(root, run, auth)
    assert result["verified"] is False
    assert result["unknown_additions"] == 1


def test_invalid_seal_rejected_even_when_file_hash_is_reauthorized(tmp_path):
    root, run, auth, contract, event_file, _ = _fixture(tmp_path)
    event = json.loads(event_file.read_text())
    event["payload"]["code"] = "CHANGED_WITHOUT_RESEAL"
    _write(event_file, event)
    contract["additions"][0].update(b._identity(root, event_file))
    _update_contract(auth, contract)
    assert any("BOUNDARY_SEAL_INVALID" in error for error in _check(root, run, auth)["errors"])


def test_broken_journal_chain_and_unexpected_kind_fail(tmp_path):
    root, run, auth, contract, event_file, _ = _fixture(tmp_path)
    event = json.loads(event_file.read_text())
    event["previous"] = "wrong"
    _write(event_file, p.seal({k: v for k, v in event.items() if k != "record_sha256"}))
    contract["additions"][0].update(b._identity(root, event_file))
    _update_contract(auth, contract)
    assert _check(root, run, auth)["verified"] is False
    event["previous"] = None
    event["kind"] = "unknown"
    _write(event_file, p.seal({k: v for k, v in event.items() if k != "record_sha256"}))
    contract["additions"][0].update(b._identity(root, event_file))
    _update_contract(auth, contract)
    assert _check(root, run, auth)["verified"] is False


def test_invalid_raw_reference_fails(tmp_path):
    root, run, auth, contract, _, _ = _fixture(tmp_path)
    source = root / contract["phase12_source_runs"][0] / "BL1/raw_sources/current/match-1.raw"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"actual")
    metadata = source.with_suffix(".json")
    _write(metadata, {"schema_version": "ffps.phase11_raw_source.v1", "provider": "fotmob:web-api",
                      "request": {"url": "https://www.fotmob.com/api/data/matchDetails",
                                  "params": {"matchId": "1"}},
                      "raw": {**b._identity(root, source), "sha256": "incorrect"}})
    for path in (source, metadata):
        contract["additions"].append({**b._identity(root, path), "classification": "UNAUTHORIZED",
                                      "namespace": "source_acquisition", "scientific_clearance": False,
                                      "validation_profile": "PENDING_SCIENTIFIC_AUTHORIZATION"})
    _update_contract(auth, contract)
    assert _check(root, run, auth)["verified"] is False


def test_source_scope_and_declared_sequence_must_match(tmp_path):
    root, run, auth, contract, _, _ = _fixture(tmp_path)
    contract["permitted_sequences"]["BL1"] = [2]
    _update_contract(auth, contract)
    assert any("SEQUENCE_NOT_PERMITTED" in error for error in _check(root, run, auth)["errors"])
    contract["permitted_sequences"]["BL1"] = [1]
    source = root / contract["phase12_source_runs"][0] / "BL1/raw_sources/current/match-1.raw"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source")
    metadata = source.with_suffix(".json")
    _write(metadata, {"schema_version": "ffps.phase11_raw_source.v1", "provider": "fotmob:web-api",
                      "request": {"url": "https://unexpected.example/api", "params": {}},
                      "raw": b._identity(root, source)})
    contract["permitted_schemas"]["source_acquisition"].append("ffps.phase11_raw_source.v1")
    for path in (source, metadata):
        contract["additions"].append({**b._identity(root, path), "classification": "UNAUTHORIZED",
                                      "namespace": "source_acquisition", "scientific_clearance": False,
                                      "validation_profile": "PENDING_SCIENTIFIC_AUTHORIZATION"})
    _update_contract(auth, contract)
    assert any("SOURCE_SCOPE" in error for error in _check(root, run, auth)["errors"])


def test_unfinished_report_never_counts_as_complete(tmp_path):
    root, run, auth, contract, _, _ = _fixture(tmp_path)
    path = root / b.PHASE12_RUN / "operational/report-000001/intent.json"
    _write(path, p.seal({"key": "test", "previous_report": None}))
    contract["additions"].append({**b._identity(root, path),
                                  "classification": "AUTHORIZED_APPEND_ONLY_EVIDENCE",
                                  "namespace": "report", "scientific_clearance": False,
                                  "validation_profile": "report"})
    _update_contract(auth, contract)
    result = _check(root, run, auth)
    assert result["verified"] is False
    assert any("INCOMPLETE" in error for error in result["errors"])


def test_quarantine_preserves_protocol_violation_and_requires_explicit_coexistence(tmp_path):
    root, run, auth, contract, event_file, _ = _fixture(tmp_path)
    event = json.loads(event_file.read_text())
    event["payload"].update(code="ARTIFACT_DRIFT", protocol_violation=True)
    _write(event_file, p.seal({k: v for k, v in event.items() if k != "record_sha256"}))
    contract["additions"][0].update(b._identity(root, event_file))
    contract["additions"][0].update(classification="QUARANTINED_HISTORICAL_EVIDENCE",
                                     validation_profile="historical_quarantine")
    contract["historical_quarantines"] = [{"incident": b._identity(root, event_file),
                                           "incident_code": "ARTIFACT_DRIFT",
                                           "report": None, "failed_outcome": None,
                                           "report_paths": []}]
    _update_contract(auth, contract)
    assert _check(root, run, auth)["verified"] is False
    contract["quarantined_may_coexist"] = True
    _update_contract(auth, contract)
    result = _check(root, run, auth)
    assert result["verified"] is True
    assert contract["additions"][0]["scientific_clearance"] is False


def test_candidate_inventory_never_grants_authorization_or_new_pair_population(tmp_path):
    root, run, auth, contract, _, _ = _fixture(tmp_path)
    contract["additions"][0].update(classification="UNAUTHORIZED",
                                     validation_profile="PENDING_SCIENTIFIC_AUTHORIZATION")
    _update_contract(auth, contract)
    result = _check(root, run, auth)
    assert result["verified"] is False
    assert result["unauthorized_additions"] == 1
    assert contract["phase19_heads"] == json.loads((run / "settlement_authorization.json").read_text())["phase19_heads"]
    contract["phase19_heads"]["BL1"] = "later-phase12-pair"
    _update_contract(auth, contract)
    assert _check(root, run, auth)["verified"] is False


def test_terminated_report_coexists_only_when_explicitly_approved(tmp_path):
    root, run, auth, contract, _ = _terminated_fixture(tmp_path)
    result = _check(root, run, auth)
    assert result["verified"] is True
    assert result["terminated_incomplete_evidence"] == 4
    assert all(row["scientific_clearance"] is False for row in contract["additions"])
    contract["terminated_may_coexist"] = False
    _update_contract(auth, contract)
    assert any("TERMINATION_BLOCKS" in error for error in _check(root, run, auth)["errors"])
    contract["terminated_may_coexist"] = True
    contract["decision_state"] = "PENDING"
    _update_contract(auth, contract)
    assert _check(root, run, auth)["verified"] is False


@pytest.mark.parametrize("mutation", ["wrong_instance", "wrong_timeout", "wrong_task_action"])
def test_terminated_report_requires_exact_invocation_provenance(tmp_path, mutation):
    root, run, auth, contract, _ = _terminated_fixture(tmp_path)
    group = contract["terminated_invocations"][0]
    if mutation == "wrong_instance":
        group["instance_id"] = "{other-instance}"
    elif mutation == "wrong_timeout":
        events = root / group["task_events"]["path"]
        values = json.loads(events.read_text())
        values[-1]["message"] = "task completed normally"
        events.write_text(json.dumps(values), encoding="utf-8")
        group["task_events"] = b._identity(root, events)
    else:
        task = root / group["task_definition"]["path"]
        task.write_text(task.read_text().replace("run_phase12_operations.py", "other.py"), encoding="utf-8")
        group["task_definition"] = b._identity(root, task)
    _update_contract(auth, contract)
    assert _check(root, run, auth)["verified"] is False


@pytest.mark.parametrize("mutation", ["final_report", "extra_file", "changed_outcome", "pair_delta",
                                       "settlement_delta"])
def test_terminated_report_cannot_masquerade_as_complete_or_add_pairs(tmp_path, mutation):
    root, run, auth, contract, files = _terminated_fixture(tmp_path)
    group = contract["terminated_invocations"][0]
    if mutation == "final_report":
        _write(files[0].parent / "report.json", p.seal({"degraded": False}))
    elif mutation == "extra_file":
        _write(files[0].parent / "unexpected.json", {"extra": True})
    else:
        outcome = files[2]
        value = json.loads(outcome.read_text())
        if mutation == "pair_delta":
            value["newly_admitted"] = 1
        elif mutation == "settlement_delta":
            value["newly_settled"] = 1
        else:
            value["error"] = "changed"
        _write(outcome, p.seal({k: v for k, v in value.items() if k != "record_sha256"}))
        updated = b._identity(root, outcome)
        group["outcome"] = updated
        next(row for row in contract["additions"] if row["path"] == updated["path"]).update(updated)
        _update_contract(auth, contract)
    assert _check(root, run, auth)["verified"] is False


def test_terminated_checkpoint_drift_and_later_journal_append_fail(tmp_path):
    root, run, auth, contract, files = _terminated_fixture(tmp_path)
    checkpoint = files[3]
    checkpoint.write_bytes(b"drift")
    assert _check(root, run, auth)["verified"] is False
    root, run, auth, contract, files = _terminated_fixture(tmp_path / "second")
    journal = root / b.PHASE12_RUN / "SA/journal/00000001.json"
    _write(journal, p.seal({"league": "SA", "sequence": 1, "previous": None,
                            "kind": "pair", "payload": {"fixture_id": "later"}}))
    assert _check(root, run, auth)["verified"] is False


def test_degraded_report_can_only_coexist_as_quarantined_history(tmp_path):
    root, run, auth, contract, _, _ = _fixture(tmp_path)
    incident_path = root / b.PHASE12_RUN / "FL1/journal/00000001.json"
    _write(incident_path, p.seal({"sequence": 1, "previous": None, "league": "FL1",
                                  "kind": "incident", "created_at": "2026-09-23T10:01:33+00:00",
                                  "payload": {"code": "ARTIFACT_DRIFT", "protocol_violation": True,
                                              "source": None}}))
    directory = root / b.PHASE12_RUN / "operational/report-000001"
    intent = directory / "intent.json"
    failed = directory / "FL1/outcome.json"
    report = directory / "report.json"
    _write(intent, p.seal({"key": "operation", "previous_report": None}))
    _write(failed, p.seal({"league": "FL1", "replay": "FAILED", "error": "SOURCE_BYTES_DRIFT"}))
    _write(report, p.seal({"schema": "ffps.phase12_operational_report.v1", "degraded": True,
                           "promotion_performed": False, "intent": b._identity(root, intent),
                           "artifacts": [b._identity(root, intent), b._identity(root, failed)]}))
    for file in (incident_path, intent, failed, report):
        contract["additions"].append({**b._identity(root, file),
                                      "classification": "QUARANTINED_HISTORICAL_EVIDENCE",
                                      "namespace": "journal" if file == incident_path else "report",
                                      "scientific_clearance": False,
                                      "validation_profile": "historical_quarantine"})
    contract["permitted_event_kinds"]["FL1"] = ["incident"]
    contract["permitted_sequences"]["FL1"] = [1]
    contract["permitted_schemas"]["report"].append("ffps.phase12_operational_report.v1")
    contract["historical_quarantines"] = [{"incident": b._identity(root, incident_path),
        "incident_code": "ARTIFACT_DRIFT", "report": b._identity(root, report),
        "failed_outcome": b._identity(root, failed),
        "report_paths": [file.relative_to(root).as_posix() for file in (intent, failed, report)]}]
    contract["quarantined_may_coexist"] = True
    _update_contract(auth, contract)
    result = _check(root, run, auth)
    assert result["verified"] is True
    assert result["quarantined_historical_evidence"] == 4
    assert all(row["scientific_clearance"] is False for row in contract["additions"])
    changed = json.loads(report.read_text())
    changed["degraded"] = False
    _write(report, p.seal({k: v for k, v in changed.items() if k != "record_sha256"}))
    identity = b._identity(root, report)
    contract["historical_quarantines"][0]["report"] = identity
    next(row for row in contract["additions"] if row["path"] == identity["path"]).update(identity)
    _update_contract(auth, contract)
    assert _check(root, run, auth)["verified"] is False

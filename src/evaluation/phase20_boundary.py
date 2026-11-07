"""Independent, fail-closed verifier for a *separate* Phase 20 boundary decision.

This module is not called by Phase 20 settlement. An authorization must be
approved outside the verifier and identified by an expected file SHA-256.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from src.evaluation import phase12_prospective as p


ROOT = p.ROOT
RUN = ROOT / "baselines/phase20/phase20-controlled-settlement-20260923-v1"
PHASE12_RUN = "baselines/phase12/phase12-prospective-20260908-v1"
SCHEMA = "ffps.phase20_boundary_authorization.v1"
CLASSES = ("AUTHORIZED_APPEND_ONLY_EVIDENCE", "QUARANTINED_HISTORICAL_EVIDENCE",
           "TERMINATED_INCOMPLETE_EVIDENCE", "UNFINISHED", "UNAUTHORIZED", "UNKNOWN")
BASES = ("baselines", "data", "src/ml/models")
SEALS = ("record_sha256", "index_sha256", "contract_sha256", "qualification_sha256")


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def _path(root: Path, name: str) -> Path:
    _require(isinstance(name, str) and name and "\\" not in name, "BOUNDARY_INVALID_PATH")
    path = (root / name).resolve()
    _require(path.is_relative_to(root.resolve()) and path != root.resolve(), "BOUNDARY_PATH_ESCAPE")
    return path


def _identity(root: Path, file: Path) -> dict:
    return {"path": file.relative_to(root).as_posix(), "sha256": p.source.sha256_file(file),
            "size_bytes": file.stat().st_size}


def _reference(root: Path, ref: dict) -> Path:
    _require(isinstance(ref, dict) and set(ref) == {"path", "sha256", "size_bytes"},
             "BOUNDARY_INVALID_REFERENCE")
    path = _path(root, ref["path"])
    _require(path.is_file(), "BOUNDARY_REFERENCE_MISSING:" + ref["path"])
    _require(_identity(root, path) == ref, "BOUNDARY_REFERENCE_DRIFT:" + ref["path"])
    return path


def _sealed(value: dict, key: str = "record_sha256") -> None:
    _require(isinstance(value, dict) and value.get(key) == p.digest({k: v for k, v in value.items() if k != key}),
             "BOUNDARY_SEAL_INVALID:" + key)


def _tracked_files(root: Path, run: Path) -> dict[str, Path]:
    current = {}
    for base in BASES:
        for path in (root / base).rglob("*"):
            if path.is_file() and not path.is_relative_to(run) and path.suffix not in (".pyc", ".lock", ".tmp"):
                current[path.relative_to(root).as_posix()] = path
    return current


def _namespace(name: str, run_id: str) -> str:
    if re.fullmatch(r"baselines/phase11/phase12-acquisition-[0-9a-f]{32}/(?:run_specification\.json|"
                    r"(?:BL1|SA|FL1|PD)/(?:raw_sources/current/.+\.(?:json|raw)|"
                    r"(?:source_index_current|fotmob_index_current)\.json|assessments/assessment-\d{3}/.+\.json))", name):
        return "source_acquisition"
    prefix = run_id + "/"
    if name.startswith(prefix):
        tail = name[len(prefix):]
        if re.fullmatch(r"(?:BL1|SA|FL1|PD)/journal/\d{8}\.json", tail):
            return "journal"
        if re.fullmatch(r"(?:BL1|SA|FL1|PD)/checkpoints/[0-9a-f]{64}\.json", tail):
            return "checkpoint"
        if re.fullmatch(r"(?:BL1|SA|FL1|PD)/retrievals/[0-9a-f]{32}/matches\.(?:json|raw)", tail):
            return "retrieval"
        if re.fullmatch(r"operational/report-\d{6}/.+\.json", tail):
            return "report"
    return "unknown"


def _refs(value, found: list[dict]) -> None:
    if isinstance(value, dict):
        if all(key in value for key in ("path", "sha256", "size_bytes")):
            found.append({key: value[key] for key in ("path", "sha256", "size_bytes")})
        for child in value.values():
            _refs(child, found)
    elif isinstance(value, list):
        for child in value:
            _refs(child, found)


def _source_scope(value: dict, league: str) -> None:
    """Check provider/request family; full source science uses Phase 12 replay."""
    provider = value.get("provider")
    request = value.get("request")
    _require(isinstance(request, dict) and isinstance(request.get("url"), str), "BOUNDARY_SOURCE_REQUEST")
    url = request["url"]
    params = request.get("params")
    _require(isinstance(params, dict), "BOUNDARY_SOURCE_PARAMS")
    if provider == "fotmob:web-api":
        _require(url in ("https://www.fotmob.com/api/data/leagues",
                         "https://www.fotmob.com/api/data/matchDetails"), "BOUNDARY_SOURCE_SCOPE")
        if url.endswith("/leagues"):
            spec = p.source.SOURCE_SPECS[league]
            _require(params == {"ccode3": spec["fotmob_country"], "id": spec["fotmob_id"],
                                "season": "2026/2027"}, "BOUNDARY_SOURCE_SCOPE")
        else:
            _require(set(params) == {"matchId"} and
                     re.fullmatch(r"[0-9]+", str(params["matchId"])) is not None,
                     "BOUNDARY_SOURCE_SCOPE")
    elif provider == "football-data.org:v4":
        _require(url in (f"https://api.football-data.org/v4/competitions/{league}/matches",
                         f"https://api.football-data.org/v4/competitions/{league}/teams") and
                 params == {"season": params.get("season")} and params.get("season") in (2025, 2026),
                 "BOUNDARY_SOURCE_SCOPE")
    elif provider == "football-data.co.uk:csv:no-www":
        division = p.source.SOURCE_SPECS[league]["division"]
        _require(url in (f"https://football-data.co.uk/mmz4281/2526/{division}.csv",
                         f"https://football-data.co.uk/mmz4281/2627/{division}.csv") and not params,
                 "BOUNDARY_SOURCE_SCOPE")
    else:
        raise ValueError("BOUNDARY_SOURCE_PROVIDER")


def _file_evidence(root: Path, name: str, namespace: str, run_id: str) -> dict:
    path = _path(root, name)
    if path.suffix == ".raw":
        metadata = path.with_suffix(".json")
        _require(metadata.is_file(), "BOUNDARY_RAW_METADATA_MISSING")
        value = p.read(metadata)
        _require(value.get("raw") == _identity(root, path), "BOUNDARY_RAW_REFERENCE")
        return {"raw_metadata": metadata.relative_to(root).as_posix()}
    _require(path.suffix == ".json", "BOUNDARY_UNEXPECTED_FILE_TYPE")
    value = p.read(path)
    _require(isinstance(value, dict), "BOUNDARY_JSON_OBJECT_REQUIRED")
    for key in SEALS:
        if key in value:
            _sealed(value, key)
    refs = []
    _refs(value, refs)
    for ref in refs:
        _reference(root, ref)
    if namespace == "source_acquisition":
        parts = name.split("/")
        league = parts[3] if len(parts) > 3 else None
        if value.get("schema_version") == "ffps.phase11_raw_source.v1":
            _require(league in ("BL1", "SA", "FL1", "PD"), "BOUNDARY_SOURCE_LEAGUE")
            _source_scope(value, league)
        elif path.name == "run_specification.json":
            _require(value.get("schema_version") == "ffps.phase12_source_refresh.v1" and
                     str(value.get("continuation", "")).replace("\\", "/") == run_id and
                     value.get("promotion_authorized") is False,
                     "BOUNDARY_SOURCE_RUN_SPEC")
    elif namespace == "journal":
        _require(value.get("kind") in ("pair", "settlement", "incident"), "BOUNDARY_EVENT_KIND")
        _require(value.get("league") == name.split("/")[-3], "BOUNDARY_EVENT_LEAGUE")
        _require(value.get("sequence") == int(path.stem), "BOUNDARY_EVENT_SEQUENCE")
    elif namespace == "checkpoint":
        _require(value.get("record_sha256") == path.stem, "BOUNDARY_CHECKPOINT_NAME")
    elif namespace == "retrieval" and path.name == "matches.json":
        _require(value.get("schema_version") == "ffps.phase11_raw_source.v1" and
                 value.get("provider") == "football-data.org:v4", "BOUNDARY_RESULT_SOURCE_SCHEMA")
        _source_scope(value, name.split("/")[3])
        _require(value["request"]["url"].endswith("/matches") and
                 value["request"]["params"] == {"season": 2026}, "BOUNDARY_RESULT_SOURCE_SCOPE")
    elif namespace == "report" and path.name == "report.json":
        _require(value.get("schema") == "ffps.phase12_operational_report.v1" and
                 value.get("promotion_performed") is False, "BOUNDARY_REPORT_SCHEMA")
    return {"schema": value.get("schema_version", value.get("schema")), "references": len(refs)}


def _journal_continuation(root: Path, rows: dict, run_id: str, starting_heads: dict) -> None:
    _require(set(starting_heads) == {"BL1", "SA", "FL1", "PD"}, "BOUNDARY_START_HEAD_INVENTORY")
    for league, head in starting_heads.items():
        _require(league in ("BL1", "SA", "FL1", "PD") and
                 isinstance(head, dict) and set(head) == {"sequence", "record_sha256"},
                 "BOUNDARY_START_HEAD_INVALID")
        sequence = head["sequence"]
        previous = head["record_sha256"]
        _require(isinstance(sequence, int) and sequence >= 0, "BOUNDARY_START_SEQUENCE")
        if sequence:
            predecessor = f"{run_id}/{league}/journal/{sequence:08d}.json"
            _require(predecessor not in rows, "BOUNDARY_START_HEAD_NOT_FROZEN")
            prior = p.read(_path(root, predecessor))
            _sealed(prior)
            _require(prior["record_sha256"] == previous, "BOUNDARY_START_HEAD_DRIFT")
        else:
            _require(previous is None, "BOUNDARY_START_HEAD_DRIFT")
        members = sorted((int(Path(name).stem), name) for name in rows
                         if re.fullmatch(re.escape(f"{run_id}/{league}/journal/") + r"\d{8}\.json", name))
        for number, name in members:
            _require(number == sequence + 1, "BOUNDARY_JOURNAL_GAP")
            event = p.read(_path(root, name))
            _sealed(event)
            _require(event.get("sequence") == number and event.get("previous") == previous and
                     event.get("league") == league, "BOUNDARY_JOURNAL_CHAIN")
            _require(event.get("kind") in ("pair", "settlement", "incident"), "BOUNDARY_EVENT_KIND")
            sequence, previous = number, event["record_sha256"]


def _report_lifecycle(root: Path, rows: dict, run_id: str) -> dict[str, bool]:
    reports = {}
    for name in rows:
        match = re.fullmatch(re.escape(run_id) + r"/operational/(report-\d{6})/.+\.json", name)
        if match:
            reports.setdefault(match.group(1), set()).add(name)
    complete = {}
    for number, files in reports.items():
        directory = f"{run_id}/operational/{number}"
        intent_path = _path(root, directory + "/intent.json")
        report_path = _path(root, directory + "/report.json")
        _require(intent_path.is_file(), "BOUNDARY_REPORT_INTENT_MISSING")
        intent = p.read(intent_path)
        _sealed(intent)
        complete[number] = report_path.is_file()
        if complete[number]:
            report = p.read(report_path)
            _sealed(report)
            _require(report.get("intent") == _identity(root, intent_path), "BOUNDARY_REPORT_INTENT_MISMATCH")
            previous = int(number.split("-")[1]) - 1
            previous_path = _path(root, f"{run_id}/operational/report-{previous:06d}/report.json") if previous else None
            expected_previous = _identity(root, previous_path) if previous_path and previous_path.is_file() else None
            _require(intent.get("previous_report") == expected_previous, "BOUNDARY_REPORT_CHAIN")
            indexed = {ref["path"] for ref in report.get("artifacts", [])
                       if isinstance(ref, dict) and ref.get("path", "").startswith(directory + "/")}
            actual = {item.relative_to(root).as_posix() for item in report_path.parent.rglob("*.json")
                      if item != report_path}
            _require(indexed == actual, "BOUNDARY_REPORT_INVENTORY")
        for name in files:
            if rows[name]["classification"] == "AUTHORIZED_APPEND_ONLY_EVIDENCE":
                _require(complete[number], "BOUNDARY_AUTHORIZED_REPORT_INCOMPLETE")
    return complete


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _scheduler_event(row: dict) -> dict:
    """Check a captured Windows event against its original XML payload."""
    event = ET.fromstring(row["xml"])
    system = event.find("{*}System")
    _require(system is not None, "BOUNDARY_SCHEDULER_XML")
    event_id = int(system.findtext("{*}EventID"))
    record_id = int(system.findtext("{*}EventRecordID"))
    provider = system.find("{*}Provider").attrib["Name"]
    timestamp = system.find("{*}TimeCreated").attrib["SystemTime"]
    _require(event_id == row["event_id"] and record_id == row["record_id"] and
             provider == row["provider"] and _utc(timestamp) == _utc(row["time_utc"]),
             "BOUNDARY_SCHEDULER_RECORD_DRIFT")
    data = {item.attrib.get("Name"): item.text for item in event.findall(".//{*}EventData/{*}Data")}
    return {"id": event_id, "time": _utc(timestamp), "data": data, "message": row["message"]}


def _verify_quarantine(root: Path, rows: dict, groups: list[dict]) -> None:
    _require(isinstance(groups, list), "BOUNDARY_QUARANTINE_GROUPS")
    covered = set()
    for group in groups:
        _require(set(group) == {"incident", "incident_code", "report", "failed_outcome", "report_paths"},
                 "BOUNDARY_QUARANTINE_GROUP_SCHEMA")
        incident_path = _reference(root, group["incident"])
        incident = p.read(incident_path)
        _sealed(incident)
        _require(incident.get("kind") == "incident" and
                 incident.get("payload", {}).get("protocol_violation") is True and
                 incident["payload"].get("code") == group["incident_code"],
                 "BOUNDARY_QUARANTINE_INCIDENT")
        covered.add(group["incident"]["path"])
        _require(group["incident"]["path"] in rows and
                 rows[group["incident"]["path"]]["classification"] == "QUARANTINED_HISTORICAL_EVIDENCE",
                 "BOUNDARY_QUARANTINE_INCIDENT_CLASS")
        if group["report"] is None:
            _require(group["failed_outcome"] is None and not group["report_paths"],
                     "BOUNDARY_QUARANTINE_REPORT_SCOPE")
            continue
        report_path = _reference(root, group["report"])
        failed_path = _reference(root, group["failed_outcome"])
        report = p.read(report_path)
        failed = p.read(failed_path)
        _sealed(report)
        _sealed(failed)
        _require(report.get("degraded") is True and
                 failed.get("league") == incident.get("league") and
                 failed.get("replay") == "FAILED" and failed.get("error") == "SOURCE_BYTES_DRIFT",
                 "BOUNDARY_QUARANTINE_DEGRADED_REPORT")
        prefix = report_path.parent.relative_to(root).as_posix() + "/"
        expected = {name for name in rows if name.startswith(prefix)}
        declared = set(group["report_paths"])
        _require(declared == expected and group["report"]["path"] in declared and
                 group["failed_outcome"]["path"] in declared,
                 "BOUNDARY_QUARANTINE_REPORT_INVENTORY")
        _require(all(rows[name]["classification"] == "QUARANTINED_HISTORICAL_EVIDENCE"
                     for name in declared), "BOUNDARY_QUARANTINE_REPORT_CLASS")
        covered.update(declared)
    _require(covered == {name for name, row in rows.items()
                         if row["classification"] == "QUARANTINED_HISTORICAL_EVIDENCE"},
             "BOUNDARY_UNGROUNDED_QUARANTINE")


def _verify_terminated(root: Path, rows: dict, groups: list[dict], run_id: str) -> None:
    _require(isinstance(groups, list), "BOUNDARY_TERMINATED_GROUPS")
    covered = set()
    for group in groups:
        _require(set(group) == {"report_number", "task_name", "instance_id", "engine_pid",
                                "task_events", "task_definition", "freeze_record", "intent", "claim",
                                "outcome", "checkpoint", "files"}, "BOUNDARY_TERMINATED_GROUP_SCHEMA")
        _reference(root, group["freeze_record"])
        events_path = _reference(root, group["task_events"])
        task_path = _reference(root, group["task_definition"])
        raw_events = p.read(events_path)
        _require(isinstance(raw_events, list), "BOUNDARY_SCHEDULER_EVENT_SET")
        events = [_scheduler_event(item) for item in raw_events]
        by_id = {item["id"]: item for item in events}
        _require(len(events) == 5 and set(by_id) == {100, 107, 200, 201, 329},
                 "BOUNDARY_SCHEDULER_EVENT_SET")
        for event in events:
            data = event["data"]
            _require(data.get("TaskName") == group["task_name"] and
                     data.get("InstanceId", data.get("TaskInstanceId")) == group["instance_id"],
                     "BOUNDARY_SCHEDULER_INVOCATION")
        _require(by_id[200]["data"].get("EnginePID") == str(group["engine_pid"]) ==
                 by_id[201]["data"].get("EnginePID") and
                 "exceeding the time allocated for execution" in by_id[329]["message"],
                 "BOUNDARY_SCHEDULER_TIMEOUT")
        started = by_id[100]["time"]
        ended = by_id[329]["time"]
        _require(0 <= (by_id[200]["time"] - started).total_seconds() <= 10 and
                 6 * 3600 <= (ended - started).total_seconds() <= 6 * 3600 + 30 and
                 by_id[201]["time"] <= ended,
                 "BOUNDARY_SCHEDULER_TIME_WINDOW")
        task_xml = ET.parse(task_path).getroot()
        def value(name):
            item = task_xml.find(".//{*}" + name)
            return item.text if item is not None else None
        _require(value("Command") == by_id[200]["data"].get("ActionName") and
                 value("Arguments") == '"' + str(root / "scripts/run_phase12_operations.py") + '" run' and
                 value("WorkingDirectory") == str(root) and value("ExecutionTimeLimit") == "PT6H" and
                 value("Enabled") == "false", "BOUNDARY_SCHEDULER_TASK_DEFINITION")
        directory = f"{run_id}/operational/{group['report_number']}"
        expected = {directory + "/intent.json", directory + "/SA/claim.json",
                    directory + "/SA/outcome.json", group["checkpoint"]["path"]}
        _require(set(group["files"]) == expected and group["report_number"] == "report-000034" and
                 all(name in rows and rows[name]["classification"] == "TERMINATED_INCOMPLETE_EVIDENCE"
                     for name in expected), "BOUNDARY_TERMINATED_FILE_SET")
        for key in ("intent", "claim", "outcome", "checkpoint"):
            _require(group[key]["path"] in expected, "BOUNDARY_TERMINATED_REFERENCE_SCOPE")
            _reference(root, group[key])
        intent = p.read(_path(root, group["intent"]["path"]))
        claim = p.read(_path(root, group["claim"]["path"]))
        outcome = p.read(_path(root, group["outcome"]["path"]))
        checkpoint = p.read(_path(root, group["checkpoint"]["path"]))
        for item in (intent, claim, outcome, checkpoint):
            _sealed(item)
        _require(intent.get("key") == "20260923T11" and
                 started <= _utc(intent["created_at"]) <= ended and
                 started <= _utc(claim["as_of"]) <= _utc(checkpoint["as_of"]) <= ended,
                 "BOUNDARY_TERMINATED_RECORD_TIME")
        _require(claim.get("league") == outcome.get("league") == checkpoint.get("league") == "SA" and
                 outcome.get("before") == claim and outcome.get("checkpoint") == checkpoint and
                 outcome.get("newly_admitted") == outcome.get("newly_settled") == 0 and
                 outcome.get("new_incidents") == [] and outcome.get("error") is None and
                 checkpoint.get("journal_count") == claim.get("journal_count") and
                 checkpoint.get("journal_head") == claim.get("journal_head") and
                 checkpoint.get("pairs") == claim.get("pairs") and
                 checkpoint.get("settled") == claim.get("settled"),
                 "BOUNDARY_TERMINATED_NO_PAIR_OR_SETTLEMENT")
        journal = root / run_id / "SA/journal"
        members = sorted(journal.glob("*.json"))
        _require(len(members) == claim["journal_count"] and
                 (p.read(members[-1])["record_sha256"] if members else None) == claim["journal_head"],
                 "BOUNDARY_TERMINATED_JOURNAL_TAIL")
        report_dir = root / directory
        actual = {item.relative_to(root).as_posix() for item in report_dir.rglob("*") if item.is_file()}
        _require(actual == expected - {group["checkpoint"]["path"]} and
                 not (report_dir / "report.json").exists(),
                 "BOUNDARY_TERMINATED_REPORT_COMPLETION")
        covered.update(expected)
    _require(covered == {name for name, row in rows.items()
                         if row["classification"] == "TERMINATED_INCOMPLETE_EVIDENCE"},
             "BOUNDARY_UNGROUNDED_TERMINATION")


def _scientific_replay(root: Path, row: dict, cache: dict) -> None:
    """Delegate prospective science to the frozen Phase 12 verifier."""
    name = row["path"]
    if row["namespace"] == "journal":
        event = p.read(_path(root, name))
        if event["kind"] == "incident":
            _require(event["payload"].get("protocol_violation") is False,
                     "BOUNDARY_PROTOCOL_VIOLATION_NOT_AUTHORIZABLE")
            return  # This authorizes recording an incident, never a pair or result.
    _require(root == ROOT.resolve(), "BOUNDARY_REPLAY_ROOT_UNSUPPORTED")
    if row["namespace"] == "source_acquisition":
        parts = name.split("/")
        league = parts[3] if len(parts) > 3 else p.read(_path(root, name)).get("league")
    elif row["namespace"] in ("journal", "checkpoint", "retrieval"):
        league = name.split("/")[3]
    else:
        league = None
    run = ROOT / PHASE12_RUN
    if league in ("BL1", "SA", "FL1", "PD") and league not in cache:
        cache[league] = p.load_cohort(run, league)
    if row["namespace"] == "checkpoint":
        from src.evaluation import phase12_evidence
        spec = p.specification(run)
        phase12_evidence.verify_checkpoint(p.read(_path(root, name)), cache[league],
                                           len(spec["leagues"][league]["prefix"]))
    elif row["namespace"] == "report":
        from src.evaluation import phase12_evidence
        number = re.search(r"/operational/(report-\d{6})/", name)
        _require(number is not None, "BOUNDARY_REPORT_PATH")
        directory = run / "operational" / number.group(1)
        report = p.read(directory / "report.json")
        _require(report.get("degraded") is False, "BOUNDARY_DEGRADED_REPORT_NOT_AUTHORIZABLE")
        _require([outcome.get("league") for outcome in report["leagues"]] == list(p.source.LEAGUES),
                 "BOUNDARY_REPORT_LEAGUES")
        spec = p.specification(run)
        for outcome in report["leagues"]:
            league = outcome["league"]
            _require("checkpoint" in outcome and outcome.get("error") is None,
                     "BOUNDARY_REPORT_FAILED_OUTCOME")
            _require(outcome == p.read(directory / league / "outcome.json"),
                     "BOUNDARY_REPORT_OUTCOME_MISMATCH")
            claim = p.read(directory / league / "claim.json")
            _sealed(claim)
            _require(outcome["before"] == claim, "BOUNDARY_REPORT_CLAIM_MISMATCH")
            if league not in cache:
                cache[league] = p.load_cohort(run, league)
            phase12_evidence.verify_checkpoint(outcome["checkpoint"], cache[league],
                                               len(spec["leagues"][league]["prefix"]))
            events = cache[league][-1]
            start = claim["journal_count"]
            end = outcome["checkpoint"]["journal_count"]
            _require(0 <= start <= end <= len(events) and
                     claim["journal_head"] == (events[start - 1]["record_sha256"] if start else None),
                     "BOUNDARY_REPORT_START_HEAD")
            _require(outcome["new_incidents"] == [event["payload"] for event in events[start:end]
                      if event["kind"] == "incident"], "BOUNDARY_REPORT_INCIDENTS")
            _require(outcome["newly_admitted"] == sum(event["kind"] == "pair" for event in events[start:end]) and
                     outcome["newly_settled"] == sum(event["kind"] == "settlement" for event in events[start:end]),
                     "BOUNDARY_REPORT_DELTAS")
            cohort = p.read(directory / league / "cohort_index.json")
            _require(cohort["ordered_journal"] == [p.identity(run / league / "journal" /
                     f"{number:08d}.json") for number in range(1, end + 1)],
                     "BOUNDARY_REPORT_COHORT_JOURNAL")


def verify(root: Path, run: Path, authorization_file: Path, *, approved_sha256: str | None = None,
           ) -> dict:
    """Verify one exact approved contract against live trees; never publish evidence.

    `approved_sha256` is an out-of-band trust anchor. A pending proposal or
    unanchored file can be inspected, but cannot return verified true.
    """
    root, run = root.resolve(), run.resolve()
    errors = []
    counts = Counter()
    result = {"verified": False, "protected_existing": 0, "changed": [], "missing": [],
              "authorized_append_only_additions": 0, "quarantined_historical_evidence": 0,
              "terminated_incomplete_evidence": 0, "unfinished_additions": 0,
              "unauthorized_additions": 0,
              "unknown_additions": 0, "errors": errors}
    try:
        _require(run.is_relative_to(root), "BOUNDARY_RUN_ESCAPE")
        authorization_file = authorization_file.resolve()
        _require(authorization_file.is_file(), "BOUNDARY_AUTHORIZATION_MISSING")
        contract = p.read(authorization_file)
        _sealed(contract)
        _require(contract.get("schema_version") == SCHEMA, "BOUNDARY_AUTHORIZATION_SCHEMA")
        _require(contract.get("phase12_run_id") == PHASE12_RUN, "BOUNDARY_RUN_ID")
        _require(contract.get("frozen_prefix_policy") == "EXACT_BYTES", "BOUNDARY_FROZEN_POLICY")
        _require(contract.get("unlisted_additions_policy") == "REJECT", "BOUNDARY_UNLISTED_POLICY")
        _require(contract.get("phase19_pair_policy") == "FROZEN_HEADS_ONLY", "BOUNDARY_PAIR_POLICY")
        _require(contract.get("quarantined_may_coexist") in (True, False), "BOUNDARY_QUARANTINE_POLICY")
        _require(contract.get("terminated_may_coexist") in (True, False), "BOUNDARY_TERMINATION_POLICY")
        _require(contract.get("decision_state") in ("PENDING", "APPROVED"), "BOUNDARY_DECISION_STATE")
        _reference(root, contract["inventory_evidence"])
        _require(contract["required_verifier"]["path"] == "src/evaluation/phase20_boundary.py",
                 "BOUNDARY_VERIFIER_SCOPE")
        _reference(root, contract["required_verifier"])
        config_ref = contract["phase12_operations_configuration"]
        _require(config_ref["path"] == PHASE12_RUN + "/operational/configuration.json",
                 "BOUNDARY_OPERATIONS_CONFIG_SCOPE")
        config = p.read(_reference(root, config_ref))
        _sealed(config)
        for source_ref in config.get("implementation", []):
            _reference(root, source_ref)
        _require(contract.get("phase12_source_runs") and
                 all(re.fullmatch(r"baselines/phase11/phase12-acquisition-[0-9a-f]{32}", x)
                     for x in contract["phase12_source_runs"]), "BOUNDARY_SOURCE_RUN_INVENTORY")
        _require(isinstance(contract.get("permitted_event_kinds"), dict) and
                 set(contract["permitted_event_kinds"]) == {"BL1", "SA", "FL1", "PD"},
                 "BOUNDARY_EVENT_RULES")
        _require(isinstance(contract.get("permitted_schemas"), dict) and
                 set(contract["permitted_schemas"]) == {"source_acquisition", "journal", "checkpoint",
                                                      "retrieval", "report"}, "BOUNDARY_SCHEMA_RULES")
        _require(isinstance(contract.get("permitted_sequences"), dict) and
                 set(contract["permitted_sequences"]) == {"BL1", "SA", "FL1", "PD"},
                 "BOUNDARY_SEQUENCE_RULES")
        settlement_auth = p.read(_reference(root, contract["phase20_authorization"]))
        _sealed(settlement_auth)
        _require(settlement_auth.get("protected_before") == contract["protected_before"] and
                 settlement_auth.get("phase19_heads") == contract["phase19_heads"] and
                 settlement_auth.get("pair_counts") == contract["phase19_pair_counts"] and
                 settlement_auth.get("phase19_pair_index") == contract["phase19_pair_index"] and
                 settlement_auth.get("pair_creation_authorized") is False,
                 "BOUNDARY_PHASE20_BINDING")
        _reference(root, contract["phase19_pair_index"])
        before = p.read(_reference(root, contract["protected_before"]))
        _require(isinstance(before, dict), "BOUNDARY_SNAPSHOT_SCHEMA")
        result["protected_existing"] = len(before)
        current = _tracked_files(root, run)
        result["missing"] = sorted(set(before) - set(current))
        result["changed"] = sorted(name for name in set(before) & set(current)
                                   if p.source.sha256_file(current[name]) != before[name])
        _require(contract["phase12_run_specification"]["path"] == PHASE12_RUN + "/run_specification.json",
                 "BOUNDARY_RUN_SPEC_SCOPE")
        run_spec = p.read(_reference(root, contract["phase12_run_specification"]))
        _sealed(run_spec)
        additions = set(current) - set(before)
        entries = contract.get("additions")
        _require(isinstance(entries, list) and all(isinstance(r, dict) for r in entries),
                 "BOUNDARY_MANIFEST_SCHEMA")
        rows = {}
        for row in entries:
            _require(set(row) == {"path", "sha256", "size_bytes", "classification", "namespace",
                                  "scientific_clearance", "validation_profile"}, "BOUNDARY_ENTRY_SCHEMA")
            name = row["path"]
            _require(name not in rows and name not in before, "BOUNDARY_DUPLICATE_OR_FROZEN_ENTRY")
            _path(root, name)
            _require(row["classification"] in CLASSES, "BOUNDARY_CLASSIFICATION")
            _require(row["namespace"] == _namespace(name, PHASE12_RUN), "BOUNDARY_NAMESPACE_MISMATCH")
            if row["namespace"] == "source_acquisition":
                _require(any(name.startswith(source_run + "/") for source_run in contract["phase12_source_runs"]),
                         "BOUNDARY_UNDECLARED_SOURCE_RUN")
            _require(row["scientific_clearance"] is False, "BOUNDARY_HISTORICAL_CLEARANCE_FORBIDDEN")
            rows[name] = row
        for name in sorted(additions - set(rows)):
            counts["UNKNOWN"] += 1
            errors.append("BOUNDARY_UNLISTED_ADDITION:" + name)
        for name, row in rows.items():
            classification = row["classification"]
            if name not in additions:
                errors.append("BOUNDARY_LISTED_ADDITION_MISSING:" + name)
                counts["UNKNOWN"] += 1
                continue
            counts[classification] += 1
            if _identity(root, current[name]) != {k: row[k] for k in ("path", "sha256", "size_bytes")}:
                errors.append("BOUNDARY_ADDITION_DRIFT:" + name)
                continue
            try:
                evidence = _file_evidence(root, name, row["namespace"], PHASE12_RUN)
                if row["namespace"] != "unknown":
                    _require(evidence.get("schema") in contract["permitted_schemas"][row["namespace"]],
                             "BOUNDARY_SCHEMA_NOT_PERMITTED")
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(str(exc) + ":" + name)
            if classification == "AUTHORIZED_APPEND_ONLY_EVIDENCE":
                _require(row["namespace"] != "unknown" and row["validation_profile"] == row["namespace"],
                         "BOUNDARY_AUTHORIZED_PROFILE")
        result["authorized_append_only_additions"] = counts["AUTHORIZED_APPEND_ONLY_EVIDENCE"]
        result["quarantined_historical_evidence"] = counts["QUARANTINED_HISTORICAL_EVIDENCE"]
        result["terminated_incomplete_evidence"] = counts["TERMINATED_INCOMPLETE_EVIDENCE"]
        result["unfinished_additions"] = counts["UNFINISHED"]
        result["unauthorized_additions"] = counts["UNAUTHORIZED"]
        result["unknown_additions"] = counts["UNKNOWN"]
        _journal_continuation(root, rows, PHASE12_RUN, contract["starting_journal_heads"])
        actual_sequences = {league: [] for league in ("BL1", "SA", "FL1", "PD")}
        for name, row in rows.items():
            if row["namespace"] == "journal":
                event = p.read(_path(root, name))
                _require(event["kind"] in contract["permitted_event_kinds"][event["league"]],
                         "BOUNDARY_EVENT_NOT_PERMITTED")
                actual_sequences[event["league"]].append(event["sequence"])
        _require({league: sorted(numbers) for league, numbers in actual_sequences.items()} ==
                 contract["permitted_sequences"], "BOUNDARY_SEQUENCE_NOT_PERMITTED")
        report_complete = _report_lifecycle(root, rows, PHASE12_RUN)
        _verify_quarantine(root, rows, contract["historical_quarantines"])
        _verify_terminated(root, rows, contract["terminated_invocations"], PHASE12_RUN)
        replay_cache = {}
        for name, row in rows.items():
            if row["classification"] == "UNFINISHED":
                number = re.search(r"/operational/(report-\d{6})/", name)
                _require(number is not None and report_complete.get(number.group(1)) is False,
                         "BOUNDARY_UNFINISHED_CLASSIFICATION_DRIFT")
            if row["classification"] == "AUTHORIZED_APPEND_ONLY_EVIDENCE":
                _require(contract["decision_state"] == "APPROVED", "BOUNDARY_NOT_APPROVED")
                _require(approved_sha256 is not None and
                         p.source.sha256_file(authorization_file) == approved_sha256,
                         "BOUNDARY_APPROVAL_HASH_REQUIRED")
                _scientific_replay(root, row, replay_cache)
        _require(not (result["missing"] or result["changed"]), "BOUNDARY_FROZEN_PREFIX_DRIFT")
        _require(not errors, "BOUNDARY_EVIDENCE_INVALID")
        _require(contract["decision_state"] == "APPROVED" and approved_sha256 is not None and
                 p.source.sha256_file(authorization_file) == approved_sha256,
                 "BOUNDARY_APPROVAL_REQUIRED")
        _require(not (counts["UNFINISHED"] or counts["UNAUTHORIZED"] or counts["UNKNOWN"]),
                 "BOUNDARY_BLOCKING_ADDITIONS")
        _require(not counts["QUARANTINED_HISTORICAL_EVIDENCE"] or contract["quarantined_may_coexist"],
                 "BOUNDARY_QUARANTINE_BLOCKS")
        _require(not counts["TERMINATED_INCOMPLETE_EVIDENCE"] or contract["terminated_may_coexist"],
                 "BOUNDARY_TERMINATION_BLOCKS")
        result["verified"] = True
    except (ValueError, KeyError, TypeError, OSError, ET.ParseError) as exc:
        errors.append(str(exc))
    return result

"""Offline connector mechanics; despite the filename, no live execution exists.

Fixture formats below are experimental test inputs, not an adopted diagnostic
grant schema, an issuer, or evidence of owner approval. All authority/time/state
adapters are synthetic. The only worker replaces HTTPAdapter.send with a local
response and denies sockets unconditionally. No credential loader exists.

The worker's file-read guard is a filename/extension denylist, not a complete
read allowlist or general credential-file isolation. The controlled executions
and environment-exclusion tests demonstrate only their exercised paths.

Selected source pins are integrity checks, not complete runtime qualification.
Consumption is thread-serialized in memory, NOT durable or cross-process safe.
Simulated supervisor loss is a failure model, NOT native containment evidence.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import threading
import time

EVIDENCE_ROOT = "tmp/BL1-live-execution-path-offline-implementation-20261026-01"
ERROR = "BL1_OFFLINE_CONNECTOR_INCOMPLETE"
AUTHORIZATION = {"mode": "synthetic", "network": False, "credentials": "generated_fixture_only"}
PROPOSAL = {"id": "BL1-DIAGNOSTIC-GRANT-CONTRACT-001", "version": "1.0.0-proposed-revision-1",
            "status": "PROPOSED_NOT_ADOPTED",
            "sha256": "211261474e7db30cd12db12b5b93a0800fd295b33c8c76d0e842e3925409d2f7"}
TARGET = {"provider": "football-data.org:v4", "method": "GET",
          "url": "https://api.football-data.org/v4/competitions/BL1/teams",
          "params": {"season": 2026}, "source_roles": ["primary-2026-teams"]}
LIMITS = {"max_attempts": 1, "max_requests": 1, "retries": 0, "redirects": 0,
          "elapsed_seconds": 30, "connect_timeout_seconds": 5, "read_timeout_seconds": 5,
          "max_response_bytes": 262144, "max_retained_bytes": 1048576}
CAPTURE_ALLOWANCE = 786432
NAME = "primary-2026-teams"
SELECTED_SOURCES = (
    "src/evaluation/bl1_live_acquisition.py", "src/evaluation/bl1_acquisition_gate.py",
    "src/evaluation/bl1_capture_transport.py", "src/evaluation/bl1_bounded_capture.py",
    "src/evaluation/future_lineage_context.py", "src/evaluation/future_lineage_qualification.py",
    "src/evaluation/phase11_current_source.py",
)
INVENTORY_MAX_BYTES = 1048576
INVENTORY_MAX_FILES = 4096
INVENTORY_FILE_BYTES = 134217728
INVENTORY_TOTAL_BYTES = 536870912
_CODE_SUFFIXES = {".py", ".pyc", ".pyd", ".dll", ".so", ".dylib", ".exe"}
_NATIVE_SUFFIXES = {".pyd", ".dll", ".so", ".dylib"}


def _require(condition, code="BL1_OFFLINE_FIXTURE_INVALID"):
    if not condition:
        raise ValueError(code)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json(body):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result)
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("BL1_OFFLINE_FIXTURE_INVALID")
    return json.loads(body, object_pairs_hook=pairs, parse_constant=invalid)


def _digest(body):
    return hashlib.sha256(body).hexdigest()


def _instant(value):
    _require(type(value) is str and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value))
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _authorize(value):
    _require(type(value) is dict and _canonical(value) == _canonical(AUTHORIZATION),
             "BL1_OFFLINE_ONLY")


def _safe(root, relative):
    _require(type(relative) is str and relative and "\\" not in relative and ":" not in relative
             and not relative.startswith("/") and len(relative) <= 512)
    parts = relative.split("/")
    _require(all(p and p not in {".", ".."} and p == p.rstrip(" .") and
        re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", p) is None for p in parts))
    path = root.joinpath(*parts)
    for item in (*reversed(root.parents), root, *[root.joinpath(*parts[:n]) for n in range(1, len(parts) + 1)]):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            _require(not item.is_symlink() and not
                getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)
    _require(path.resolve().is_relative_to(root.resolve()))
    return path


def _directory(root, relative):
    _require(type(relative) is str and relative.startswith(EVIDENCE_ROOT + "/"), "BL1_OFFLINE_OUTPUT_SCOPE")
    return _safe(root, relative)


def _runtime_roots(repository):
    # Hash only code/runtime files or disposable evidence fixtures. This is an
    # inventory-reader boundary, NOT a replacement for the worker read denylist.
    return tuple(Path(p).absolute() for p in (repository / "src", repository / EVIDENCE_ROOT,
        repository / "venv", sys.prefix, sys.base_prefix))


def _runtime_file(path, roots, checkpoint, totals):
    row = dict(path=path, status="OUT_OF_SCOPE", size_bytes=None, sha256=None)
    if type(path) is not str:
        return row
    target = Path(path)
    if (not target.is_absolute() or ".." in target.parts or target.suffix.lower() not in _CODE_SUFFIXES
            or target.name.lower().startswith(".env") or not any(target.is_relative_to(base) for base in roots)):
        return row
    checkpoint()
    try:
        checked = totals.setdefault("checked_ancestors", set())
        for ancestor in (*reversed(target.parents), target):
            if ancestor != target and ancestor in checked:
                continue
            info = ancestor.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                return {**row, "status": "REPARSE_POINT"}
            if ancestor != target:
                checked.add(ancestor)  # phase-local observation, not a continuous namespace guarantee
        with target.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                return {**row, "status": "NONREGULAR"}
            if before.st_size > INVENTORY_FILE_BYTES:
                return {**row, "status": "SIZE_LIMIT"}
            digest, count = hashlib.sha256(), 0
            while True:
                checkpoint()
                block = stream.read(1048576)
                if not block:
                    break
                count += len(block)
                totals["bytes"] += len(block)
                _require(count <= INVENTORY_FILE_BYTES and totals["bytes"] <= INVENTORY_TOTAL_BYTES,
                         "BL1_OFFLINE_INVENTORY_LIMIT")
                digest.update(block)
            after = os.fstat(stream.fileno())
        current = target.lstat()
        _require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                 (after.st_size, after.st_mtime_ns, after.st_ino) ==
                 (current.st_size, current.st_mtime_ns, current.st_ino) and count == before.st_size,
                 "BL1_OFFLINE_INVENTORY_CHANGED_DURING_READ")
        checkpoint()
        return dict(path=path, status="IDENTIFIED", size_bytes=count, sha256=digest.hexdigest())
    except FileNotFoundError:
        return {**row, "status": "MISSING"}
    except OSError:
        return {**row, "status": "UNREADABLE"}


def _collect_runtime_inventory(repository, module_snapshot, code_open_attempts, binding, checkpoint):
    """Metadata snapshot + observed code opens, not proof of loaded memory bytes.

    Built-in/frozen/no-file modules have no individual file digest. Native rows
    cover identified extension files, not transitive loaded DLLs. Audit open
    events precede IO: missing optional cache attempts are not successful reads.
    Callers in disposable tests supply artificial snapshots; workers use sys.modules.
    """
    _require(len(module_snapshot) <= INVENTORY_MAX_FILES, "BL1_OFFLINE_INVENTORY_LIMIT")
    roots, modules, paths = _runtime_roots(repository), [], set(code_open_attempts)
    for name, module in sorted(module_snapshot.items()):
        checkpoint()
        if module is None:
            modules.append(dict(name=name, kind="unavailable", path=None))
            continue
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        reported = getattr(module, "__file__", None)
        if origin in {"built-in", "frozen"}:
            kind, path = ("builtin" if origin == "built-in" else "frozen"), None
        elif type(reported) is str:
            path = str(Path(reported).absolute())
            kind = "extension" if Path(path).suffix.lower() in _NATIVE_SUFFIXES else "python"
            paths.add(path)
        else:
            kind, path = "no_file", None
        modules.append(dict(name=name, kind=kind, path=path))
    interpreters = [dict(role=role, path=str(Path(value).absolute()) if type(value) is str and value else None)
        for role, value in (("invoked_interpreter", sys.executable),
                            ("base_interpreter", getattr(sys, "_base_executable", None)))]
    paths.update(row["path"] for row in interpreters if row["path"] is not None)
    _require(len(paths) <= INVENTORY_MAX_FILES, "BL1_OFFLINE_INVENTORY_LIMIT")
    totals = {"bytes": 0}
    files = [_runtime_file(path, roots, checkpoint, totals) for path in sorted(paths)]
    record = dict(schema="bl1.offline_observed_runtime_inventory.v1", execution_mode="synthetic",
        runtime_closure="UNQUALIFIED", contract_status="PROPOSED_NOT_ADOPTED", binding=dict(binding),
        python=dict(version=sys.version, implementation=sys.implementation.name,
                    isolated=bool(sys.flags.isolated), no_site=bool(sys.flags.no_site),
                    dont_write_bytecode=bool(sys.dont_write_bytecode)),
        interpreters=interpreters, modules=modules, code_file_open_attempts=sorted(set(code_open_attempts)), files=files,
        native_components=sorted(row["path"] for row in files if row["status"] == "IDENTIFIED"
                                 and Path(row["path"]).suffix.lower() in _NATIVE_SUFFIXES),
        limitations=["Observed worker snapshot and pre-operation code-file open attempts only; instrumentation included.",
            "Built-in/frozen/no-file modules lack individual executing-byte identities.",
            "On-disk hashes do not prove loaded-memory identity or which bytecode cache executed.",
            "Transitive OS/native DLLs, unloaded modules, unexecuted branches and startup reads before the audit hook are not complete.",
            "No deployment trust, comprehensive file-read isolation or full T3 qualification."])
    payload = _canonical(record)
    _require(len(payload) <= INVENTORY_MAX_BYTES, "BL1_OFFLINE_INVENTORY_LIMIT")
    checkpoint()
    return payload


def verify_runtime_inventory(repository, payload, *, expected_sha256, expected_size_bytes, checkpoint=lambda: None):
    """Verify exact inventory input bytes and available file identities, offline only.

    The supplied pin is an evidence/test reference, never an owner authorization.
    Unavailable module/interpreter file identities reject. Optional open attempts
    can remain unavailable, with explicit states and no fabricated hashes. No
    filesystem/configuration changes or imports of inventoried code.
    """
    _require(type(payload) is bytes and len(payload) <= INVENTORY_MAX_BYTES and
             type(expected_size_bytes) is int and len(payload) == expected_size_bytes and
             type(expected_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
             and _digest(payload) == expected_sha256, "BL1_OFFLINE_INVENTORY_PIN")
    record = _json(payload)
    _require(type(record) is dict and set(record) == {"schema", "execution_mode", "runtime_closure", "contract_status",
        "binding", "python", "interpreters", "modules", "code_file_open_attempts", "files", "native_components", "limitations"})
    _require(record["schema"] == "bl1.offline_observed_runtime_inventory.v1" and
        record["execution_mode"] == "synthetic" and record["runtime_closure"] == "UNQUALIFIED" and
        record["contract_status"] == "PROPOSED_NOT_ADOPTED")
    python = record["python"]
    _require(type(python) is dict and set(python) == {"version", "implementation", "isolated", "no_site", "dont_write_bytecode"}
             and type(python["version"]) is str and type(python["implementation"]) is str
             and all(type(python[key]) is bool for key in ("isolated", "no_site", "dont_write_bytecode")))
    binding = record["binding"]
    _require(type(binding) is dict and set(binding) == {"operation", "directory", "receipt_sha256", "source_manifest_sha256"}
             and binding["operation"] in {"capture", "intake"})
    repository = Path(repository).absolute()
    _directory(repository, binding["directory"])
    for field in ("receipt_sha256", "source_manifest_sha256"):
        _require(type(binding[field]) is str and re.fullmatch(r"[0-9a-f]{64}", binding[field]))
    interpreters, modules, files, attempts = (record[key] for key in ("interpreters", "modules", "files", "code_file_open_attempts"))
    _require(type(interpreters) is list and len(interpreters) == 2 and all(type(row) is dict and set(row) == {"role", "path"}
             and (row["path"] is None or type(row["path"]) is str) for row in interpreters)
             and [row["role"] for row in interpreters] == ["invoked_interpreter", "base_interpreter"])
    _require(type(modules) is list and len(modules) <= INVENTORY_MAX_FILES and all(type(row) is dict and
        set(row) == {"name", "kind", "path"} and type(row["name"]) is str and type(row["kind"]) is str and row["kind"] in
        {"builtin", "frozen", "python", "extension", "no_file", "unavailable"} and
        ((type(row["path"]) is str) if row["kind"] in {"python", "extension"} else row["path"] is None) for row in modules))
    names = [row["name"] for row in modules]
    _require(names == sorted(set(names)))
    _require(type(attempts) is list and all(type(path) is str for path in attempts) and attempts == sorted(set(attempts)))
    required_paths = {row["path"] for row in modules + interpreters if row["path"] is not None}
    expected_paths = set(attempts) | required_paths
    _require(type(files) is list and len(files) <= INVENTORY_MAX_FILES and all(type(row) is dict and
        set(row) == {"path", "status", "size_bytes", "sha256"} and type(row["path"]) is str for row in files))
    _require([row["path"] for row in files] == sorted(expected_paths))
    roots, totals = _runtime_roots(repository), {"bytes": 0}
    for row in files:
        checkpoint()
        if row["status"] == "IDENTIFIED":
            _require(type(row["size_bytes"]) is int and row["size_bytes"] >= 0
                     and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]))
        else:
            _require(row["path"] not in required_paths and row["status"] in
                {"MISSING", "UNREADABLE", "NONREGULAR", "OUT_OF_SCOPE", "REPARSE_POINT", "SIZE_LIMIT"}
                and row["size_bytes"] is None and row["sha256"] is None)
        _require(_runtime_file(row["path"], roots, checkpoint, totals) == row,
                 "BL1_OFFLINE_INVENTORY_FILE_MISMATCH")
    _require(record["native_components"] == sorted(row["path"] for row in files
        if row["status"] == "IDENTIFIED" and Path(row["path"]).suffix.lower() in _NATIVE_SUFFIXES))
    _require(type(record["limitations"]) is list and len(record["limitations"]) >= 5
             and all(type(value) is str for value in record["limitations"]))
    checkpoint()
    return record


class SyntheticClock:
    """Explicit simulated time; never a trusted deployment clock."""
    def __init__(self, at, *, status="SYNTHETIC_OK", uncertainty_seconds=0):
        self.at = _instant(at)
        self.status, self.uncertainty_seconds = status, uncertainty_seconds

    def advance(self, seconds):
        self.at += timedelta(seconds=seconds)

    def read(self):
        _require(self.status == "SYNTHETIC_OK" and type(self.uncertainty_seconds) in (int, float)
                 and self.uncertainty_seconds == 0, "BL1_OFFLINE_CLOCK_UNVERIFIABLE")
        _require(type(self.at) is datetime and self.at.tzinfo == timezone.utc)
        return self.at


class SyntheticGrantStore:
    """In-memory fixture bytes and independently supplied TEST pins, not an owner store."""
    def __init__(self, records, pins):
        self.records = dict(records)
        self.pins = _json(_canonical(pins))

    def retrieve(self, grant_id):
        body, pin = self.records.get(grant_id), self.pins.get(grant_id)
        _require(type(body) is bytes and len(body) <= 16384 and type(pin) is dict)
        _require(set(pin) == {"fixture_only", "grant_id", "sha256", "size_bytes", "manifest_sha256"}
                 and pin["fixture_only"] is True and pin["grant_id"] == grant_id
                 and type(pin["size_bytes"]) is int and pin["size_bytes"] == len(body)
                 and pin["sha256"] == _digest(body))
        return body, dict(pin)


class SyntheticManifest:
    """Detached byte pin + selected source identities; closure remains UNQUALIFIED."""
    def __init__(self, body, *, sha256, size_bytes):
        self.body, self.sha256, self.size_bytes = body, sha256, size_bytes

    def verify(self, root):
        _require(type(self.body) is bytes and len(self.body) <= 32768 and
                 type(self.size_bytes) is int and len(self.body) == self.size_bytes and
                 _digest(self.body) == self.sha256)
        record = _json(self.body)
        _require(type(record) is dict and set(record) == {"schema", "mode", "proposal", "closure", "files"}
                 and record["schema"] == "bl1.offline_manifest_fixture.v1" and record["mode"] == "synthetic"
                 and record["proposal"] == PROPOSAL and record["closure"] == "SELECTED_SOURCES_ONLY")
        rows = record["files"]
        _require(type(rows) is list and len(rows) == len(SELECTED_SOURCES))
        _require(all(type(row) is dict and set(row) == {"path", "sha256", "size_bytes"} for row in rows))
        _require(sorted(row["path"] for row in rows) == sorted(SELECTED_SOURCES))
        for row in rows:
            path = _safe(root, row["path"])
            _require(path.is_file() and type(row["size_bytes"]) is int and row["size_bytes"] >= 0
                     and path.stat().st_size == row["size_bytes"])
            _require(_digest(path.read_bytes()) == row["sha256"])
        return record


class SyntheticConsumption:
    """Thread-only revocation/consumption model. No durability/cross-process claim."""
    def __init__(self):
        self.status = "NOT_REVOKED"
        self.fresh = True
        self.consumed = {}
        self.interrupt_after_consume = False
        self._lock = threading.Lock()

    def check(self):
        _require(self.fresh is True and self.status == "NOT_REVOKED", "BL1_OFFLINE_REVOKED_OR_UNKNOWN")

    def consume_once(self, grant_id, grant_digest, manifest_digest, scope):
        with self._lock:
            self.check()  # same synthetic critical section as the transition
            _require(grant_id not in self.consumed, "BL1_OFFLINE_ALREADY_CONSUMED")
            self.consumed[grant_id] = (grant_digest, manifest_digest, _digest(_canonical(scope)))
            if self.interrupt_after_consume:
                raise InterruptedError("BL1_OFFLINE_CONSUMPTION_AMBIGUOUS")


class SyntheticSupervisor:
    """Controlled worker or explicit fault model; no native parent-death protection."""
    def __init__(self, scenario="controlled_worker"):
        _require(scenario in {"controlled_worker", "launch_failure", "interrupted",
                             "supervisor_terminated_worker_survives"})
        self.scenario = scenario
        self.outcome = "NOT_RUN"

    def execute(self, job, remaining):
        if self.scenario != "controlled_worker":
            self.outcome = "SIMULATED_" + self.scenario.upper()
            raise InterruptedError("BL1_OFFLINE_SUPERVISOR_INCOMPLETE")
        # Only OS directory names are inherited. No secret environment value is read.
        env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
        env["PYTHON_DOTENV_DISABLED"] = "1"
        process = None
        try:
            process = subprocess.Popen([sys.executable, "-I", "-S", "-B", __file__, "--offline-worker"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            stdout, _ = process.communicate(_canonical(job), timeout=remaining())
            remaining()
            _require(process.returncode == 0 and len(stdout) <= 4194304)
            result = _json(stdout)
            _require(type(result) is dict and result.get("ok") is True)
            remaining()  # decoding/validation may itself cross the deadline
            self.outcome = "CONTROLLED_WORKER_RETURNED"
            return result
        except BaseException:
            self.outcome = "INCOMPLETE"
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                    process.communicate(timeout=1)
                except BaseException:
                    self.outcome = "WORKER_TERMINATION_UNRESOLVED"
        raise ValueError(ERROR)


class OfflineConnector:
    """One fixture attempt. No option, grant field or adapter can select live mode."""
    def __init__(self, repository, relative_directory, *, authorization, grants, clock,
                 consumption, manifest, supervisor):
        _authorize(authorization)  # before application imports or any output
        for value, expected in ((grants, SyntheticGrantStore), (clock, SyntheticClock),
                (consumption, SyntheticConsumption), (manifest, SyntheticManifest),
                (supervisor, SyntheticSupervisor)):
            _require(type(value) is expected, "BL1_OFFLINE_SYNTHETIC_ADAPTER_REQUIRED")
        self.root = Path(repository).absolute()
        self.relative_directory = relative_directory
        self.directory = _directory(self.root, relative_directory)
        _require(not self.directory.exists(), "BL1_OFFLINE_DESTINATION_EXISTS")
        _require(self.directory.parent.is_dir())
        self.grants, self.clock, self.consumption = grants, clock, consumption
        self.manifest, self.supervisor = manifest, supervisor
        self.state, self.attempts, self.request_slots, self.reserved_bytes = "NEW", 0, 0, 0
        self._verified_receipt = None
        self._owns_directory = False
        self.stage = "NOT_STARTED"
        self._lock = threading.Lock()

    def _verify(self, grant_id):
        body, pin = self.grants.retrieve(grant_id)
        grant = _json(body)
        _require(type(grant) is dict and set(grant) == {"schema", "mode", "grant_id", "owner", "proposal",
            "target", "output", "issued_at_utc_asserted", "expires_at_utc_exclusive", "manifest_sha256", "limits"})
        _require(grant["schema"] == "bl1.offline_grant_fixture.v1" and grant["mode"] == "synthetic"
                 and grant["owner"] == "SYNTHETIC_TEST_OWNER_NOT_AUTHENTICATED"
                 and grant["proposal"] == PROPOSAL and grant["grant_id"] == grant_id
                 and type(grant_id) is str and re.fullmatch(r"[A-Za-z0-9_-]{1,96}", grant_id)
                 and _canonical(grant["target"]) == _canonical(TARGET)
                 and grant["output"] == self.relative_directory)
        issued, expires = _instant(grant["issued_at_utc_asserted"]), _instant(grant["expires_at_utc_exclusive"])
        _require(expires - issued == timedelta(minutes=30) and issued <= self.clock.read() < expires,
                 "BL1_OFFLINE_GRANT_TIME")
        limits = grant["limits"]
        _require(type(limits) is dict and set(limits) == set(LIMITS))
        for key, cap in LIMITS.items():
            value = limits[key]
            _require(type(value) in (int, float) and math.isfinite(value))
            if key in {"max_attempts", "max_requests", "retries", "redirects"}:
                _require(type(value) is int and value == cap)
            elif key in {"max_response_bytes", "max_retained_bytes"}:
                _require(type(value) is int and 0 < value <= cap)
            else:
                _require(0 < value <= cap)
        self.manifest.verify(self.root)
        _require(grant["manifest_sha256"] == pin["manifest_sha256"] == self.manifest.sha256)
        if hasattr(self, "grant_digest"):
            _require(_digest(body) == self.grant_digest and self.manifest.sha256 == self.manifest_digest
                     and _digest(_canonical(pin)) == self.pin_digest, "BL1_OFFLINE_ADMISSION_DRIFT")
        self.consumption.check()
        return body, pin, grant

    def _remaining(self):
        left = self.deadline - time.monotonic()
        _require(left > 0, "BL1_OFFLINE_DEADLINE")
        return left

    def _persist(self, name, payload):
        _require(name in {"admission.json", "grant-fixture.json", "manifest-fixture.json", "pin-fixture.json",
                          "result.json", "failure.json"})
        _directory(self.root, self.relative_directory)
        with (self.directory / name).open("xb") as stream:
            _require(stream.write(payload) == len(payload))
            stream.flush()
            os.fsync(stream.fileno())

    def _retained_inputs(self):
        # Retained fixture copies are evidence, not authority. They must still
        # match the exact admitted bytes before intake/completion can succeed.
        for name, expected in self._retained_records.items():
            path = _safe(self.root, self.relative_directory + "/" + name)
            _require(path.stat().st_size == expected[0] and _digest(path.read_bytes()) == expected[1],
                     "BL1_OFFLINE_RETAINED_FIXTURE_DRIFT")

    def _result(self, result, *, operation, observe_runtime=False):
        receipt = result.get("receipt")
        _require(type(receipt) is dict and type(receipt.get("record_sha256")) is str
                 and re.fullmatch(r"[0-9a-f]{64}", receipt["record_sha256"]))
        path = _safe(self.root, self.relative_directory + "/" + NAME + ".transport.json")
        _require(_json(path.read_bytes()) == receipt)
        _require(("runtime_inventory" in result) is observe_runtime)
        if observe_runtime:
            envelope = result["runtime_inventory"]
            _require(type(envelope) is dict and set(envelope) == {"base64", "sha256", "size_bytes"})
            payload = base64.b64decode(envelope["base64"], validate=True)
            inventory = verify_runtime_inventory(self.root, payload, expected_sha256=envelope["sha256"],
                expected_size_bytes=envelope["size_bytes"], checkpoint=self._remaining)
            _require(inventory["binding"] == {"operation": operation, "directory": self.relative_directory,
                "receipt_sha256": receipt["record_sha256"], "source_manifest_sha256": self.manifest_digest})
        return {**result, "execution_mode": "synthetic", "status": "PROVISIONAL",
                "human_acceptance": "NOT_PERFORMED", "real_execution_available": False,
                "runtime_closure": "UNQUALIFIED", "durable_cross_process_consumption": "UNQUALIFIED",
                "native_supervisor_death_safety": "UNQUALIFIED",
                "budget": {"attempts": self.attempts, "request_slots": self.request_slots,
                           "reserved_bytes": self.reserved_bytes}}

    def _fail(self):
        self.state, self._verified_receipt = "INCOMPLETE", None
        self.failure_evidence = "UNAVAILABLE"
        # Only fixed stages/outcomes and validated fixture-artifact digests;
        # exception messages, tracebacks and credential hashes are never retained.
        def digest_or_none(value):
            return value if type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) else None
        outcome = self.supervisor.outcome
        if outcome not in {"NOT_RUN", "INCOMPLETE", "CONTROLLED_WORKER_RETURNED", "WORKER_TERMINATION_UNRESOLVED",
                "SIMULATED_LAUNCH_FAILURE", "SIMULATED_INTERRUPTED", "SIMULATED_SUPERVISOR_TERMINATED_WORKER_SURVIVES"}:
            outcome = "UNRECOGNIZED"
        self.failure_detail = {"status": "INCOMPLETE", "code": ERROR, "execution_mode": "synthetic",
            "human_acceptance": "NOT_PERFORMED", "stage": self.stage,
            "grant_sha256": digest_or_none(getattr(self, "grant_digest", None)),
            "manifest_sha256": digest_or_none(self.manifest.sha256),
            "supervisor_outcome": outcome}
        try:
            if self._owns_directory and self.directory.is_dir() and not (self.directory / "failure.json").exists():
                payload = _canonical(self.failure_detail)
                _require(len(payload) <= 512)
                self._persist("failure.json", payload)
                self.failure_evidence = "WRITTEN"
        except BaseException:
            pass

    def run_fixture(self, grant_id, body, *, names=None, headers=None, status=200,
                    read_delay=0, fail_send=False, observe_runtime=False):
        with self._lock:
            _require(self.state == "NEW", "BL1_OFFLINE_SESSION_NOT_REUSABLE")
            self.state, self.attempts = "ADMITTING", 1
            self.stage = "AUTHORITY"
            started = time.monotonic()
            self.deadline = started + LIMITS["elapsed_seconds"]
            try:
                original, pin, grant = self._verify(grant_id)
                self.grant_id, self.grant_digest = grant_id, _digest(original)
                self.manifest_digest, self.pin_digest = self.manifest.sha256, _digest(_canonical(pin))
                self.deadline = min(started + grant["limits"]["elapsed_seconds"], time.monotonic() +
                    (_instant(grant["expires_at_utc_exclusive"]) - self.clock.read()).total_seconds())
                _require(type(body) is bytes and len(body) <= LIMITS["max_response_bytes"] + 1)
                _require(type(status) is int and type(fail_send) is bool and
                    type(read_delay) in (int, float) and math.isfinite(read_delay) and read_delay >= 0)
                _require(type(observe_runtime) is bool)
                _require(len(_canonical([names, headers])) <= 32768)
                self.stage = "RESERVATION"
                admission = _canonical({"mode": "synthetic", "attempts": 1, "request_slots": 1,
                    "capture_allowance": CAPTURE_ALLOWANCE, "grant_sha256": self.grant_digest})
                # Full possible capture names (including staging/incomplete), exact
                # fixture bytes, admission and 1024 bytes for result + failure.
                self.reserved_bytes = CAPTURE_ALLOWANCE + len(original) + len(self.manifest.body) + len(_canonical(pin)) + len(admission) + 1024
                _require(self.reserved_bytes <= grant["limits"]["max_retained_bytes"], "BL1_OFFLINE_STORAGE_LIMIT")
                self._remaining()
                self.stage = "CONSUMPTION"
                # Charge before consumption can commit and then raise. This is
                # a pessimistic slot, never evidence of a provider request.
                self.request_slots = 1
                self.consumption.consume_once(grant_id, self.grant_digest, self.manifest.sha256,
                    {"target": grant["target"], "output": grant["output"]})
                self.stage = "ADMISSION_PERSISTENCE"
                self.directory.mkdir()  # exclusive; collisions never authorize reuse
                self._owns_directory = True
                records = {"admission.json": admission, "grant-fixture.json": original,
                           "manifest-fixture.json": self.manifest.body, "pin-fixture.json": _canonical(pin)}
                self._retained_records = {name: (len(payload), _digest(payload)) for name, payload in records.items()}
                self._persist("admission.json", admission)
                for name, payload in (("grant-fixture.json", original), ("manifest-fixture.json", self.manifest.body),
                                      ("pin-fixture.json", _canonical(pin))):
                    self._remaining()
                    self._persist(name, payload)
                self._verify(grant_id)
                self._retained_inputs()
                job = self._job("capture", names=names, observe_runtime=observe_runtime)
                job.update(body=base64.b64encode(body).decode("ascii"), headers=headers, status=status,
                           read_delay=read_delay, fail_send=fail_send)
                self.stage = "WORKER"
                result = self._result(self.supervisor.execute(job, self._remaining), operation="capture", observe_runtime=observe_runtime)
                completion = _canonical({"status": "PROVISIONAL", "execution_mode": "synthetic",
                    "receipt_sha256": result["receipt"]["record_sha256"], "human_acceptance": "NOT_PERFORMED"})
                _require(len(completion) <= 512)
                self._remaining()
                self.stage = "COMPLETION_PERSISTENCE"
                self._persist("result.json", completion)
                self.stage = "FINAL_VERIFICATION"
                self._verify(grant_id)
                self._retained_inputs()
                self._remaining()  # after persistence AND verification; never durable PASS
                self._verified_receipt = result["receipt"]["record_sha256"]
                self.state = "PROVISIONAL"
                return result
            except BaseException:
                self._fail()
        raise ValueError(ERROR)

    def _job(self, operation, *, names, observe_runtime=False):
        return {"authorization": dict(AUTHORIZATION), "operation": operation, "repository": str(self.root),
                "directory": self.relative_directory, "deadline": self.deadline, "allowance": CAPTURE_ALLOWANCE,
                "limits": _json(self.grants.retrieve(self.grant_id)[0])["limits"], "names": names,
                "manifest": base64.b64encode(self.manifest.body).decode("ascii"),
                "manifest_sha256": self.manifest.sha256, "manifest_size_bytes": self.manifest.size_bytes,
                "verified_receipt": self._verified_receipt, "observe_runtime": observe_runtime}

    def replay_fixture(self, *, names=None, observe_runtime=False):
        with self._lock:
            _require(self.state == "PROVISIONAL" and self._verified_receipt is not None,
                     "BL1_OFFLINE_SESSION_NOT_REUSABLE")
            try:
                self.stage = "REPLAY_AUTHORITY"
                _require(type(observe_runtime) is bool)
                self._remaining()
                self._verify(self.grant_id)
                self._retained_inputs()
                self.stage = "REPLAY_WORKER"
                result = self._result(self.supervisor.execute(self._job("intake", names=names, observe_runtime=observe_runtime),
                    self._remaining), operation="intake", observe_runtime=observe_runtime)
                _require(result["receipt"]["record_sha256"] == self._verified_receipt)
                self.stage = "REPLAY_FINAL_VERIFICATION"
                self._verify(self.grant_id)
                self._retained_inputs()
                self._remaining()
                return result
            except BaseException:
                self._fail()
        raise ValueError(ERROR)


def _worker(job):
    _authorize(job.get("authorization"))  # before project/third-party imports
    _require(job.get("operation") in {"capture", "intake"})
    observe_runtime = job.get("observe_runtime", False)
    _require(type(observe_runtime) is bool)
    root = Path(job["repository"]).absolute()
    directory = _directory(root, job["directory"])
    manifest = SyntheticManifest(base64.b64decode(job["manifest"], validate=True),
        sha256=job["manifest_sha256"], size_bytes=job["manifest_size_bytes"])
    manifest.verify(root)
    _require(directory.is_dir() and time.monotonic() < job["deadline"])
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root), str(root / "venv/Lib/site-packages")]
    files = {directory / (NAME + suffix) for suffix in (".raw", ".json", ".transport.json")}
    files |= {directory / (NAME + ".pending") / "response.raw",
              directory / (NAME + ".pending") / "metadata.json",
              directory / (NAME + ".transport.pending") / "receipt.json"}
    dirs = {directory / (NAME + ".pending"), directory / (NAME + ".transport.pending")}
    observed_code_opens = set()
    runtime_roots = _runtime_roots(root)

    def audit(event, args):
        if event.startswith("socket.") or event in {"subprocess.Popen", "os.system", "os.startfile", "os.posix_spawn", "os.fork"}:
            raise PermissionError("BL1_OFFLINE_EFFECT_PROHIBITED")
        if event == "open":
            path, mode, flags = args
            _require(not isinstance(path, int))
            path = Path(path).absolute()
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                _require(path in files and job["operation"] == "capture" and time.monotonic() < job["deadline"])
            else:
                # Limited name-based read exclusions only. Other names, such as
                # secrets.json, credentials.json, *.key or *.pem, are not denied
                # by this predicate alone; no general credential-read claim.
                _require(not path.name.lower().startswith(".env") and path.name.lower() != "credentials"
                         and path.suffix.lower() not in {".pkl", ".pickle", ".joblib", ".onnx"})
                if observe_runtime and path.suffix.lower() in _CODE_SUFFIXES and any(path.is_relative_to(base) for base in runtime_roots):
                    observed_code_opens.add(str(path))
                    _require(len(observed_code_opens) <= INVENTORY_MAX_FILES, "BL1_OFFLINE_INVENTORY_LIMIT")
        if event in {"os.mkdir", "os.rmdir"}:
            _require(Path(args[0]).absolute() in dirs)
        if event == "os.remove":
            _require(Path(args[0]).absolute() in files)
        if event in {"os.rename", "os.link"}:
            _require(all(Path(p).absolute() in files for p in args[:2]) and time.monotonic() < job["deadline"])
        if event in {"os.symlink", "os.chmod", "os.utime", "os.truncate"}:
            raise PermissionError("BL1_OFFLINE_EFFECT_PROHIBITED")

    sys.addaudithook(audit)
    import socket
    socket.has_ipv6 = False  # avoid urllib3's import-time connectivity probe
    import io
    from unittest.mock import patch
    import requests
    from urllib3._collections import HTTPHeaderDict
    from urllib3.response import HTTPResponse
    from src.evaluation import bl1_capture_transport as transport
    from src.evaluation.bl1_acquisition_gate import _verified_teams, _WorkerBudget
    budget = _WorkerBudget(job["deadline"], job["allowance"])
    if job["operation"] == "intake":
        result = _json((directory / "result.json").read_bytes())
        _require(not (directory / "failure.json").exists() and result["status"] == "PROVISIONAL"
                 and result["receipt_sha256"] == job["verified_receipt"])
    else:
        body = base64.b64decode(job["body"], validate=True)
        credential = "BL1_SYNTHETIC_FIXTURE_TOKEN_NEVER_A_REAL_CREDENTIAL"

        class LocalStream(io.BytesIO):
            def read(self, count=-1):
                if job["read_delay"]:
                    time.sleep(job["read_delay"])
                return super().read(count)

        def send(adapter, prepared, **kwargs):
            _require(adapter.max_retries.total == 0 and kwargs == {"stream": True,
                "timeout": (job["limits"]["connect_timeout_seconds"], job["limits"]["read_timeout_seconds"]),
                "verify": True, "cert": None, "proxies": {}})
            if job["fail_send"]:
                raise OSError("BL1_SYNTHETIC_SEND_FAILURE")
            raw = HTTPResponse(body=LocalStream(body), status=job["status"],
                headers=HTTPHeaderDict(job["headers"] if job["headers"] is not None else [("Content-Length", str(len(body)))]),
                preload_content=False, decode_content=False, enforce_content_length=True)
            response = requests.Response()
            response.request, response.url, response.status_code = prepared, prepared.url, job["status"]
            response.raw = raw
            response.headers = requests.structures.CaseInsensitiveDict(raw.headers)
            return response

        with patch.object(requests.adapters.HTTPAdapter, "send", send):
            transport.capture_transport(root, directory, NAME, credential=credential,
                connect_timeout_seconds=job["limits"]["connect_timeout_seconds"],
                read_timeout_seconds=job["limits"]["read_timeout_seconds"],
                max_response_bytes=job["limits"]["max_response_bytes"], budget=budget)
    budget.checkpoint()
    result = _verified_teams(root, directory, names=job["names"], expected_receipt_sha256=job["verified_receipt"])
    budget.checkpoint()
    if observe_runtime:
        payload = _collect_runtime_inventory(root, dict(sys.modules), tuple(observed_code_opens),
            {"operation": job["operation"], "directory": job["directory"],
             "receipt_sha256": result["receipt"]["record_sha256"], "source_manifest_sha256": job["manifest_sha256"]},
            budget.checkpoint)
        result["runtime_inventory"] = {"base64": base64.b64encode(payload).decode("ascii"),
            "sha256": _digest(payload), "size_bytes": len(payload)}
    budget.checkpoint()
    return result


if __name__ == "__main__":
    try:
        _require(sys.argv[1:] == ["--offline-worker"], "BL1_OFFLINE_ONLY")
        payload = sys.stdin.buffer.read(1048577)
        _require(len(payload) <= 1048576)
        result = _worker(_json(payload))
        sys.stdout.buffer.write(_canonical(result))
    except BaseException:
        sys.stdout.buffer.write(_canonical({"ok": False, "code": ERROR}))
        sys.exit(1)

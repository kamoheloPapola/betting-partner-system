"""Offline BL1 admission, aggregate budgets and receipt-first intake.

Standard-library-only startup. There is deliberately no live grant interpreter,
credential-file loader or network-enabled execution mode. A future live connector
requires separate authorization and review. Existing direct transport APIs retain
their previous contract; only this entry point enforces these aggregate rules.
"""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading
import time

EVIDENCE_ROOT = "tmp/BL1-connection-budget-review-remediation-20261025-01"
OFFLINE = {"mode": "offline_fixture", "credential_source": "generated", "network": False}
LIMIT_FIELDS = {"max_attempts", "max_requests", "elapsed_seconds", "max_retained_bytes",
                "capture_storage_bytes", "max_response_bytes"}
ERROR = "BL1_GATE_INCOMPLETE"
_STAGE = "STARTUP"


def _require(value, code=ERROR):
    if not value:
        raise ValueError(code)


def _authorize(authorization):
    # Exact types prevent integer 0 being accepted as the boolean network flag.
    _require(type(authorization) is dict and set(authorization) == set(OFFLINE) and
             authorization.get("mode") == "offline_fixture" and
             authorization.get("credential_source") == "generated" and
             authorization.get("network") is False, "BL1_GATE_OFFLINE_AUTHORIZATION_REQUIRED")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _boundary(repository, directory):
    root, path = Path(repository).absolute(), Path(directory).absolute()
    expected = root / EVIDENCE_ROOT
    _require(".." not in path.parts and path.is_relative_to(expected) and path != expected
             and path.resolve().is_relative_to(expected.resolve()), "BL1_GATE_OUTPUT_SCOPE")
    for item in (*reversed(path.parents), path):
        info = item.lstat()
        _require(not item.is_symlink() and not
                 getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT,
                 "BL1_GATE_REPARSE_POINT")
    return root, path


class OfflineSession:
    """One non-resumable session; pessimistic storage/dispatch reservations.

    A fresh empty existing boundary is exclusively claimed with a zero-byte
    marker. Failures consume attempts and reservations; none are refunded. The
    marker is never automatically removed, so process restart cannot reset the
    budget in this namespace. This is not native cross-process/crash qualification.
    """

    def __init__(self, repository, directory, *, authorization, limits):
        _authorize(authorization)  # before application imports, writes or secrets
        _require(type(limits) is dict and set(limits) == LIMIT_FIELDS, "BL1_GATE_LIMIT_SCHEMA")
        for key, value in limits.items():
            _require((type(value) is int and value > 0) if key != "elapsed_seconds" else
                     type(value) in (int, float) and math.isfinite(value) and value > 0,
                     "BL1_GATE_LIMIT_INVALID")
        self.root, self.directory = _boundary(repository, directory)
        _require(not any(self.directory.iterdir()), "BL1_GATE_DESTINATION_EXISTS")
        with (self.directory / "session.reserved").open("xb"):
            pass
        # A competing claimant cannot pass the exclusive marker creation.
        _require(set(p.name for p in self.directory.iterdir()) == {"session.reserved"})
        self.limits = dict(limits)
        self.deadline = time.monotonic() + limits["elapsed_seconds"]
        self.attempts = self.requests = self.reserved_bytes = 0
        self.failed = False
        # Disk completion records remain provisional. Acceptance exists only in
        # this owning session after persistence and a final deadline check.
        self._accepted = {}
        self._lock = threading.Lock()

    def _remaining(self):
        left = self.deadline - time.monotonic()
        _require(left > 0, "BL1_GATE_DEADLINE")
        return left

    def _persist(self, path, payload):
        _boundary(self.root, path.parent)
        with path.open("xb") as stream:
            _require(stream.write(payload) == len(payload))
            stream.flush()
            os.fsync(stream.fileno())

    def intake_saved(self, attempt, *, names=None):
        """Read a completed session slot through the same isolated receipt gate.

        No files or request slots are added. Incomplete sessions and expired
        sessions cannot be consumed or resumed through this API.
        """
        with self._lock:
            _require(not self.failed, "BL1_GATE_SESSION_INCOMPLETE")
            _require(type(attempt) is int and 1 <= attempt <= self.attempts)
            _require(attempt in self._accepted, "BL1_GATE_CAPTURE_NOT_ACCEPTED")
            self._remaining()
            directory = self.directory / f"attempt-{attempt:06d}"
            job = dict(authorization=dict(OFFLINE), repository=str(self.root), directory=str(directory),
                deadline=self.deadline, allowance=0, operation="intake", names=names,
                accepted_receipt_sha256=self._accepted[attempt])
            env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
            env["PYTHON_DOTENV_DISABLED"] = "1"
            process = None
            try:
                process = subprocess.Popen([sys.executable, "-I", "-S", "-B", __file__, "--offline-worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                stdout, _ = process.communicate(_canonical(job), timeout=self._remaining())
                self._remaining()
                _require(process.returncode == 0)
                result = json.loads(stdout)
                _require(result.get("ok") is True)
                self._remaining()
                return result
            except BaseException:
                self.failed = True
                self._accepted.clear()
                if process is not None and process.poll() is None:
                    process.kill()
                    process.communicate()
            raise ValueError(ERROR)

    def capture_fixture(self, body, *, names=None, headers=None, status=200,
                        read_delay=0, fail_send=False, reflect_credential=False):
        """Execute exactly one controlled response, never a provider operation.

        A dispatched-request slot is charged before worker launch and never
        refunded, even if startup fails. It is an upper bound on actual dispatch,
        not a fabricated count of network requests (which is always zero here).
        """
        with self._lock:
            _require(not self.failed, "BL1_GATE_SESSION_INCOMPLETE")
            self._remaining()
            _require(self.attempts < self.limits["max_attempts"], "BL1_GATE_ATTEMPT_LIMIT")
            self.attempts += 1
            _require(self.requests < self.limits["max_requests"], "BL1_GATE_REQUEST_LIMIT")
            _require(type(body) is bytes and type(status) is int and
                     type(read_delay) in (int, float) and math.isfinite(read_delay) and read_delay >= 0)
            slot = self.directory / f"attempt-{self.attempts:06d}"
            _require(not slot.exists(), "BL1_GATE_DESTINATION_EXISTS")
            self.requests += 1
            admission = _canonical(dict(attempt=self.attempts, request_slot=self.requests,
                storage_reservation=self.limits["capture_storage_bytes"],
                mode="offline_fixture", automatic_retries=0, redirects=0))
            # Reserve every possible capture byte, admission, provisional record
            # and separate failure record (256 bytes each) before any file writes.
            # Count staging/incomplete bytes; no refunds.
            reservation = self.limits["capture_storage_bytes"] + len(admission) + 512
            _require(self.reserved_bytes + reservation <= self.limits["max_retained_bytes"],
                     "BL1_GATE_STORAGE_LIMIT")
            self.reserved_bytes += reservation
            process = None
            stage = "ADMISSION"
            try:
                _boundary(self.root, self.directory)
                slot.mkdir()
                self._persist(slot / "admission.json", admission)
                job = dict(authorization=dict(OFFLINE), repository=str(self.root), directory=str(slot),
                    deadline=self.deadline, allowance=self.limits["capture_storage_bytes"],
                    max_response_bytes=self.limits["max_response_bytes"],
                    body=base64.b64encode(body).decode("ascii"), names=names, headers=headers,
                    status=status, read_delay=read_delay, fail_send=fail_send,
                    reflect_credential=reflect_credential)
                stage = "LAUNCH"
                # Do not inherit credential values into the worker. No values of
                # secret environment variables are read while building this map.
                env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR") if key in os.environ}
                env["PYTHON_DOTENV_DISABLED"] = "1"
                self._remaining()
                process = subprocess.Popen([sys.executable, "-I", "-S", "-B", __file__, "--offline-worker"],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    env=env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                stage = "WORKER"
                stdout, _ = process.communicate(_canonical(job), timeout=self._remaining())
                self._remaining()
                _require(process.returncode == 0)
                result = json.loads(stdout)
                _require(result.get("ok") is True)
                completion = _canonical(dict(status="PROVISIONAL",
                    receipt_sha256=result["receipt"]["record_sha256"]))
                _require(len(completion) <= 256)
                self._remaining()
                self._persist(slot / "result.json", completion)
                result["budget"] = dict(attempts=self.attempts, request_slots=self.requests,
                                         reserved_bytes=self.reserved_bytes)
                self._remaining()
                self._accepted[self.attempts] = result["receipt"]["record_sha256"]
                return result
            except BaseException as error:
                self.failure_stage = stage
                kind = type(error).__name__
                self.failure_type = kind if kind in {"TimeoutExpired", "ValueError", "OSError",
                    "PermissionError", "FileNotFoundError", "KeyboardInterrupt", "SystemExit"} else "Exception"
                self.failed = True
                self._accepted.clear()
                if process is not None and process.poll() is None:
                    process.kill()
                    process.communicate()
                # Preserve partial admissions/provisional results verbatim.
                # Failure reporting uses a separate bounded record and fixed
                # codes; exception text and credential-bearing frames are never
                # serialized. No disk record independently establishes success.
                self.failure_evidence_status = "UNAVAILABLE"
                try:
                    failure_path = slot / "failure.json"
                    if slot.is_dir() and not failure_path.exists():
                        failure = _canonical(dict(status="INCOMPLETE", code=ERROR,
                            stage=stage, error_type=self.failure_type))
                        _require(len(failure) <= 256)
                        self._persist(failure_path, failure)
                        self.failure_evidence_status = "WRITTEN"
                except BaseException:
                    pass
            raise ValueError(ERROR)


class _WorkerBudget:
    def __init__(self, deadline, allowance):
        self.deadline, self.allowance = deadline, allowance
        self.dispatched = False

    def checkpoint(self):
        _require(time.monotonic() < self.deadline, "BL1_GATE_DEADLINE")

    def reserve_capture(self, bound):
        self.checkpoint()
        _require(bound <= self.allowance, "BL1_GATE_STORAGE_LIMIT")

    def dispatch(self):
        self.checkpoint()
        _require(not self.dispatched, "BL1_GATE_REQUEST_LIMIT")
        self.dispatched = True


def _verified_teams(repository, directory, *, names=None, expected_receipt_sha256=None):
    """Internal worker intake. Originals and derived validator view are distinct."""
    from src.evaluation import bl1_capture_transport as transport
    from src.evaluation import future_lineage_context as contract
    receipt = transport.read_transport_capture(repository, directory, "primary-2026-teams")
    if expected_receipt_sha256 is not None:
        _require(receipt["record_sha256"] == expected_receipt_sha256)
    metadata = contract.read(contract.reference(repository, receipt["metadata"]))
    view = json.loads(contract.canonical(metadata))
    view["credential_headers_present"] = list(receipt["authentication"]["header_names"])
    # Receipt validation above must precede importing/calling the legacy parser.
    from src.evaluation import phase11_current_source as legacy
    parsed = legacy._json_payload(view)
    mapped, findings = (legacy._mapped_teams(parsed, "BL1", 2026, names)
                        if names is not None else ({}, []))
    _require(transport.read_transport_capture(repository, directory, "primary-2026-teams") == receipt)
    return dict(ok=True, execution_mode="offline_fixture", receipt=receipt, original_metadata=metadata,
                validator_view=view, parsed=parsed, mapped=mapped, findings=findings)


def _worker(job):
    # This function runs only in a fresh isolated interpreter, before any project
    # or third-party import. Live modes fail before credential access/imports.
    global _STAGE
    _authorize(job.get("authorization"))
    budget = _WorkerBudget(job["deadline"], job["allowance"])
    budget.checkpoint()
    root, directory = _boundary(job["repository"], job["directory"])
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root), str(root / "venv/Lib/site-packages")]
    name = "primary-2026-teams"
    allowed_files = {directory / (name + suffix) for suffix in (".raw", ".json", ".transport.json")}
    allowed_files.update({directory / (name + ".pending") / "response.raw",
        directory / (name + ".pending") / "metadata.json",
        directory / (name + ".transport.pending") / "receipt.json"})
    allowed_dirs = {directory / (name + ".pending"), directory / (name + ".transport.pending")}

    def audit(event, args):
        if event.startswith("socket.") or event in {
                "subprocess.Popen", "os.system", "os.startfile", "os.posix_spawn", "os.fork"}:
            raise PermissionError(ERROR)
        if event == "open":
            path, mode, flags = args
            _require(not isinstance(path, int))
            path = Path(path).absolute()
            writing = ((isinstance(mode, str) and any(c in mode for c in "wax+")) or
                       isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                budget.checkpoint()
                _require(path in allowed_files)
            else:
                _require(not path.name.lower().startswith(".env") and path.name.lower() != "credentials"
                    and path.suffix.lower() not in {".pkl", ".pickle", ".joblib", ".onnx"})
        if event in {"os.mkdir", "os.rmdir"}:
            _require(Path(args[0]).absolute() in allowed_dirs)
        if event == "os.remove":
            _require(Path(args[0]).absolute() in allowed_files)
        if event in {"os.rename", "os.link"}:
            budget.checkpoint()
            _require(all(Path(p).absolute() in allowed_files for p in args[:2]))
        if event in {"os.symlink", "os.chmod", "os.utime", "os.truncate"}:
            raise PermissionError(ERROR)

    sys.addaudithook(audit)
    _STAGE = "IMPORT_TRANSPORT"
    import socket
    socket.has_ipv6 = False
    import io
    import secrets
    from unittest.mock import patch
    import requests
    from urllib3._collections import HTTPHeaderDict
    from urllib3.response import HTTPResponse
    from src.evaluation import bl1_capture_transport as transport
    _STAGE = "CAPTURE"
    budget.checkpoint()
    if job.get("operation") == "intake":
        completion = json.loads((directory / "result.json").read_bytes())
        _require(not (directory / "failure.json").exists() and
                 completion.get("status") == "PROVISIONAL" and
                 isinstance(job.get("accepted_receipt_sha256"), str) and
                 completion.get("receipt_sha256") == job["accepted_receipt_sha256"])
        result = _verified_teams(root, directory, names=job["names"],
                                 expected_receipt_sha256=completion["receipt_sha256"])
        budget.checkpoint()
        return result
    credential = secrets.token_urlsafe(32)  # generated fixture only; never serialized
    body = base64.b64decode(job["body"], validate=True)
    if job["reflect_credential"]:
        body = _canonical(dict(reflection=credential))

    class LocalStream(io.BytesIO):
        def read(self, count=-1):
            time.sleep(job["read_delay"])
            return super().read(count)

    def send(adapter, prepared, **kwargs):
        budget.checkpoint()
        _require(adapter.max_retries.total == 0 and kwargs["proxies"] == {} and kwargs["verify"] is True)
        if job["fail_send"]:
            raise OSError(credential)
        raw = HTTPResponse(body=LocalStream(body), status=job["status"],
            headers=HTTPHeaderDict(job["headers"] if job["headers"] is not None else
                                   [("Content-Length", str(len(body)))]),
            preload_content=False, decode_content=False, enforce_content_length=True)
        response = requests.Response()
        response.request, response.url, response.status_code = prepared, prepared.url, job["status"]
        response.raw = raw
        response.headers = requests.structures.CaseInsensitiveDict(raw.headers)
        return response

    with patch.object(requests.adapters.HTTPAdapter, "send", send):
        transport.capture_transport(root, directory, name, credential=credential,
            connect_timeout_seconds=1, read_timeout_seconds=1,
            max_response_bytes=job["max_response_bytes"], budget=budget)
    budget.checkpoint()
    _STAGE = "INTAKE"
    result = _verified_teams(root, directory, names=job["names"])
    budget.checkpoint()
    _require(credential.encode() not in _canonical(result))
    return result


if __name__ == "__main__":
    # Fixed-code public failure; worker tracebacks and credential-bearing frames
    # never cross the process reporting boundary.
    try:
        _require(sys.argv[1:] == ["--offline-worker"])
        result = _worker(json.loads(sys.stdin.buffer.read()))
        sys.stdout.buffer.write(_canonical(result))
    except BaseException:
        sys.stdout.buffer.write(_canonical(dict(ok=False, code=ERROR, stage=_STAGE)))
        sys.exit(1)

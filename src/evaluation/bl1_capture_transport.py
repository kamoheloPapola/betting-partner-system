"""D1-D4 local transport attestation; no qualification or acquisition grant.

Only the approved BL1 teams/2026 GET is supported. The client owns preparation
and adapter dispatch. Offline tests replace the adapter or its non-networking
pool boundary, never supply independent metadata/response factories. No credential
loader or live authorization is provided. Genuine acquisition requires permission.

Receipts attest local client behavior, not independent provider authentication.
Source pins are declared producer files, not a complete runtime qualification.
"""
from __future__ import annotations

from contextlib import closing
import http.client
import json
import math
from pathlib import Path
import re

import requests
from urllib3 import response as urllib_response
from urllib3 import connection as urllib_connection

from src.evaluation import bl1_bounded_capture as capture
from src.evaluation import future_lineage_context as contract
from src.evaluation import future_lineage_qualification as qualification

URL = "https://api.football-data.org/v4/competitions/BL1/teams"
PREPARED_URL = URL + "?season=2026"
PROVIDER = "football-data.org:v4"
FIELDS = frozenset({"schema_version", "provider", "request_sha256", "method",
    "prepared_url", "authentication", "controls", "limits", "http_status", "framing",
    "metadata", "raw", "started_at", "completed_at", "producer_files", "record_sha256"})
AUTHENTICATION = {"method": "X-Auth-Token", "header_names": ["x-auth-token"],
                  "credential_values_retained": False}
CONTROLS = {"allow_redirects": False, "automatic_retries": 0, "tls_verify": True,
            "trust_env": False, "cookies_enabled": False, "hooks_enabled": False}


def _request():
    return {"url": URL, "params": {"season": 2026}}


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _limits(value):
    contract.require(isinstance(value, dict) and set(value) == {
        "connect_timeout_seconds", "read_timeout_seconds", "max_response_bytes"},
        "BL1_TRANSPORT_LIMIT_SCHEMA")
    contract.require(_positive(value["connect_timeout_seconds"]) and
        _positive(value["read_timeout_seconds"]) and
        type(value["max_response_bytes"]) is int and value["max_response_bytes"] > 0,
        "BL1_TRANSPORT_LIMIT_INVALID")


def _paths(repository, output_directory, name):
    contract.require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name),
                     "BL1_TRANSPORT_NAME_INVALID")
    repository = Path(repository).absolute()
    boundary = contract.safe_path(repository,
        Path(output_directory).absolute().relative_to(repository).as_posix())
    contract.require(boundary.is_dir(), "BL1_TRANSPORT_BOUNDARY_MISSING")
    paths = {kind: contract.safe_path(boundary, name + suffix) for kind, suffix in (
        ("raw", ".raw"), ("metadata", ".json"), ("receipt", ".transport.json"),
        ("pending", ".transport.pending"), ("capture_pending", ".pending"))}
    return repository, boundary, paths


def _producer_files(repository):
    files = [Path(p).absolute() for p in (
        __file__, capture.__file__, contract.__file__, qualification.__file__,
        requests.adapters.__file__, requests.models.__file__, requests.sessions.__file__,
        urllib_response.__file__)]
    return sorted((contract.identity(repository, p) for p in files), key=lambda x: x["path"])


def _prepared(request, credential):
    contract.require(request.method == "GET" and request.url == PREPARED_URL and
        request.body is None and dict(request.headers) == {
            "X-Auth-Token": credential, "Accept-Encoding": "identity"} and
        request.hooks == {"response": []} and not list(request._cookies),
        "BL1_TRANSPORT_REQUEST_CHANGED")


def _check_json_reflection(body, credential):
    """Inspect decoded JSON keys/values without rewriting the original bytes."""
    contract.require(credential.encode("latin-1") not in body,
                     "BL1_TRANSPORT_CREDENTIAL_REFLECTION")
    if not body:
        return  # Empty bytes cannot contain a nonempty credential.
    def pairs(items):
        result = {}
        for key, value in items:
            contract.require(key not in result, "BL1_TRANSPORT_JSON_DUPLICATE_KEY")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("BL1_TRANSPORT_JSON_NONFINITE")
    values = [json.loads(body, object_pairs_hook=pairs, parse_constant=nonfinite)]
    while values:
        value = values.pop()
        if isinstance(value, str):
            contract.require(credential not in value, "BL1_TRANSPORT_CREDENTIAL_REFLECTION")
        elif isinstance(value, dict):
            values.extend(value.keys())
            values.extend(value.values())
        elif isinstance(value, list):
            values.extend(value)


def _quiet_transport():
    # http.client's debug output can print authentication headers. Fail before
    # preparing/sending secrets if any relevant class has debugging enabled.
    contract.require(all(cls.debuglevel == 0 for cls in (http.client.HTTPConnection,
        http.client.HTTPSConnection, urllib_connection.HTTPConnection,
        urllib_connection.HTTPSConnection)), "BL1_TRANSPORT_DEBUG_LOGGING_PROHIBITED")


def retention_bound(repository, output_directory, name, *, connect_timeout_seconds,
                    read_timeout_seconds, max_response_bytes):
    """Conservative logical-file byte reservation, including staging duplicates.

    This derives the maximum metadata/receipt lengths from the unchanged producer
    formats. It counts both staging and published names even on filesystems where
    they share storage. Directory allocation, memory and physical disk blocks are
    not represented. No response bytes or credentials are required.
    """
    repository, _, paths = _paths(repository, output_directory, name)
    limits = dict(connect_timeout_seconds=connect_timeout_seconds,
                  read_timeout_seconds=read_timeout_seconds, max_response_bytes=max_response_bytes)
    _limits(limits)
    stamp = "9999-12-31T23:59:59.999999+00:00"  # longest current now() representation
    raw = dict(path=paths["raw"].relative_to(repository).as_posix(),
               sha256="0" * 64, size_bytes=max_response_bytes)
    metadata = dict(schema_version="ffps.phase11_raw_source.v1", provider=PROVIDER,
        request=_request(), started_at=stamp, retrieved_at=stamp, status="RETRIEVED",
        http_status=200, raw=raw, archived_observation=contract.seal(dict(
            basis="archived_response_observation", provider=PROVIDER, raw=raw,
            request_sha256=contract.fingerprint(_request()), not_before=stamp, observed_by=stamp)))
    metadata_size = len(contract.canonical(metadata))
    receipt = contract.seal(dict(schema_version="bl1.transport_observation_receipt.v1",
        provider=PROVIDER, request_sha256=contract.fingerprint(_request()), method="GET",
        prepared_url=PREPARED_URL, authentication=AUTHENTICATION, controls=CONTROLS,
        limits=limits, http_status=200, framing=dict(content_encoding="identity",
            content_length=max_response_bytes, received_entity_bytes=max_response_bytes, complete=True),
        metadata=dict(path=paths["metadata"].relative_to(repository).as_posix(),
                      sha256="0" * 64, size_bytes=metadata_size), raw=raw,
        started_at=stamp, completed_at=stamp, producer_files=_producer_files(repository)))
    return 2 * (max_response_bytes + metadata_size + len(contract.canonical(receipt)))


class _Response:
    """Bounded identity-body reader tied to the owned prepared request."""
    status_code = 200

    def __init__(self, response, prepared, credential, ceiling, budget=None):
        self.budget = budget
        self.response, self.prepared, self.credential = response, prepared, credential
        self.secret = credential.encode("latin-1")
        self.received = 0
        self.complete = False
        self.body = None
        self.offset = 0
        _prepared(prepared, credential)
        contract.require(response.request is prepared and response.url == PREPARED_URL and
            type(response.status_code) is int and response.status_code == 200 and
            response.history == [] and response._content is False and
            response._content_consumed is False, "BL1_TRANSPORT_RESPONSE_CONTEXT")
        raw = response.raw
        contract.require(isinstance(raw, urllib_response.HTTPResponse),
                         "BL1_TRANSPORT_RAW_TYPE")
        lengths = raw.headers.getlist("Content-Length")
        codings = raw.headers.getlist("Content-Encoding")
        contract.require(len(lengths) == 1 and
            re.fullmatch(r"[0-9]+", lengths[0]) is not None and
            not raw.headers.getlist("Transfer-Encoding") and
            (not codings or len(codings) == 1 and codings[0].lower() == "identity"),
            "BL1_TRANSPORT_FRAMING_UNSUPPORTED")
        self.length = int(lengths[0])
        contract.require(self.length <= ceiling and raw.tell() == 0 and
            raw.length_remaining == self.length and raw.enforce_content_length is True and
            raw.chunked is False and raw.decode_content is False,
            "BL1_TRANSPORT_FRAMING_STATE")
        # Both client views must agree; duplicated/conflicting raw fields fail above.
        contract.require(response.headers.get("Content-Length") == lengths[0] and
            "Transfer-Encoding" not in response.headers and
            response.headers.get("Content-Encoding", "identity").lower() == "identity",
            "BL1_TRANSPORT_HEADER_DISAGREEMENT")
        self.raw = raw
        self.header_state = tuple(tuple(raw.headers.getlist(key)) for key in
                                  ("Content-Length", "Content-Encoding", "Transfer-Encoding"))

    def _context(self):
        if self.budget is not None:
            self.budget.checkpoint()
        _prepared(self.prepared, self.credential)
        contract.require(self.response.request is self.prepared and
            self.response.url == PREPARED_URL and self.response.status_code == 200 and
            self.response.history == [] and self.response.raw is self.raw and
            self.raw.enforce_content_length is True and self.raw.chunked is False and
            self.raw.decode_content is False and self.header_state == tuple(
                tuple(self.raw.headers.getlist(key)) for key in
                ("Content-Length", "Content-Encoding", "Transfer-Encoding")) and
            self.response.headers.get("Content-Length") == str(self.header_state[0][0]) and
            "Transfer-Encoding" not in self.response.headers and
            self.response.headers.get("Content-Encoding", "identity").lower() == "identity",
            "BL1_TRANSPORT_RESPONSE_CHANGED")

    def _load(self):
        # Nothing is returned to the capture helper until the complete bounded
        # body passes framing and decoded-JSON privacy checks. No disk quarantine
        # or sanitized substitute for original bytes is created.
        body = bytearray()
        while True:
            count = min(65536, self.length - self.received + 1)
            self._context()
            block = self.raw.read(count, decode_content=False)
            self._context()
            contract.require(isinstance(block, bytes) and len(block) <= count,
                             "BL1_TRANSPORT_READ_INVALID")
            self.received += len(block)
            contract.require(self.received <= self.length, "BL1_TRANSPORT_LENGTH_EXCEEDED")
            if not block:
                contract.require(self.received == self.length and self.raw.length_remaining == 0,
                                 "BL1_TRANSPORT_TRUNCATED")
                break
            body.extend(block)
        original = bytes(body)
        _check_json_reflection(original, self.credential)
        self.body = original

    def read(self, count):
        self._context()
        if self.body is None:
            self._load()
        block = self.body[self.offset:self.offset + count]
        self.offset += len(block)
        self._context()
        if not block:
            self.complete = True
        return block

    def close(self):
        self._context()
        self.response.close()
        self._context()


def _validate(repository, paths, *, pending_allowed=False):
    contract.require(pending_allowed or not paths["pending"].exists(),
                     "BL1_TRANSPORT_INCOMPLETE")
    contract.require(not paths["capture_pending"].exists(), "BL1_TRANSPORT_INCOMPLETE")
    receipt = contract.read(paths["receipt"])
    contract.require(isinstance(receipt, dict) and set(receipt) == FIELDS,
                     "BL1_TRANSPORT_RECEIPT_SCHEMA")
    contract.sealed(receipt)
    contract.require(receipt["schema_version"] == "bl1.transport_observation_receipt.v1" and
        receipt["provider"] == PROVIDER and receipt["method"] == "GET" and
        receipt["prepared_url"] == PREPARED_URL and
        receipt["request_sha256"] == contract.fingerprint(_request()) and
        type(receipt["http_status"]) is int and receipt["http_status"] == 200,
        "BL1_TRANSPORT_RECEIPT_IDENTITY")
    contract.require(contract.canonical(receipt["authentication"]) == contract.canonical(AUTHENTICATION)
        and contract.canonical(receipt["controls"]) == contract.canonical(CONTROLS),
        "BL1_TRANSPORT_RECEIPT_CONTROLS")
    _limits(receipt["limits"])
    for kind in ("metadata", "raw"):
        ref = receipt[kind]
        contract.require(isinstance(ref, dict) and type(ref.get("size_bytes")) is int and
            ref["size_bytes"] >= 0, "BL1_TRANSPORT_REFERENCE_TYPE")
        contract.require(contract.reference(repository, receipt[kind]) == paths[kind],
                         "BL1_TRANSPORT_ARTIFACT_SLOT")
    metadata = contract.read(paths["metadata"])
    contract.require(metadata["provider"] == PROVIDER and metadata["request"] == _request()
        and contract.canonical(receipt["raw"]) == contract.canonical(metadata["raw"]) and
        receipt["started_at"] == metadata["started_at"], "BL1_TRANSPORT_CROSS_BINDING")
    qualification.observation(metadata, repository, cutoff=receipt["completed_at"])
    contract.require(contract.instant(receipt["completed_at"]) >= contract.instant(metadata["retrieved_at"]),
                     "BL1_TRANSPORT_COMPLETION_TIME")
    frame = receipt["framing"]
    contract.require(isinstance(frame, dict) and set(frame) == {
        "content_encoding", "content_length", "received_entity_bytes", "complete"} and
        frame["content_encoding"] == "identity" and frame["complete"] is True and
        type(frame["content_length"]) is int and frame["content_length"] >= 0 and
        type(frame["received_entity_bytes"]) is int and
        frame["content_length"] == frame["received_entity_bytes"] == receipt["raw"]["size_bytes"] and
        frame["received_entity_bytes"] <= receipt["limits"]["max_response_bytes"],
        "BL1_TRANSPORT_RECEIPT_FRAMING")
    contract.require(isinstance(receipt["producer_files"], list) and
        contract.canonical(receipt["producer_files"]) == contract.canonical(_producer_files(repository)),
        "BL1_TRANSPORT_PRODUCER_DRIFT")
    contract.require(pending_allowed or not paths["pending"].exists(),
                     "BL1_TRANSPORT_INCOMPLETE")
    return receipt


def read_transport_capture(repository, output_directory, name):
    """Require the approved receipt; old metadata alone never means success here."""
    repository, _, paths = _paths(repository, output_directory, name)
    return _validate(repository, paths)


def _capture_sensitive(repository, output_directory, name, *, credential,
                       connect_timeout_seconds, read_timeout_seconds, max_response_bytes, budget=None):
    """Owned, one-operation transport path. Execute only under separate authority.

    This stage tests this path with a patched HTTPAdapter.send and a network guard.
    A failed operation retains its pending marker and any published artifacts;
    neither this producer nor its consumer automatically reuses that destination.
    Error messages are fixed codes: no third-party exception text or request
    object containing authentication values crosses this public error boundary.
    """
    try:
        limits = dict(connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds, max_response_bytes=max_response_bytes)
        _limits(limits)
        contract.require(isinstance(credential, str) and bool(credential),
                         "BL1_TRANSPORT_CREDENTIAL_REQUIRED")
        credential.encode("latin-1")
        _quiet_transport()
        public = {"repository": str(repository), "directory": str(output_directory),
            "name": name, "provider": PROVIDER, "request": _request(),
            "authentication": AUTHENTICATION, "controls": CONTROLS, "limits": limits}
        contract.require(credential.encode("latin-1") not in contract.canonical(public),
                         "BL1_TRANSPORT_CREDENTIAL_REFLECTION")
        repository, _, paths = _paths(repository, output_directory, name)
        if budget is not None:
            budget.checkpoint()
            budget.reserve_capture(retention_bound(repository, output_directory, name, **limits))
        paths["pending"].mkdir()
        if any(p.exists() for kind, p in paths.items() if kind != "pending"):
            # We own only this new, still-empty reservation. No operation began.
            # Releasing it preserves any completed capture from a competing call.
            paths["pending"].rmdir()
            raise ValueError("BL1_TRANSPORT_DESTINATION_EXISTS")
        prepared = requests.Request("GET", URL, params={"season": 2026},
            headers={"X-Auth-Token": credential, "Accept-Encoding": "identity"}).prepare()
        _prepared(prepared, credential)
        state = {}
        # Direct owned adapter dispatch has no Session environment/cookie/hook or
        # redirect machinery. Requests' adapter disables urllib3 redirects.
        with closing(requests.adapters.HTTPAdapter(max_retries=0)) as adapter:
            contract.require(adapter.max_retries.total == 0, "BL1_TRANSPORT_RETRY_POLICY")
            def open_response():
                if budget is not None:
                    budget.dispatch()
                response = adapter.send(prepared, stream=True,
                    timeout=(connect_timeout_seconds, read_timeout_seconds),
                    verify=True, cert=None, proxies={})
                try:
                    wrapped = _Response(response, prepared, credential, max_response_bytes, budget)
                except Exception:
                    response.close()
                    raise
                state["response"] = wrapped
                return wrapped
            metadata = capture.capture_response(repository, output_directory, name,
                provider=PROVIDER, request=_request(), open_response=open_response,
                max_response_bytes=max_response_bytes)
        wrapped = state["response"]
        contract.require(wrapped.complete, "BL1_TRANSPORT_INCOMPLETE")
        # Owner-approved amendment: completed_at is local attestation-sealing time
        # after metadata/raw validation, before receipt publication/final readback.
        # Success still requires all three artifacts; this is no publication or
        # durability certificate. The closed field set is unchanged.
        metadata_ref = contract.identity(repository, paths["metadata"])
        contract.require(contract.read(paths["metadata"]) == metadata,
                         "BL1_TRANSPORT_METADATA_READBACK")
        qualification.observation(metadata, repository, cutoff=contract.now())
        receipt = contract.seal({
            "schema_version": "bl1.transport_observation_receipt.v1", "provider": PROVIDER,
            "request_sha256": contract.fingerprint(metadata["request"]), "method": "GET",
            "prepared_url": PREPARED_URL, "authentication": dict(AUTHENTICATION),
            "controls": dict(CONTROLS), "limits": limits, "http_status": 200,
            "framing": {"content_encoding": "identity", "content_length": wrapped.length,
                "received_entity_bytes": wrapped.received, "complete": True},
            "metadata": metadata_ref, "raw": metadata["raw"],
            "started_at": metadata["started_at"], "completed_at": contract.now(),
            "producer_files": _producer_files(repository),
        })
        payload = contract.canonical(receipt)
        if budget is not None:
            budget.checkpoint()
        contract.require(credential.encode("latin-1") not in payload,
                         "BL1_TRANSPORT_CREDENTIAL_REFLECTION")
        temporary = paths["pending"] / "receipt.json"
        capture._persist(temporary, payload)
        contract.require(contract.read(temporary) == receipt, "BL1_TRANSPORT_RECEIPT_READBACK")
        # Recheck containment immediately before exclusive publication.
        _, _, checked = _paths(repository, output_directory, name)
        capture._publish(temporary, checked["receipt"])
        validated = _validate(repository, paths, pending_allowed=True)
        if budget is not None:
            budget.checkpoint()
        temporary.unlink(missing_ok=True)
        paths["pending"].rmdir()
        return validated
    except Exception:
        # Exit the handler before raising: even __context__ must not expose a
        # third-party exception whose message might contain the credential.
        pass
    raise ValueError("BL1_TRANSPORT_CAPTURE_INCOMPLETE")


def capture_transport(repository, output_directory, name, *, credential,
                      connect_timeout_seconds, read_timeout_seconds, max_response_bytes, budget=None):
    """Capture with a credential-safe public failure frame.

    Permitted reporting is format_transport_error(error) only. Do not serialize
    traceback objects, caller locals, request objects, inputs or exception reprs.
    Caller frames can contain secrets and are outside this function's control.
    This is a reporting policy, not process-memory erasure or hostile-debugger
    protection. Non-Exception process termination is not a logging guarantee.
    """
    try:
        return _capture_sensitive(repository, output_directory, name, credential=credential,
            connect_timeout_seconds=connect_timeout_seconds, read_timeout_seconds=read_timeout_seconds,
            max_response_bytes=max_response_bytes, budget=budget)
    except Exception:
        pass
    # The new exception has no sensitive helper frames or cause/context. Clear
    # arguments before raising so even this component's captured locals are safe.
    repository = output_directory = name = credential = None
    connect_timeout_seconds = read_timeout_seconds = max_response_bytes = None
    budget = None
    raise ValueError("BL1_TRANSPORT_CAPTURE_INCOMPLETE")


def format_transport_error(error):
    """The only permitted log/report value; never format the supplied exception."""
    return "BL1_TRANSPORT_CAPTURE_INCOMPLETE"

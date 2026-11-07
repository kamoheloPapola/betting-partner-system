"""Bounded response-body capture; no transport, qualification or grant issuance.

The caller supplies an authorized response factory, safe request metadata, and an
existing output boundary. ``read(n)`` must return at most n decoded entity-body
bytes, not a sanitized projection. This component never imports an HTTP client.
Offline streams prove mechanics only, not provider provenance or acceptance.

Raw bytes are published before the immutable metadata sidecar. An interruption
between publications can leave an orphan raw file; it is not a completed capture.
Existing destinations are never reused. Path checks are point-in-time checks,
not continuous protection against another process modifying the filesystem.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import re

from src.evaluation.future_lineage_context import (
    canonical, fingerprint, identity, instant, now, read, require, safe_path, seal,
)
from src.evaluation.future_lineage_qualification import observation


def _write_checked(stream, block):
    require(stream.write(block) == len(block), "BL1_CAPTURE_INCOMPLETE_WRITE")


def _persist(path, payload):
    with path.open("xb") as stream:
        _write_checked(stream, payload)
        stream.flush()
        os.fsync(stream.fileno())


def _publish(source, destination):
    # Windows rename refuses an existing destination. POSIX rename can replace,
    # so use exclusive hard-link publication there. Neither branch falls back
    # after an access denial or uses os.replace.
    if os.name == "nt":
        os.rename(source, destination)
    else:
        os.link(source, destination)


def capture_response(repository, output_directory, name, *, provider, request,
                     open_response, max_response_bytes, expected_response_bytes=None):
    """Persist one HTTP-200 response supplied by a caller-owned response factory.

    The factory is called after the genuine local start timestamp. Its object
    must expose ``status_code``, ``read(n)`` and ``close()``. It must not retry,
    redirect or prebuffer an unbounded response. No network is performed here.
    Request metadata uses the existing {url, params} representation, without
    credentials. The caller is responsible for its authority and provenance.
    An optional expected_response_bytes checks decoded body length, when known.
    It must not be an encoded HTTP Content-Length for a compressed body. Without
    an independently known length, the supplied reader must signal truncation;
    this component cannot distinguish silent truncation from legitimate EOF.

    Completion means EOF, successful closure, flushed bytes and validated local
    readback. It does not establish publication time at the provider, maximum
    age, power-loss durability, process-crash qualification or source acceptance.
    Exceptions propagate; no completed sidecar is published for an unfinished
    body. A leftover .pending directory after a process crash blocks name reuse
    and requires separate reconciliation, rather than automatic deletion/retry.
    """
    require(type(max_response_bytes) is int and max_response_bytes > 0,
            "BL1_CAPTURE_SIZE_LIMIT_REQUIRED")
    require(expected_response_bytes is None or
            (type(expected_response_bytes) is int and
             0 <= expected_response_bytes <= max_response_bytes),
            "BL1_CAPTURE_EXPECTED_SIZE_INVALID")
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name),
            "BL1_CAPTURE_NAME_INVALID")
    require(isinstance(provider, str) and bool(provider.strip()),
            "BL1_CAPTURE_PROVIDER_REQUIRED")
    require(isinstance(request, dict) and set(request) == {"url", "params"}
            and isinstance(request["url"], str) and bool(request["url"])
            and isinstance(request["params"], dict), "BL1_CAPTURE_REQUEST_INVALID")
    # Snapshot the exact existing JSON representation before invoking the caller.
    request = json.loads(canonical(request))
    repository = Path(repository).absolute()
    output_directory = Path(output_directory).absolute()
    relative = output_directory.relative_to(repository).as_posix()
    boundary = safe_path(repository, relative)
    require(boundary.is_dir(), "BL1_CAPTURE_BOUNDARY_MISSING")
    raw_path = safe_path(boundary, name + ".raw")
    meta_path = safe_path(boundary, name + ".json")
    require(not raw_path.exists() and not meta_path.exists(),
            "BL1_CAPTURE_DESTINATION_EXISTS")
    pending = safe_path(boundary, name + ".pending")
    # Exclusive reservation prevents two instances of this helper sharing a name.
    pending.mkdir()
    private_raw, private_meta = pending / "response.raw", pending / "metadata.json"
    try:
        started = now()
        total = 0
        with closing(open_response()) as response:
            require(type(response.status_code) is int and response.status_code == 200,
                    "BL1_CAPTURE_HTTP_STATUS")
            with private_raw.open("xb") as stream:
                while True:
                    # One extra byte distinguishes exact-limit EOF from overflow.
                    requested = min(65536, max_response_bytes - total + 1)
                    block = response.read(requested)
                    require(isinstance(block, bytes) and len(block) <= requested,
                            "BL1_CAPTURE_STREAM_INVALID")
                    if not block:
                        observed = now()
                        break
                    total += len(block)
                    require(total <= max_response_bytes, "BL1_CAPTURE_RESPONSE_TOO_LARGE")
                    _write_checked(stream, block)
                require(expected_response_bytes is None or total == expected_response_bytes,
                        "BL1_CAPTURE_RESPONSE_TRUNCATED_OR_SIZE_MISMATCH")
                stream.flush()
                os.fsync(stream.fileno())
        retrieved = now()
        require(instant(started) <= instant(observed) <= instant(retrieved),
                "BL1_CAPTURE_CLOCK_CONTRADICTION")
        raw = {**identity(repository, private_raw),
               "path": raw_path.relative_to(repository).as_posix()}
        require(raw["size_bytes"] == total, "BL1_CAPTURE_SIZE_DRIFT")
        record = {
            "schema_version": "ffps.phase11_raw_source.v1",
            "provider": provider, "request": request,
            "started_at": started, "retrieved_at": retrieved,
            "status": "RETRIEVED", "http_status": 200, "raw": raw,
            "archived_observation": seal({
                "basis": "archived_response_observation", "provider": provider,
                "raw": raw, "request_sha256": fingerprint(request),
                "not_before": started, "observed_by": observed,
            }),
        }
        _persist(private_meta, canonical(record))
        require(read(private_meta) == record, "BL1_CAPTURE_METADATA_READBACK")
        safe_path(repository, relative)
        safe_path(boundary, name + ".raw")
        safe_path(boundary, name + ".json")
        _publish(private_raw, raw_path)
        # Reuse the existing byte-reference, seal and temporal validator before
        # publishing any completed metadata. No freshness policy is added.
        observation(read(private_meta), repository, cutoff=retrieved)
        _publish(private_meta, meta_path)
        persisted = read(meta_path)
        require(persisted == record, "BL1_CAPTURE_METADATA_READBACK")
        observation(persisted, repository, cutoff=retrieved)
        return persisted
    finally:
        # Only our reserved staging files are removed. Published files, including
        # any orphan raw file, and pre-existing user files are never cleaned up.
        private_raw.unlink(missing_ok=True)
        private_meta.unlink(missing_ok=True)
        pending.rmdir()

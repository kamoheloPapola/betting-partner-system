"""Bounded local publication primitives, separate from qualification and evidence.

These Python objects are internal interfaces, not a new wire format. The caller
supplies authenticated, current per-operation permission and an IO adapter bound
to one target. This module authenticates neither that caller nor its environment.
Readback establishes an observation, not continuous namespace/leaf protection,
an actual acceptance event, a retained acknowledgement, or VERIFIED_COMPLETE.

No retry, replacement, unlink, automatic evidence write, worker, or child process
is used. The existing v1 validator and its wire contract remain unchanged.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Protocol

from src.evaluation import phase20_local_acceptance as acceptance


# Retain the preparation publisher's ceiling; this is not a budget amendment.
MAX_PAYLOAD_BYTES = 65536


@dataclass(frozen=True)
class FrozenAcceptance:
    """Exact canonical bytes checked against an independently supplied digest.

    RecordingContext remains caller-supplied validation context, not filesystem
    permission. Constructing this object has no filesystem or clock effects.
    """

    payload: bytes
    expected_sha256: str
    context: acceptance.RecordingContext

    def __post_init__(self):
        if type(self.payload) is not bytes or not 0 < len(self.payload) <= MAX_PAYLOAD_BYTES:
            raise ValueError('P20_PUBLICATION_PAYLOAD_BOUND')
        result = acceptance.validate_record_bytes(
            self.payload, context=self.context,
            expected_whole_file_sha256=self.expected_sha256,
            require_external_digest=True)
        if not result.valid:
            raise ValueError(';'.join(result.errors))
        if result.record_status != acceptance.RECORDED:
            raise ValueError('P20_PUBLICATION_RECORDED_INPUT_REQUIRED')
        if acceptance.canonical_bytes(acceptance.parse_record(self.payload)) != self.payload:
            raise ValueError('P20_PUBLICATION_CANONICAL_BYTES_REQUIRED')


@dataclass(frozen=True)
class FileIdentity:
    volume_serial: int
    file_id: bytes
    final_path: str

    def __post_init__(self):
        if (type(self.volume_serial) is not int or not 0 <= self.volume_serial < 2**64
                or type(self.file_id) is not bytes or len(self.file_id) != 16
                or type(self.final_path) is not str or not self.final_path):
            raise ValueError('P20_PUBLICATION_INVALID_IDENTITY')


class PublicationIO(Protocol):
    """Synchronous, single-target IO. A failed close must never be retried here.

    The adapter must enforce actual ownership and reject reuse/foreign handles.
    Native calls and observation costs belong in future complete accounting.
    """

    def create_new(self) -> object: ...
    def identity(self, handle: object) -> FileIdentity: ...
    def write_once(self, handle: object, payload: bytes) -> int: ...
    def flush(self, handle: object) -> None: ...
    def close(self, handle: object) -> None: ...
    def reopen_read(self) -> object: ...
    def length(self, handle: object) -> int: ...
    def read(self, handle: object, count: int) -> bytes: ...


@dataclass(frozen=True)
class PublicationResult:
    """An in-memory observation; never a completion confirmation or receipt.

    pending_handle retains the actual adapter capability when release could not
    be confirmed. It is not a new permission to close/retry an uncertain handle.
    None in returned_count means no reliable write count was observed.
    """

    local_readback_verified: bool
    last_operation: str
    requested_count: int
    returned_count: int | None
    release_confirmed: bool
    errors: tuple[str, ...] = ()
    pending_handle: object | None = field(default=None, repr=False, compare=False)
    completion_status: str = field(default='COMPLETION_UNRESOLVED', init=False)


class PublicationStopped(RuntimeError):
    """The current operation lacks permission; no later ordinary IO may start."""


def _error(operation: str, exc: BaseException) -> str:
    return operation + ':' + type(exc).__name__ + ':' + str(exc)


def publish_acceptance_bytes(
    frozen: FrozenAcceptance,
    io: PublicationIO,
    *,
    permit: Callable[[str], bool],
    release_permit: Callable[[], bool] | None = None,
) -> PublicationResult:
    """Run one bounded publication prefix with explicit per-operation gates.

    This low-level routine is not connected to production acceptance or a CLI.
    The IO adapter binds its target independently; synthetic targets must not be
    confused with the v1 destination literal inside validation context.

    permit must return exactly True before each ordinary operation. After a stop
    or error only a separate release_permit can permit one close of a known owned
    handle, and only if no close was already attempted. An ambiguous close is
    never retried. Abrupt BaseException interruptions propagate after that same
    bounded release policy; no fictitious operation/evidence record is emitted.
    """
    if type(frozen) is not FrozenAcceptance or not callable(permit):
        raise TypeError('P20_PUBLICATION_INPUT')
    if release_permit is not None and not callable(release_permit):
        raise TypeError('P20_PUBLICATION_RELEASE_PERMIT')
    # Revalidate at the boundary; callers cannot use a forged dataclass instance.
    frozen.__post_init__()
    handle = None
    acquisition_unresolved = False
    close_attempted = False
    operation = 'PRECHECK'
    returned_count = None
    errors = []
    readback_verified = False

    def allowed(name):
        nonlocal operation
        operation = name
        if permit(name) is not True:
            raise PublicationStopped('P20_PUBLICATION_PERMISSION_REQUIRED')

    def close_owned(name):
        nonlocal handle, close_attempted
        allowed(name)
        close_attempted = True
        io.close(handle)
        handle = None
        close_attempted = False

    try:
        allowed('CREATE_NEW')
        acquisition_unresolved = True
        handle = io.create_new()
        if handle is None:
            raise ValueError('P20_PUBLICATION_OWNED_HANDLE_REQUIRED')
        acquisition_unresolved = False
        allowed('WRITER_IDENTITY')
        writer_identity = io.identity(handle)
        if type(writer_identity) is not FileIdentity:
            raise ValueError('P20_PUBLICATION_IDENTITY_REQUIRED')
        allowed('WRITE_FILE')
        returned_count = io.write_once(handle, frozen.payload)
        if type(returned_count) is not int or returned_count != len(frozen.payload):
            raise ValueError('P20_PUBLICATION_SHORT_OR_INVALID_WRITE')
        allowed('FLUSH_FILE_BUFFERS')
        io.flush(handle)
        close_owned('CLOSE_WRITER')
        allowed('REOPEN_READ')
        acquisition_unresolved = True
        handle = io.reopen_read()
        if handle is None:
            raise ValueError('P20_PUBLICATION_OWNED_HANDLE_REQUIRED')
        acquisition_unresolved = False
        allowed('READER_IDENTITY')
        if io.identity(handle) != writer_identity:
            raise ValueError('P20_PUBLICATION_IDENTITY_CHANGED')
        allowed('QUERY_LENGTH')
        length = io.length(handle)
        if type(length) is not int or length != len(frozen.payload):
            raise ValueError('P20_PUBLICATION_LENGTH_MISMATCH')
        allowed('READ_FILE')
        observed = io.read(handle, len(frozen.payload))
        if type(observed) is not bytes or observed != frozen.payload:
            raise ValueError('P20_PUBLICATION_BYTES_MISMATCH')
        allowed('READ_EOF')
        eof = io.read(handle, 1)
        if type(eof) is not bytes or eof != b'':
            raise ValueError('P20_PUBLICATION_EOF_REQUIRED')
        if hashlib.sha256(observed).hexdigest() != frozen.expected_sha256:
            raise ValueError('P20_PUBLICATION_DIGEST_MISMATCH')
        close_owned('CLOSE_READER')
        # Permission may have changed while readback or release was in progress.
        allowed('FINAL_CHECK')
        readback_verified = True
    except Exception as exc:
        errors.append(_error(operation, exc))
    finally:
        if handle is not None and not close_attempted and release_permit is not None:
            try:
                if release_permit() is True:
                    close_attempted = True
                    io.close(handle)
                    handle = None
            except Exception as exc:
                errors.append(_error('RELEASE', exc))

    return PublicationResult(readback_verified, operation, len(frozen.payload),
                             returned_count, handle is None and not acquisition_unresolved,
                             tuple(errors), handle)

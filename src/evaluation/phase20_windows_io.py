"""Synchronous Win32 primitives for the local publication controller.

Importing this module performs no Windows calls. Instantiation binds kernel32;
operations occur only on explicit calls. This is not an FC-B01 namespace guard:
ancestor mutation, drive-map changes, existing capabilities, and holder-failure
ordering remain outside its guarantee. Final-path/identity checks detect some
changes, but do not prove continuous protection or make this qualification-ready.

There is no worker, timeout, cancellation API, retry, automatic close, destructor,
directory creation, or evidence publication. Each handle belongs to one adapter
instance. A failed close is quarantined and cannot be retried through the adapter.
"""
from __future__ import annotations

import ctypes
import os
import re
from ctypes import wintypes
from dataclasses import dataclass, field

from src.evaluation.phase20_publication import FileIdentity, MAX_PAYLOAD_BYTES


class _FileIdInfo(ctypes.Structure):
    _fields_ = [('volume', ctypes.c_ulonglong), ('file_id', ctypes.c_ubyte * 16)]


class _AttributeTagInfo(ctypes.Structure):
    _fields_ = [('attributes', wintypes.DWORD), ('reparse_tag', wintypes.DWORD)]


@dataclass(eq=False)
class _OwnedHandle:
    value: int
    writer: bool
    close_attempted: bool = False
    write_attempted: bool = False
    write_complete: bool = False
    flush_attempted: bool = False
    uncertain_buffers: list = field(default_factory=list, repr=False)


def validate_local_path(target: str) -> None:
    """Reject ambiguous lexical targets; this does not inspect any ancestor."""
    if (type(target) is not str or len(target) > 512
            or re.fullmatch(r'[A-Za-z]:\\[^/]+', target) is None):
        raise ValueError('P20_PUBLICATION_LOCAL_ABSOLUTE_PATH_REQUIRED')
    parts = target[3:].split('\\')
    if not 1 <= len(parts) <= 32:
        raise ValueError('P20_PUBLICATION_PATH_DEPTH')
    for part in parts:
        stem = part.split('.', 1)[0].rstrip(' ')
        if (not part or part in ('.', '..') or part.endswith(('.', ' '))
                or any(ord(c) < 32 or ord(c) > 126 or c in '<>:"|?*~' for c in part)
                or re.fullmatch(r'CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]|CONIN\$|CONOUT\$|CLOCK\$',
                                stem, flags=re.IGNORECASE)):
            raise ValueError('P20_PUBLICATION_UNSAFE_PATH_COMPONENT')


class WindowsPublicationIO:
    """Bind one exact path. Caller gates and environmental assurance are external.

    _kernel32/_last_error are test seams, not qualification adapters or authority.
    No native execution is implied by tests using these seams.
    """

    def __init__(self, target: str, *, _kernel32=None, _last_error=None):
        validate_local_path(target)
        if _kernel32 is None:
            if os.name != 'nt':
                raise OSError('P20_PUBLICATION_WINDOWS_REQUIRED')
            _kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        self._target = target
        self._api = _kernel32
        self._last_error = _last_error if _last_error is not None else ctypes.get_last_error
        self._handles = set()
        self._open_attempts = set()
        self._bind()

    @property
    def target(self):
        return self._target

    @property
    def pending_handles(self):
        """Owned capabilities still open or uncertain; not permission to close."""
        return tuple(self._handles)

    def _bind(self):
        declarations = {
            'CreateFileW': ([wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                             ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                             wintypes.HANDLE], wintypes.HANDLE),
            'WriteFile': ([wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                           ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p], wintypes.BOOL),
            'ReadFile': ([wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                          ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p], wintypes.BOOL),
            'FlushFileBuffers': ([wintypes.HANDLE], wintypes.BOOL),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
            'GetFileInformationByHandleEx': ([wintypes.HANDLE, ctypes.c_int,
                                             ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            'GetFinalPathNameByHandleW': ([wintypes.HANDLE, wintypes.LPWSTR,
                                          wintypes.DWORD, wintypes.DWORD], wintypes.DWORD),
            'GetFileSizeEx': ([wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)], wintypes.BOOL),
        }
        for name, (args, result) in declarations.items():
            function = getattr(self._api, name)
            function.argtypes = args
            function.restype = result

    def _failure(self, operation, code):
        error = OSError(code, operation + ' failed')
        error.winerror = code
        return error

    def _owned(self, handle, *, writer=None):
        if (type(handle) is not _OwnedHandle or handle not in self._handles
                or handle.close_attempted or handle.uncertain_buffers
                or (writer is not None and handle.writer is not writer)):
            raise ValueError('P20_PUBLICATION_HANDLE_OWNERSHIP_OR_STATE')
        return handle.value

    def _open(self, *, writer):
        if self._handles or writer in self._open_attempts:
            raise ValueError('P20_PUBLICATION_OPEN_RETRY_OR_OUTSTANDING_HANDLE')
        self._open_attempts.add(writer)
        # Writer: exclusive CREATE_NEW. Reader: OPEN_EXISTING, deny write/delete.
        access = 0x80000000 | (0x40000000 if writer else 0)
        raw = self._api.CreateFileW(self.target, access, 0 if writer else 1,
                                    None, 1 if writer else 3,
                                    0x80 if writer else 0x80 | 0x00200000, None)
        if raw is None or raw == ctypes.c_void_p(-1).value:
            code = self._last_error()
            raise self._failure('CreateFileW', code)
        handle = _OwnedHandle(raw, writer)
        self._handles.add(handle)
        return handle

    def create_new(self):
        return self._open(writer=True)

    def reopen_read(self):
        return self._open(writer=False)

    def identity(self, handle):
        raw = self._owned(handle)
        info = _FileIdInfo()
        if not self._api.GetFileInformationByHandleEx(raw, 18, ctypes.byref(info), ctypes.sizeof(info)):
            code = self._last_error()
            raise self._failure('GetFileInformationByHandleEx(FileIdInfo)', code)
        attributes = _AttributeTagInfo()
        if not self._api.GetFileInformationByHandleEx(raw, 9, ctypes.byref(attributes), ctypes.sizeof(attributes)):
            code = self._last_error()
            raise self._failure('GetFileInformationByHandleEx(FileAttributeTagInfo)', code)
        if attributes.attributes & (0x400 | 0x10):
            raise ValueError('P20_PUBLICATION_REGULAR_NONREPARSE_FILE_REQUIRED')
        # 512 path characters + extended DOS prefix and terminator. No resize retry.
        buffer = ctypes.create_unicode_buffer(517)
        count = self._api.GetFinalPathNameByHandleW(raw, buffer, len(buffer), 0)
        if not count:
            code = self._last_error()
            raise self._failure('GetFinalPathNameByHandleW', code)
        if count >= len(buffer) or buffer.value != '\\\\?\\' + self.target:
            raise ValueError('P20_PUBLICATION_FINAL_PATH_MISMATCH')
        return FileIdentity(info.volume, bytes(info.file_id), buffer.value)

    def write_once(self, handle, payload):
        raw = self._owned(handle, writer=True)
        if handle.write_attempted:
            raise ValueError('P20_PUBLICATION_WRITE_RETRY')
        if type(payload) is not bytes or not 0 < len(payload) <= MAX_PAYLOAD_BYTES:
            raise ValueError('P20_PUBLICATION_WRITE_BOUND')
        buffer = ctypes.create_string_buffer(payload, len(payload))
        count = wintypes.DWORD()
        # Consume the attempt before crossing the native boundary, including
        # exceptions, short writes, and unknown results. No repair write follows.
        handle.write_attempted = True
        if not self._api.WriteFile(raw, buffer, len(payload), ctypes.byref(count), None):
            code = self._last_error()
            # Unexpected pending status must not free possibly borrowed storage.
            if code == 997:
                handle.uncertain_buffers.extend((buffer, count))
            raise self._failure('WriteFile', code)
        handle.write_complete = count.value == len(payload)
        return count.value

    def flush(self, handle):
        raw = self._owned(handle, writer=True)
        if not handle.write_complete or handle.flush_attempted:
            raise ValueError('P20_PUBLICATION_FLUSH_ORDER_OR_RETRY')
        handle.flush_attempted = True
        if not self._api.FlushFileBuffers(raw):
            code = self._last_error()
            raise self._failure('FlushFileBuffers', code)

    def close(self, handle):
        raw = self._owned(handle)
        handle.close_attempted = True
        if not self._api.CloseHandle(raw):
            code = self._last_error()
            raise self._failure('CloseHandle', code)
        self._handles.remove(handle)

    def length(self, handle):
        raw = self._owned(handle, writer=False)
        count = ctypes.c_longlong()
        if not self._api.GetFileSizeEx(raw, ctypes.byref(count)):
            code = self._last_error()
            raise self._failure('GetFileSizeEx', code)
        if not 0 <= count.value <= MAX_PAYLOAD_BYTES:
            raise ValueError('P20_PUBLICATION_LENGTH_BOUND')
        return count.value

    def read(self, handle, count):
        raw = self._owned(handle, writer=False)
        if type(count) is not int or not 1 <= count <= MAX_PAYLOAD_BYTES:
            raise ValueError('P20_PUBLICATION_READ_BOUND')
        buffer = ctypes.create_string_buffer(count)
        actual = wintypes.DWORD()
        if not self._api.ReadFile(raw, buffer, count, ctypes.byref(actual), None):
            code = self._last_error()
            if code == 997:
                handle.uncertain_buffers.extend((buffer, actual))
            raise self._failure('ReadFile', code)
        if actual.value > count:
            raise ValueError('P20_PUBLICATION_INVALID_READ_COUNT')
        return buffer.raw[:actual.value]

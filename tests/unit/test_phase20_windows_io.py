"""Win32 binding tests with injected functions only; no native qualification."""
from __future__ import annotations

import ctypes
import runpy
import unittest
from ctypes import wintypes
from pathlib import Path

from src.evaluation import phase20_publication as publication
from src.evaluation import phase20_windows_io as windows


_support = runpy.run_path(str(Path(__file__).with_name('test_phase20_publication.py')))


class Function:
    def __init__(self, name, function, owner):
        self.name, self.function, self.owner = name, function, owner

    def __call__(self, *args):
        self.owner.calls.append((self.name, args))
        if self.name == self.owner.fail:
            return 0
        return self.function(*args)


class Kernel32Double:
    def __init__(self, target=r'C:\UnitOnly\output.json'):
        self.target = target
        self.data = None
        self.calls = []
        self.fail = None
        self.error = 5
        self.attributes = 0x80
        self.final_path = '\\\\?\\' + target
        self.handles = {}
        for name, function in {
            'CreateFileW': self.create, 'WriteFile': self.write, 'ReadFile': self.read,
            'FlushFileBuffers': lambda handle: 1, 'CloseHandle': self.close,
            'GetFileInformationByHandleEx': self.info,
            'GetFinalPathNameByHandleW': self.path,
            'GetFileSizeEx': self.length,
        }.items():
            setattr(self, name, Function(name, function, self))

    def create(self, path, access, share, security, disposition, flags, template):
        if disposition == 1:
            if self.data is not None:
                self.error = 80
                return ctypes.c_void_p(-1).value
            self.data = b''
        elif self.data is None:
            self.error = 2
            return ctypes.c_void_p(-1).value
        handle = len(self.calls) + 100
        self.handles[handle] = 0
        return handle

    def write(self, handle, buffer, count, returned, overlapped):
        self.data = ctypes.string_at(buffer, count)
        ctypes.cast(returned, ctypes.POINTER(wintypes.DWORD)).contents.value = count
        return 1

    def read(self, handle, buffer, count, returned, overlapped):
        start = self.handles[handle]
        data = self.data[start:start + count]
        self.handles[handle] += len(data)
        ctypes.memmove(buffer, data, len(data))
        ctypes.cast(returned, ctypes.POINTER(wintypes.DWORD)).contents.value = len(data)
        return 1

    def close(self, handle):
        del self.handles[handle]
        return 1

    def info(self, handle, kind, pointer, size):
        if kind == 18:
            info = ctypes.cast(pointer, ctypes.POINTER(windows._FileIdInfo)).contents
            info.volume = 42
            info.file_id[:] = b'x' * 16
        elif kind == 9:
            ctypes.cast(pointer, ctypes.POINTER(windows._AttributeTagInfo)).contents.attributes = self.attributes
        else:
            raise AssertionError('unexpected information class')
        return 1

    def path(self, handle, buffer, size, flags):
        if len(self.final_path) >= size:
            return len(self.final_path) + 1
        buffer.value = self.final_path
        return len(self.final_path)

    def length(self, handle, pointer):
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_longlong)).contents.value = len(self.data)
        return 1


class WindowsBindingTests(unittest.TestCase):
    def setUp(self):
        self.api = Kernel32Double()
        self.io = windows.WindowsPublicationIO(
            self.api.target, _kernel32=self.api, _last_error=lambda: self.api.error)

    def test_instantiation_binds_signatures_without_calling_functions(self):
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.api.CreateFileW.restype, wintypes.HANDLE)
        self.assertEqual(len(self.api.WriteFile.argtypes), 5)
        self.assertEqual(ctypes.sizeof(windows._FileIdInfo), 24)

    def test_target_cannot_be_rebound_and_pending_handles_are_inspectable(self):
        with self.assertRaises(AttributeError):
            self.io.target = r'C:\Other\output.json'
        handle = self.io.create_new()
        self.assertEqual(self.io.pending_handles, (handle,))
        self.io.close(handle)
        self.assertEqual(self.io.pending_handles, ())

    def test_controller_through_injected_binding_has_expected_native_flags(self):
        frozen = _support['frozen_fixture']()
        result = publication.publish_acceptance_bytes(frozen, self.io, permit=lambda op: True)
        self.assertTrue(result.local_readback_verified)
        self.assertEqual(result.completion_status, 'COMPLETION_UNRESOLVED')
        creates = [args for name, args in self.api.calls if name == 'CreateFileW']
        self.assertEqual(creates[0], (self.api.target, 0xC0000000, 0, None, 1, 0x80, None))
        self.assertEqual(creates[1], (self.api.target, 0x80000000, 1, None, 3, 0x00200080, None))
        self.assertEqual(len([1 for name, _ in self.api.calls if name == 'WriteFile']), 1)
        self.assertEqual(len([1 for name, _ in self.api.calls if name == 'ReadFile']), 2)
        self.assertEqual(len([1 for name, _ in self.api.calls if name == 'CloseHandle']), 2)
        self.assertEqual(self.api.data, frozen.payload)
        self.assertEqual(self.api.handles, {})
        # Identity includes three native queries, rather than one abstract Call.
        self.assertEqual(len(self.api.calls), 15)

    def test_write_attempt_is_consumed_on_success_short_error_and_interruption(self):
        for outcome in ('success', 'short', 'error', 'interrupted'):
            with self.subTest(outcome=outcome):
                api = Kernel32Double()
                io = windows.WindowsPublicationIO(api.target, _kernel32=api,
                                                  _last_error=lambda: api.error)
                handle = io.create_new()
                if outcome == 'short':
                    def short(raw, buffer, count, returned, overlapped):
                        api.data = ctypes.string_at(buffer, 1)
                        ctypes.cast(returned, ctypes.POINTER(wintypes.DWORD)).contents.value = 1
                        return 1
                    api.WriteFile.function = short
                elif outcome == 'error':
                    api.fail = 'WriteFile'
                elif outcome == 'interrupted':
                    def interrupted(*args):
                        api.data = b'partial'
                        raise KeyboardInterrupt()
                    api.WriteFile.function = interrupted
                if outcome == 'error':
                    with self.assertRaises(OSError):
                        io.write_once(handle, b'payload')
                elif outcome == 'interrupted':
                    with self.assertRaises(KeyboardInterrupt):
                        io.write_once(handle, b'payload')
                else:
                    io.write_once(handle, b'payload')
                saved = api.data
                with self.assertRaises(ValueError):
                    io.write_once(handle, b'repair')
                self.assertEqual(api.data, saved)
                self.assertEqual(sum(name == 'WriteFile' for name, _ in api.calls), 1)
                if outcome != 'success':
                    with self.assertRaises(ValueError):
                        io.flush(handle)
                    self.assertNotIn('FlushFileBuffers', [name for name, _ in api.calls])

    def test_flush_requires_full_write_and_cannot_be_retried(self):
        handle = self.io.create_new()
        with self.assertRaises(ValueError):
            self.io.flush(handle)
        self.io.write_once(handle, b'payload')
        self.io.flush(handle)
        with self.assertRaises(ValueError):
            self.io.flush(handle)
        self.assertEqual(sum(name == 'FlushFileBuffers' for name, _ in self.api.calls), 1)
        self.assertEqual(self.api.data, b'payload')

    def test_failed_flush_cannot_be_retried_or_followed_by_repair_write(self):
        handle = self.io.create_new()
        self.io.write_once(handle, b'payload')
        self.api.fail = 'FlushFileBuffers'
        with self.assertRaises(OSError):
            self.io.flush(handle)
        self.api.fail = None
        with self.assertRaises(ValueError):
            self.io.flush(handle)
        with self.assertRaises(ValueError):
            self.io.write_once(handle, b'repair')
        self.assertEqual(sum(name == 'FlushFileBuffers' for name, _ in self.api.calls), 1)
        self.assertEqual(self.api.data, b'payload')
        self.io.close(handle)

    def test_foreign_handle_and_wrong_access_mode_are_rejected_before_native_call(self):
        handle = self.io.create_new()
        other = windows.WindowsPublicationIO(self.api.target, _kernel32=self.api,
                                             _last_error=lambda: self.api.error)
        before = len(self.api.calls)
        for operation in (lambda: other.close(handle), lambda: self.io.read(handle, 1),
                          lambda: self.io.length(handle)):
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(len(self.api.calls), before)
        self.io.close(handle)

    def test_failed_close_is_quarantined_and_never_retried(self):
        handle = self.io.create_new()
        self.api.fail = 'CloseHandle'
        with self.assertRaises(OSError) as caught:
            self.io.close(handle)
        self.assertEqual(caught.exception.winerror, 5)
        self.api.fail = None
        with self.assertRaises(ValueError):
            self.io.close(handle)
        self.assertEqual(sum(name == 'CloseHandle' for name, _ in self.api.calls), 1)

    def test_uncertain_close_quarantines_reopen_and_new_creation(self):
        handle = self.io.create_new()
        self.api.fail = 'CloseHandle'
        with self.assertRaises(OSError):
            self.io.close(handle)
        before = len(self.api.calls)
        for operation in (self.io.create_new, self.io.reopen_read):
            with self.assertRaises(ValueError):
                operation()
        self.assertEqual(len(self.api.calls), before)
        self.assertIn(handle, self.io.pending_handles)

    def test_successful_close_cannot_be_reused(self):
        handle = self.io.create_new()
        self.io.close(handle)
        before = len(self.api.calls)
        with self.assertRaises(ValueError):
            self.io.close(handle)
        with self.assertRaises(ValueError):
            self.io.write_once(handle, b'x')
        self.assertEqual(len(self.api.calls), before)

    def test_collision_is_not_replaced_or_truncated(self):
        self.api.data = b'partial'
        with self.assertRaises(OSError) as caught:
            self.io.create_new()
        self.assertEqual(caught.exception.winerror, 80)
        self.assertEqual(self.api.data, b'partial')
        self.assertEqual([name for name, _ in self.api.calls], ['CreateFileW'])

    def test_open_is_not_retried_after_refusal(self):
        self.api.data = b'partial'
        with self.assertRaises(OSError):
            self.io.create_new()
        with self.assertRaises(ValueError):
            self.io.create_new()
        self.assertEqual([name for name, _ in self.api.calls], ['CreateFileW'])

    def test_outstanding_handle_blocks_another_open_before_native_call(self):
        handle = self.io.create_new()
        before = len(self.api.calls)
        with self.assertRaises(ValueError):
            self.io.reopen_read()
        self.assertEqual(len(self.api.calls), before)
        self.io.close(handle)

    def test_reparse_directory_and_final_path_mismatch_rejected(self):
        handle = self.io.create_new()
        for attributes in (0x400, 0x10):
            with self.subTest(attributes=attributes):
                self.api.attributes = attributes
                with self.assertRaises(ValueError):
                    self.io.identity(handle)
        self.api.attributes = 0x80
        for path in ('\\\\?\\C:\\Elsewhere\\output.json', 'x' * 600):
            with self.subTest(path=path[:25]):
                self.api.final_path = path
                with self.assertRaises(ValueError):
                    self.io.identity(handle)

    def test_native_errors_retain_the_immediate_code(self):
        for name, operation in (
            ('WriteFile', lambda h: self.io.write_once(h, b'x')),
            ('FlushFileBuffers', lambda h: self.io.flush(h)),
            ('GetFileInformationByHandleEx', lambda h: self.io.identity(h)),
            ('GetFinalPathNameByHandleW', lambda h: self.io.identity(h)),
        ):
            with self.subTest(name=name):
                self.api.data = None
                self.io = windows.WindowsPublicationIO(
                    self.api.target, _kernel32=self.api, _last_error=lambda: self.api.error)
                handle = self.io.create_new()
                if name == 'FlushFileBuffers':
                    self.io.write_once(handle, b'x')
                self.api.fail, self.api.error = name, 123
                with self.assertRaises(OSError) as caught:
                    operation(handle)
                self.assertEqual(caught.exception.winerror, 123)
                self.api.fail = None
                self.io.close(handle)

    def test_unexpected_pending_write_retains_buffers_and_prohibits_close(self):
        handle = self.io.create_new()
        self.api.fail, self.api.error = 'WriteFile', 997
        with self.assertRaises(OSError):
            self.io.write_once(handle, b'x')
        self.assertEqual(len(handle.uncertain_buffers), 2)
        before = len(self.api.calls)
        with self.assertRaises(ValueError):
            self.io.close(handle)
        self.assertEqual(len(self.api.calls), before)

    def test_reader_errors_retain_immediate_codes(self):
        self.api.data = b'x'
        handle = self.io.reopen_read()
        for name, operation in (('ReadFile', lambda: self.io.read(handle, 1)),
                                ('GetFileSizeEx', lambda: self.io.length(handle))):
            with self.subTest(name=name):
                self.api.fail, self.api.error = name, 123
                with self.assertRaises(OSError) as caught:
                    operation()
                self.assertEqual(caught.exception.winerror, 123)
        self.api.fail = None
        self.io.close(handle)

    def test_unexpected_pending_read_retains_buffers_and_prohibits_close(self):
        self.api.data = b'x'
        handle = self.io.reopen_read()
        self.api.fail, self.api.error = 'ReadFile', 997
        with self.assertRaises(OSError):
            self.io.read(handle, 1)
        self.assertEqual(len(handle.uncertain_buffers), 2)
        before = len(self.api.calls)
        with self.assertRaises(ValueError):
            self.io.close(handle)
        self.assertEqual(len(self.api.calls), before)

    def test_write_and_read_bounds_reject_without_native_call(self):
        handle = self.io.create_new()
        before = len(self.api.calls)
        for data in (b'', b'x' * 65537, bytearray(b'x')):
            with self.assertRaises(ValueError):
                self.io.write_once(handle, data)
        self.assertEqual(len(self.api.calls), before)
        self.io.close(handle)
        handle = self.io.reopen_read()
        before = len(self.api.calls)
        for count in (0, -1, 65537, True):
            with self.assertRaises(ValueError):
                self.io.read(handle, count)
        self.assertEqual(len(self.api.calls), before)

    def test_unsafe_paths_rejected_without_binding_or_opening(self):
        paths = ['relative.json', r'\\server\share\out.json', r'\\?\C:\out.json',
                 r'C:\a\..\out.json', r'C:\a\.\out.json', r'C:\a\CON.json',
                 r'C:\a\lpt1', r'C:\a\out.json:stream', 'C:\\a\\out.json ',
                 'C:\\a\\out.json.', r'C:/a/out.json', r'C:\a\*.json',
                 r'C:\a\NUL .txt', r'C:\a\CONOUT$', r'C:\a\FOO~1\out.json',
                 'C:\\a\\out\x00.json', 'C:\\' + 'x' * 510,
                 'C:\\' + '\\'.join(['a'] * 33), 'C:\\a\\']
        for path in paths:
            with self.subTest(path=repr(path)):
                with self.assertRaises(ValueError):
                    windows.WindowsPublicationIO(path, _kernel32=self.api,
                                                  _last_error=lambda: 5)
        self.assertEqual(self.api.calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""New in-memory unit cases, not recovered CONTROL witnesses or qualification.

Run with unittest and the existing isolated acceptance-test launcher conventions.
No publication path, historical evidence, or native Windows API is accessed.
"""
from __future__ import annotations

import hashlib
import runpy
import unittest
from dataclasses import replace
from pathlib import Path

from src.evaluation import phase20_local_acceptance as acceptance
from src.evaluation import phase20_publication as publication


_support = runpy.run_path(str(Path(__file__).with_name('test_phase20_local_acceptance.py')))


def frozen_fixture():
    record, context = _support['synthetic_fixture']()
    payload = acceptance.canonical_bytes(record)
    return publication.FrozenAcceptance(payload, hashlib.sha256(payload).hexdigest(), context)


class MemoryIO:
    def __init__(self, *, seed=None, fail=None, short=False, mismatch=None):
        self.data = seed
        self.fail = fail
        self.short = short
        self.mismatch = mismatch
        self.calls = []
        self.handles = {}
        self.cursor = 0
        self.identity_value = publication.FileIdentity(1, b'x' * 16, r'C:\UnitOnly\output.json')

    def _call(self, operation):
        self.calls.append(operation)
        if self.fail == operation:
            raise OSError('injected unit error')

    def create_new(self):
        self._call('CREATE_NEW')
        if self.data is not None:
            raise FileExistsError('unit collision')
        self.data = b''
        handle = object()
        self.handles[handle] = 'writer'
        return handle

    def identity(self, handle):
        writer = self.handles[handle] == 'writer'
        self._call('WRITER_IDENTITY' if writer else 'READER_IDENTITY')
        return replace(self.identity_value, file_id=b'y' * 16) if not writer and self.mismatch == 'identity' else self.identity_value

    def write_once(self, handle, payload):
        self._call('WRITE_FILE')
        self.data = payload[:7] if self.short else payload
        return len(self.data)

    def flush(self, handle):
        self._call('FLUSH_FILE_BUFFERS')

    def close(self, handle):
        self._call('CLOSE_WRITER' if self.handles[handle] == 'writer' else 'CLOSE_READER')
        del self.handles[handle]

    def reopen_read(self):
        self._call('REOPEN_READ')
        self.cursor = 0
        handle = object()
        self.handles[handle] = 'reader'
        return handle

    def length(self, handle):
        self._call('QUERY_LENGTH')
        return len(self.data) + (1 if self.mismatch == 'length' else 0)

    def read(self, handle, count):
        self._call('READ_FILE' if self.cursor == 0 else 'READ_EOF')
        data = self.data[self.cursor:self.cursor + count]
        self.cursor += len(data)
        if self.mismatch == 'bytes' and data:
            return b'!' + data[1:]
        if self.mismatch == 'eof' and not data:
            return b'!'
        return data


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.frozen = frozen_fixture()

    def publish(self, io, **kwargs):
        return publication.publish_acceptance_bytes(
            self.frozen, io, permit=kwargs.pop('permit', lambda operation: True), **kwargs)

    def test_local_success_is_never_verified_complete(self):
        io = MemoryIO()
        result = self.publish(io)
        self.assertTrue(result.local_readback_verified)
        self.assertTrue(result.release_confirmed)
        self.assertEqual(result.completion_status, 'COMPLETION_UNRESOLVED')
        self.assertEqual(result.errors, ())
        self.assertEqual(io.data, self.frozen.payload)
        self.assertEqual(io.calls, ['CREATE_NEW', 'WRITER_IDENTITY', 'WRITE_FILE',
            'FLUSH_FILE_BUFFERS', 'CLOSE_WRITER', 'REOPEN_READ', 'READER_IDENTITY',
            'QUERY_LENGTH', 'READ_FILE', 'READ_EOF', 'CLOSE_READER'])
        self.assertEqual(io.handles, {})

    def test_complete_and_partial_existing_entries_are_never_reused(self):
        for seed in (self.frozen.payload, self.frozen.payload[:9], b''):
            with self.subTest(seed_length=len(seed)):
                io = MemoryIO(seed=seed)
                result = self.publish(io)
                self.assertFalse(result.local_readback_verified)
                self.assertEqual(io.calls, ['CREATE_NEW'])
                self.assertEqual(io.data, seed)

    def test_short_write_is_preserved_without_flush_or_retry(self):
        io = MemoryIO(short=True)
        result = self.publish(io, release_permit=lambda: True)
        self.assertEqual(result.returned_count, 7)
        self.assertEqual(io.data, self.frozen.payload[:7])
        self.assertEqual(io.calls, ['CREATE_NEW', 'WRITER_IDENTITY', 'WRITE_FILE', 'CLOSE_WRITER'])
        self.assertTrue(result.release_confirmed)
        self.assertFalse(result.local_readback_verified)

    def test_fault_at_every_io_stage_stops_ordinary_work(self):
        stages = ['CREATE_NEW', 'WRITER_IDENTITY', 'WRITE_FILE', 'FLUSH_FILE_BUFFERS',
                  'CLOSE_WRITER', 'REOPEN_READ', 'READER_IDENTITY', 'QUERY_LENGTH',
                  'READ_FILE', 'READ_EOF', 'CLOSE_READER']
        for stage in stages:
            with self.subTest(stage=stage):
                io = MemoryIO(fail=stage)
                result = self.publish(io, release_permit=lambda: True)
                self.assertFalse(result.local_readback_verified)
                self.assertEqual(result.completion_status, 'COMPLETION_UNRESOLVED')
                self.assertEqual(result.last_operation, stage)
                self.assertEqual(io.calls.count(stage), 1)
                index = stages.index(stage)
                release = ([] if stage in ('CREATE_NEW', 'CLOSE_WRITER', 'REOPEN_READ', 'CLOSE_READER')
                           else ['CLOSE_WRITER' if index < 4 else 'CLOSE_READER'])
                self.assertEqual(io.calls, stages[:index + 1] + release)
                self.assertTrue(result.errors)
                if stage in ('CLOSE_WRITER', 'CLOSE_READER', 'CREATE_NEW', 'REOPEN_READ'):
                    self.assertFalse(result.release_confirmed)
                if stage == 'WRITE_FILE':
                    self.assertIsNone(result.returned_count)

    def test_permission_loss_at_every_boundary_prevents_the_operation(self):
        stages = ['CREATE_NEW', 'WRITER_IDENTITY', 'WRITE_FILE', 'FLUSH_FILE_BUFFERS',
                  'CLOSE_WRITER', 'REOPEN_READ', 'READER_IDENTITY', 'QUERY_LENGTH',
                  'READ_FILE', 'READ_EOF', 'CLOSE_READER', 'FINAL_CHECK']
        for stage in stages:
            with self.subTest(stage=stage):
                io = MemoryIO()
                result = self.publish(io, permit=lambda operation: operation != stage)
                self.assertFalse(result.local_readback_verified)
                self.assertEqual(result.last_operation, stage)
                self.assertNotIn(stage, io.calls)
                self.assertEqual(io.calls, stages[:stages.index(stage)])
                self.assertEqual(result.completion_status, 'COMPLETION_UNRESOLVED')

    def test_abrupt_io_interruption_preserves_exact_prefix_and_release_boundary(self):
        stages = ['CREATE_NEW', 'WRITER_IDENTITY', 'WRITE_FILE', 'FLUSH_FILE_BUFFERS',
                  'CLOSE_WRITER', 'REOPEN_READ', 'READER_IDENTITY', 'QUERY_LENGTH',
                  'READ_FILE', 'READ_EOF', 'CLOSE_READER']

        class InterruptedMemoryIO(MemoryIO):
            def _call(self, operation):
                super()._call(operation)
                if operation == interrupted_stage:
                    raise KeyboardInterrupt('in-memory unit interruption')

        for interrupted_stage in stages:
            for release_allowed in (False, True):
                with self.subTest(stage=interrupted_stage, release_allowed=release_allowed):
                    io = InterruptedMemoryIO()
                    with self.assertRaises(KeyboardInterrupt):
                        self.publish(io, release_permit=lambda: release_allowed)
                    index = stages.index(interrupted_stage)
                    # An unreturned acquisition or attempted close cannot be
                    # released here. Other known handles get at most one close,
                    # only with the separate release permission.
                    no_release = interrupted_stage in (
                        'CREATE_NEW', 'CLOSE_WRITER', 'REOPEN_READ', 'CLOSE_READER')
                    release = ([] if no_release or not release_allowed else
                               ['CLOSE_WRITER' if index < 4 else 'CLOSE_READER'])
                    self.assertEqual(io.calls, stages[:index + 1] + release)
                    expected_data = (None if index == 0 else
                                     b'' if index <= 2 else self.frozen.payload)
                    self.assertEqual(io.data, expected_data)
                    if interrupted_stage in ('CLOSE_WRITER', 'CLOSE_READER'):
                        self.assertEqual(len(io.handles), 1)

    def test_no_cleanup_authority_retains_the_owned_capability(self):
        io = MemoryIO(fail='FLUSH_FILE_BUFFERS')
        result = self.publish(io)
        self.assertFalse(result.release_confirmed)
        self.assertIn(result.pending_handle, io.handles)
        self.assertNotIn('CLOSE_WRITER', io.calls)
        self.assertEqual(io.data, self.frozen.payload)

    def test_separate_release_permission_after_stop_only_closes(self):
        io = MemoryIO()
        result = self.publish(io, permit=lambda op: op != 'WRITE_FILE', release_permit=lambda: True)
        self.assertEqual(io.calls, ['CREATE_NEW', 'WRITER_IDENTITY', 'CLOSE_WRITER'])
        self.assertTrue(result.release_confirmed)
        self.assertEqual(io.data, b'')

    def test_cleanup_failure_and_close_failure_are_never_retried(self):
        io = MemoryIO(fail='CLOSE_WRITER')
        result = self.publish(io, release_permit=lambda: True)
        self.assertEqual(io.calls.count('CLOSE_WRITER'), 1)
        self.assertFalse(result.release_confirmed)
        self.assertIsNotNone(result.pending_handle)
        io = MemoryIO(fail='CLOSE_WRITER')
        result = self.publish(io, permit=lambda op: op != 'WRITE_FILE', release_permit=lambda: True)
        self.assertEqual(io.calls.count('CLOSE_WRITER'), 1)
        self.assertFalse(result.release_confirmed)
        self.assertIn('RELEASE:', result.errors[-1])

    def test_unknown_or_denied_release_permission_leaves_handle_unresolved(self):
        def unknown():
            raise OSError('authority unavailable')
        for permit in (lambda: False, lambda: 1, unknown):
            with self.subTest(permit=permit):
                io = MemoryIO(fail='WRITE_FILE')
                result = self.publish(io, release_permit=permit)
                self.assertFalse(result.release_confirmed)
                self.assertNotIn('CLOSE_WRITER', io.calls)

    def test_all_readback_mismatches_reject_success(self):
        for mismatch in ('identity', 'length', 'bytes', 'eof'):
            with self.subTest(mismatch=mismatch):
                io = MemoryIO(mismatch=mismatch)
                result = self.publish(io, release_permit=lambda: True)
                self.assertFalse(result.local_readback_verified)
                self.assertTrue(result.release_confirmed)
                self.assertEqual(io.data, self.frozen.payload)

    def test_truthy_permission_is_not_permission(self):
        io = MemoryIO()
        result = self.publish(io, permit=lambda operation: 1)
        self.assertFalse(result.local_readback_verified)
        self.assertEqual(io.calls, [])

    def test_gate_exception_stops_without_io(self):
        def uncertain(operation):
            raise OSError('authority unavailable')
        io = MemoryIO()
        result = self.publish(io, permit=uncertain)
        self.assertFalse(result.local_readback_verified)
        self.assertEqual(io.calls, [])

    def test_base_exception_propagates_after_only_permitted_release(self):
        def interrupted(operation):
            if operation == 'WRITE_FILE':
                raise KeyboardInterrupt()
            return True
        io = MemoryIO()
        with self.assertRaises(KeyboardInterrupt):
            self.publish(io, permit=interrupted, release_permit=lambda: True)
        self.assertEqual(io.calls, ['CREATE_NEW', 'WRITER_IDENTITY', 'CLOSE_WRITER'])
        self.assertEqual(io.data, b'')

    def test_input_cannot_be_mutated_into_different_publication(self):
        with self.assertRaises(Exception):
            self.frozen.payload = b'changed'
        io = MemoryIO()
        self.assertTrue(self.publish(io).local_readback_verified)

    def test_bypassed_frozen_payload_is_revalidated_before_any_io(self):
        object.__setattr__(self.frozen, 'payload', self.frozen.payload + b'\n')
        io = MemoryIO()
        with self.assertRaises(ValueError):
            self.publish(io)
        self.assertEqual(io.calls, [])
        self.assertIsNone(io.data)

    def test_bypassed_context_mutation_is_revalidated_before_any_io(self):
        object.__setattr__(self.frozen.context.authorization, 'destination', r'C:\Other\out.json')
        io = MemoryIO()
        with self.assertRaises(ValueError):
            self.publish(io)
        self.assertEqual(io.calls, [])

    def test_bad_digest_noncanonical_and_malformed_input_fail_before_io(self):
        candidates = [(self.frozen.payload, '0' * 64),
                      (self.frozen.payload + b'\n', hashlib.sha256(self.frozen.payload + b'\n').hexdigest()),
                      (b'{"x":1,"x":2}', hashlib.sha256(b'{"x":1,"x":2}').hexdigest())]
        for payload, digest in candidates:
            with self.subTest(payload=payload[:30]):
                with self.assertRaises(ValueError):
                    publication.FrozenAcceptance(payload, digest, self.frozen.context)

    def test_payload_bounds_and_context_are_required(self):
        for payload in (b'', b'x' * (publication.MAX_PAYLOAD_BYTES + 1), bytearray(b'{}')):
            with self.subTest(type=type(payload), size=len(payload)):
                with self.assertRaises(ValueError):
                    publication.FrozenAcceptance(payload, '0' * 64, self.frozen.context)
        with self.assertRaises(ValueError):
            publication.FrozenAcceptance(self.frozen.payload, self.frozen.expected_sha256, None)


if __name__ == '__main__':
    unittest.main(verbosity=2)

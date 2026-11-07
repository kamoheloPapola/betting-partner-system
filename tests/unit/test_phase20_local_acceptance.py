"""Synthetic-only unit tests; standalone unittest execution has no file writes.

Do not use these fixture times, authorization strings, or seals as real records.
The standalone command is documented in the accompanying schema contract.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfoNotFoundError

from src.evaluation import phase20_local_acceptance as a


SYNTHETIC_AUTH_REFERENCE = 'SYNTHETIC-FIXTURE-ONLY-NOT-WRITE-AUTHORIZATION'
SYNTHETIC_AUTH_DIGEST = hashlib.sha256(b'SYNTHETIC FIXTURE ONLY').hexdigest()
ROOT = Path(__file__).resolve().parents[2]


def synthetic_fixture():
    """Independent, explicit contract fixture; no real acceptance is represented."""
    package = 'phase20-local-20261019T174318Z-fbd145f179a742358bbea7076c65112d'
    root = r'C:\Users\papol\my_systems\phase20_evidence_archive' + '\\' + package
    record = {
        'schema_version': 'ffps.phase20_local_preservation_acceptance.v1',
        'decision': {
            'decision_id': 'P20-FORMAL-ACCEPTANCE-SYNTHETIC-ONLY-01',
            'decision': 'ACCEPT_WITH_DOCUMENTED_LIMITATIONS',
            'owner_stated_decision_date': '2026-10-19',
            'authority': {'name': 'Kamohelo Papola', 'windows_account': r'FRIDAY\papol',
                          'basis': 'OWNER_SELF_ATTESTATION',
                          'independent_verification': 'NOT_ESTABLISHED'},
            'write_authorization': {'reference': SYNTHETIC_AUTH_REFERENCE, 'sha256': SYNTHETIC_AUTH_DIGEST},
        },
        'recording': {
            'record_status': 'RECORDED', 'created_at_utc': '2030-04-30T21:59:59Z',
            'timezone': 'Africa/Johannesburg', 'acceptance_date': '2030-04-30',
            'acceptance_date_source': 'ACTUAL_LOCAL_RECORDING_EVENT',
        },
        'package': {
            'identifier': package, 'root': root,
            'manifest_sha256': '3312cd4f868f7bb9bcef26646ce9d971513d546ea332690c498cf7009818d014',
            'payload_file_count': 16, 'payload_bytes': 4552385, 'historical_discrepancy_count': 10,
            'discrepancies_preserved': True, 'discrepancies_resolved': False,
            'discrepancies_restored': False, 'identity_provenance': 'PREVIOUSLY_REPORTED',
            'independently_reverified_for_acceptance': False,
        },
        'verification': {
            'existing_record_reference': root + r'\records\single-operator-verification.json',
            'capture_result': 'CAPTURE_PASS', 'verification_result': 'READ_ONLY_VERIFICATION_PASS',
            'provenance': 'PREVIOUSLY_REPORTED', 'classification': 'SINGLE_OPERATOR_VERIFICATION',
            'independence': 'NOT_ESTABLISHED', 'independent_assurance': 'NOT_ESTABLISHED',
        },
        'limitations': {
            'single_volume_loss_risk': 'ACCEPTED', 'independent_backup': 'NOT_ESTABLISHED; DEFERRED',
            'encryption': 'UNVERIFIED; VERIFICATION DEFERRED',
            'independent_manifest_digest_anchoring': 'NOT_ESTABLISHED; DEFERRED',
            'strong_immutability': 'NOT_ESTABLISHED', 'privileged_local_access_remains': True,
            'disaster_recovery_readiness': 'NOT_ESTABLISHED',
            'independent_verification': 'NOT_ESTABLISHED',
            'acceptance_record_independent_anchoring': 'NOT_ESTABLISHED',
        },
        'retention': {
            'status': 'STARTED', 'trigger': 'FORMAL_ACCEPTANCE_DATE_P', 'calendar_months': 12,
            'missing_day_convention': 'LAST_VALID_DAY_OF_TARGET_MONTH',
            'start': '2030-04-30', 'minimum_end': '2031-04-30', 'review_date': '2031-04-30',
            'automatic_deletion': False,
        },
        'destruction': {
            'minimum_expiry_is_authorization': False, 'investigation_closed_required': True,
            'minimum_retention_elapsed_required': True, 'separate_explicit_decision_required': True,
            'authorized': False, 'authorization_reference': None,
        },
        'boundaries': {
            'package_and_historical_receipts_unchanged': True, 'restoration_authorized': False,
            'scientific_boundary_amendment_authorized': False, 'evidence_deletion_authorized': False,
            'phase20_closure_authorized': False, 'acceptance_is_scientific_clearance': False,
            'phase20_status': 'BLOCKED',
        },
        'record_sha256': None,
    }
    context = a.RecordingContext(
        '2030-04-30T21:59:59Z', '2030-04-30T22:00:00Z',
        a.WriteAuthorization(SYNTHETIC_AUTH_REFERENCE, SYNTHETIC_AUTH_DIGEST,
                             r'C:\Users\papol\my_systems\betting_partner_system\docs\phase20-local-preservation-acceptance-v1.json',
                             'EXCLUSIVE_CREATE_ACCEPTANCE_RECORD',
                             'ffps.phase20_local_preservation_acceptance.v1', '2026-10-19'),
        a.UniquenessCheck(frozenset(), 'PHASE20_LOCAL_PRESERVATION_ACCEPTANCE_DECISIONS',
                          True, 'SYNTHETIC-COMPLETE-REGISTRY-SNAPSHOT'),
    )
    return synthetic_seal(record), context


def synthetic_seal(record):
    """Only test fixtures are sealed; this helper is never a production writer."""
    record = copy.deepcopy(record)
    record['record_sha256'] = a.internal_record_sha256(record)
    return record


def synthetic_draft():
    record, _ = synthetic_fixture()
    record['decision']['write_authorization'] = None
    record['recording'].update(record_status='DRAFT_NOT_RECORDED', created_at_utc=None, acceptance_date=None)
    record['retention'].update(status='NOT_STARTED', start=None, minimum_end=None, review_date=None)
    record['record_sha256'] = None
    return record


def changed(record, path, value=None, *, delete=False, reseal=True):
    record = copy.deepcopy(record)
    target = record
    for key in path[:-1]:
        target = target[key]
    if delete:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return synthetic_seal(record) if reseal and path != ('record_sha256',) else record


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.record, self.context = synthetic_fixture()

    def assert_invalid(self, record, code, context=None):
        result = a.validate_record(record, context=self.context if context is None else context)
        self.assertFalse(result.valid)
        self.assertEqual(result.errors, ('P20_ACCEPTANCE_' + code,))
        self.assertIsNone(result.record_status)

    def test_valid_synthetic_record_does_not_mutate_inputs(self):
        before = copy.deepcopy(self.record)
        result = a.validate_record(self.record, context=self.context)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.record_status, 'RECORDED')
        self.assertEqual(result.self_seal, 'PASS')
        self.assertEqual(result.uniqueness, 'WITHIN_CALLER_SUPPLIED_SCOPE')
        self.assertEqual(result.independent_anchoring, 'NOT_ESTABLISHED')
        self.assertEqual(self.record, before)

    def test_draft_is_not_a_completed_acceptance(self):
        draft = synthetic_draft()
        before = copy.deepcopy(draft)
        result = a.validate_record(draft)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.record_status, 'DRAFT_NOT_RECORDED')
        self.assertEqual(result.self_seal, 'NOT_APPLICABLE')
        self.assertEqual(result.uniqueness, 'NOT_ESTABLISHED')
        self.assertEqual(draft, before)
        self.assertEqual(draft['retention']['status'], 'NOT_STARTED')

    def test_draft_rejects_each_completed_field(self):
        for path, value in [
            (('recording', 'created_at_utc'), '2030-04-30T21:59:59Z'),
            (('recording', 'acceptance_date'), '2030-04-30'),
            (('retention', 'start'), '2030-04-30'),
            (('retention', 'minimum_end'), '2031-04-30'),
            (('retention', 'review_date'), '2031-04-30'),
            (('record_sha256',), '0' * 64),
            (('decision', 'write_authorization'), self.record['decision']['write_authorization']),
        ]:
            with self.subTest(path=path):
                result = a.validate_record(changed(synthetic_draft(), path, value, reseal=False))
                self.assertFalse(result.valid)

    def test_draft_rejects_recording_context(self):
        self.assert_invalid(synthetic_draft(), 'DRAFT_RECORDING_CONTEXT')

    def test_recorded_input_requires_external_event(self):
        result = a.validate_record(self.record)
        self.assertEqual(result.errors, ('P20_ACCEPTANCE_RECORDING_CONTEXT_REQUIRED',))

    def test_authorization_scope_is_exact_without_path_normalization(self):
        auth = self.context.authorization
        for field, value in [('destination', auth.destination.upper()),
                             ('destination', auth.destination.replace('\\', '/')),
                             ('operation', 'OVERWRITE'), ('schema_version', 'other')]:
            with self.subTest(field=field, value=value):
                context = replace(self.context, authorization=replace(auth, **{field: value}))
                self.assert_invalid(self.record, 'AUTHORIZATION_SCOPE', context)

    def test_authorization_cannot_be_self_supplied_by_record(self):
        self.assert_invalid(self.record, 'WRITE_AUTHORIZATION_REQUIRED',
                            replace(self.context, authorization=None))
        for key in ('reference', 'sha256'):
            value = 'WRONG' if key == 'reference' else '0' * 64
            self.assert_invalid(changed(self.record, ('decision', 'write_authorization', key), value),
                                'AUTHORIZATION_MISMATCH')

    def test_context_reference_digest_and_owner_date_are_validated(self):
        for field, value, code in [('reference', '', 'AUTHORIZATION_REFERENCE'),
                                   ('sha256', 'XYZ', 'DIGEST_FORMAT:authorization.sha256'),
                                   ('owner_stated_decision_date', '2026-10-20', 'OWNER_DATE_CONTEXT_MISMATCH')]:
            with self.subTest(field=field):
                context = replace(self.context, authorization=replace(self.context.authorization, **{field: value}))
                self.assert_invalid(self.record, code, context)

    def test_uniqueness_requires_complete_scoped_context(self):
        self.assert_invalid(self.record, 'UNIQUENESS_CONTEXT_REQUIRED',
                            replace(self.context, uniqueness=None))
        for field, value in [('complete', False), ('complete', 1), ('scope', 'REPOSITORY'),
                             ('reference', ''), ('existing_ids', set()), ('existing_ids', frozenset({1}))]:
            with self.subTest(field=field, value=value):
                unique = replace(self.context.uniqueness, **{field: value})
                self.assert_invalid(self.record, 'UNIQUENESS_UNRESOLVED', replace(self.context, uniqueness=unique))

    def test_reused_id_rejected_but_other_registered_ids_allowed(self):
        unique = replace(self.context.uniqueness, existing_ids=frozenset({self.record['decision']['decision_id']}))
        self.assert_invalid(self.record, 'DECISION_ID_REUSED', replace(self.context, uniqueness=unique))
        unique = replace(unique, existing_ids=frozenset({'P20-FORMAL-ACCEPTANCE-OTHER'}))
        self.assertTrue(a.validate_record(self.record, context=replace(self.context, uniqueness=unique)).valid)

    def test_synthetic_recording_dates_and_expected_retention(self):
        cases = [
            ('2028-02-28T22:30:00Z', '2028-02-29', '2029-02-28'),
            ('2027-03-01T10:00:00Z', '2027-03-01', '2028-03-01'),
            ('2030-01-31T12:00:00Z', '2030-01-31', '2031-01-31'),
            ('2030-12-31T22:00:00Z', '2031-01-01', '2032-01-01'),
            ('2030-06-15T00:00:00+00:00', '2030-06-15', '2031-06-15'),
            # Owner-stated date and P may coincide; they still have distinct sources.
            ('2026-10-19T10:00:00Z', '2026-10-19', '2027-10-19'),
        ]
        for timestamp, p, end in cases:
            with self.subTest(timestamp=timestamp):
                record = copy.deepcopy(self.record)
                record['recording'].update(created_at_utc=timestamp, acceptance_date=p)
                record['retention'].update(start=p, minimum_end=end, review_date=end)
                context = replace(self.context, event_utc=timestamp, observed_at_utc=timestamp)
                self.assertTrue(a.validate_record(synthetic_seal(record), context=context).valid)

    def test_calendar_month_end_and_invalid_inputs(self):
        for start, months, end in [('2028-02-29', 12, '2029-02-28'),
                                   ('2030-01-31', 1, '2030-02-28'),
                                   ('2028-01-31', 1, '2028-02-29'),
                                   ('2030-03-31', 1, '2030-04-30'),
                                   ('2030-08-31', 1, '2030-09-30')]:
            self.assertEqual(a.add_calendar_months(date.fromisoformat(start), months).isoformat(), end)
        for start, months in [(date(2030, 1, 1), True), (date(2030, 1, 1), -1),
                              (datetime(2030, 1, 1), 12), ('2030-01-01', 12)]:
            with self.assertRaisesRegex(a.AcceptanceValidationError, 'CALENDAR_INPUT'):
                a.add_calendar_months(start, months)
        with self.assertRaisesRegex(a.AcceptanceValidationError, 'CALENDAR_RANGE'):
            a.add_calendar_months(date(9999, 1, 1), 12)

    def test_calendar_retention_is_not_365_days(self):
        record = copy.deepcopy(self.record)
        record['recording'].update(created_at_utc='2027-03-01T10:00:00Z', acceptance_date='2027-03-01')
        wrong = (date(2027, 3, 1) + timedelta(days=365)).isoformat()
        self.assertEqual(wrong, '2028-02-29')
        record['retention'].update(start='2027-03-01', minimum_end=wrong, review_date=wrong)
        context = replace(self.context, event_utc='2027-03-01T10:00:00Z', observed_at_utc='2027-03-01T10:00:00Z')
        self.assert_invalid(synthetic_seal(record), 'CALENDAR_END', context)

    def test_retention_and_review_dates_must_each_match(self):
        for field, value, code in [('start', '2030-04-29', 'RETENTION_START'),
                                   ('minimum_end', '2031-04-29', 'CALENDAR_END'),
                                   ('review_date', '2031-05-01', 'REVIEW_DATE')]:
            with self.subTest(field=field):
                self.assert_invalid(changed(self.record, ('retention', field), value), code)

    def test_leap_day_roll_forward_and_utc_date_substitution_rejected(self):
        record = copy.deepcopy(self.record)
        record['recording'].update(created_at_utc='2028-02-28T22:30:00Z', acceptance_date='2028-02-29')
        record['retention'].update(start='2028-02-29', minimum_end='2029-03-01', review_date='2029-03-01')
        context = replace(self.context, event_utc='2028-02-28T22:30:00Z', observed_at_utc='2028-02-28T22:30:00Z')
        self.assert_invalid(synthetic_seal(record), 'CALENDAR_END', context)
        record['recording']['acceptance_date'] = '2028-02-28'
        self.assert_invalid(synthetic_seal(record), 'P_RECORDING_LOCAL_DATE', context)

    def test_decision_identifier_and_owner_date_are_not_unchecked_strings(self):
        for ident in ['', 'looks-unique', 'P20-FORMAL-ACCEPTANCE-' + 'A' * 98]:
            with self.subTest(ident=ident):
                self.assert_invalid(changed(self.record, ('decision', 'decision_id'), ident), 'DECISION_ID_FORMAT')
        self.assert_invalid(changed(self.record, ('decision', 'owner_stated_decision_date'), '2026-10-20'),
                            'OWNER_DATE_CONTEXT_MISMATCH')
        self.assert_invalid(changed(self.record, ('decision', 'owner_stated_decision_date'), '2026-02-30'),
                            'DATE_INVALID')

    def test_backdating_and_recording_clock_mismatch(self):
        self.assert_invalid(changed(self.record, ('recording', 'acceptance_date'), '2026-10-19'),
                            'P_RECORDING_LOCAL_DATE')
        self.assert_invalid(changed(self.record, ('recording', 'created_at_utc'), '2026-10-19T10:00:00Z'),
                            'RECORD_EVENT_MISMATCH')
        self.assert_invalid(self.record, 'RECORD_IN_FUTURE',
                            replace(self.context, observed_at_utc='2030-04-29T00:00:00Z'))

    def test_naive_non_utc_and_malformed_dates_rejected(self):
        for timestamp in ['2030-04-30T21:59:59', '2030-04-30T23:59:59+02:00', '', 123]:
            with self.subTest(timestamp=timestamp):
                result = a.validate_record(self.record, context=replace(self.context, event_utc=timestamp))
                self.assertEqual(result.errors, ('P20_ACCEPTANCE_UTC_TIMESTAMP_FORMAT',))
        self.assert_invalid(changed(self.record, ('recording', 'acceptance_date'), '2030-02-30'), 'DATE_INVALID')
        self.assert_invalid(changed(self.record, ('recording', 'acceptance_date'), '2030-4-30'), 'DATE_FORMAT')

    def test_timezone_unavailable_fails_closed(self):
        with patch.object(a, 'ZoneInfo', side_effect=ZoneInfoNotFoundError('synthetic missing tzdata')):
            self.assert_invalid(self.record, 'TIMEZONE_UNAVAILABLE')

    def test_stale_invalid_and_missing_self_seals(self):
        self.assert_invalid(changed(self.record, ('record_sha256',), '0' * 64), 'SELF_SEAL_INVALID')
        self.assert_invalid(changed(self.record, ('record_sha256',), 'ABC'), 'DIGEST_FORMAT:record_sha256')
        self.assert_invalid(changed(self.record, ('decision', 'decision_id'),
                                    'P20-FORMAL-ACCEPTANCE-SYNTHETIC-ONLY-02', reseal=False), 'SELF_SEAL_INVALID')

    def test_seal_excludes_only_top_level_seal_and_does_not_mutate(self):
        record = copy.deepcopy(self.record)
        before = copy.deepcopy(record)
        self.assertEqual(a.internal_record_sha256(record), record['record_sha256'])
        self.assertEqual(record, before)
        record['record_sha256'] = 'ignored'
        self.assertEqual(a.internal_record_sha256(record), before['record_sha256'])
        record['decision']['record_sha256'] = 'included'
        self.assertNotEqual(a.internal_record_sha256(record), before['record_sha256'])
        self.assert_invalid(synthetic_seal(record), 'FIELDS:decision')

    def test_external_digest_is_distinct_and_optional_check_is_explicit(self):
        raw = a.canonical_bytes(self.record)
        expected = hashlib.sha256(raw).hexdigest()
        self.assertNotEqual(expected, self.record['record_sha256'])
        result = a.validate_record_bytes(raw, context=self.context, expected_whole_file_sha256=expected,
                                         require_external_digest=True)
        self.assertTrue(result.valid, result.errors)
        self.assertEqual(result.external_digest, 'PASS')
        self.assertEqual(result.independent_anchoring, 'NOT_ESTABLISHED')
        self.assertEqual(a.validate_record_bytes(raw, context=self.context).external_digest, 'NOT_CHECKED')
        for expected, code in [('0' * 64, 'EXTERNAL_DIGEST_MISMATCH'),
                               (self.record['record_sha256'], 'EXTERNAL_DIGEST_MISMATCH'),
                               ('XYZ', 'DIGEST_FORMAT:external_whole_file_sha256'),
                               (None, 'EXTERNAL_DIGEST_REQUIRED')]:
            with self.subTest(expected=expected):
                result = a.validate_record_bytes(raw, context=self.context,
                                                 expected_whole_file_sha256=expected,
                                                 require_external_digest=True)
                self.assertEqual(result.errors, ('P20_ACCEPTANCE_' + code,))
        self.assertFalse(a.validate_record_bytes(raw, context=self.context, require_external_digest=1).valid)

    def test_whitespace_changes_whole_file_digest_but_not_self_seal(self):
        raw = a.canonical_bytes(self.record)
        pretty = json.dumps(self.record, indent=2).encode('utf-8')
        result = a.validate_record_bytes(pretty, context=self.context)
        self.assertTrue(result.valid)
        self.assertEqual(result.self_seal, 'PASS')
        result = a.validate_record_bytes(pretty, context=self.context,
                                         expected_whole_file_sha256=hashlib.sha256(raw).hexdigest())
        self.assertEqual(result.errors, ('P20_ACCEPTANCE_EXTERNAL_DIGEST_MISMATCH',))

    def test_json_parser_rejects_duplicates_malformed_nonfinite_and_nonobjects(self):
        cases = [(b'{"x":1,"x":2}', 'DUPLICATE_JSON_KEY:x'),
                 (b'{"x":{"y":1,"y":2}}', 'DUPLICATE_JSON_KEY:y'),
                 (b'{"x":NaN}', 'NONFINITE_JSON:NaN'),
                 (b'{"x":Infinity}', 'NONFINITE_JSON:Infinity'),
                 (b'\xff', 'MALFORMED_JSON'), (b'\xef\xbb\xbf{}', 'MALFORMED_JSON'),
                 (b'{"x":}', 'MALFORMED_JSON'), (b'[]', 'ROOT_OBJECT_REQUIRED'),
                 ('{}', 'BYTES_REQUIRED')]
        for raw, code in cases:
            with self.subTest(raw=raw):
                result = a.validate_record_bytes(raw, context=self.context)
                self.assertEqual(result.errors, ('P20_ACCEPTANCE_' + code,))
        raw = a.canonical_bytes(self.record)
        duplicate = b'{"schema_version":"other",' + raw[1:]
        self.assertEqual(a.validate_record_bytes(duplicate, context=self.context).errors,
                         ('P20_ACCEPTANCE_DUPLICATE_JSON_KEY:schema_version',))

    def test_native_types_and_nonfinite_coercion_are_rejected(self):
        for path, value in [(('package', 'payload_file_count'), '16'),
                             (('package', 'payload_file_count'), 16.0),
                             (('package', 'payload_file_count'), True),
                             (('retention', 'calendar_months'), '12'),
                             (('retention', 'automatic_deletion'), 'false'),
                             (('retention', 'automatic_deletion'), 0),
                             (('retention', 'automatic_deletion'), None),
                             (('destruction', 'authorization_reference'), 'separate authorization')]:
            with self.subTest(path=path, value=value):
                self.assert_invalid(changed(self.record, path, value), 'TYPE:' + '.'.join(path))
        for value in [float('nan'), float('inf'), float('-inf')]:
            with self.assertRaisesRegex(a.AcceptanceValidationError, 'NONFINITE_JSON'):
                a.canonical_bytes({'x': value})
        for value in [{1: 'key'}, {1, 2}, ('tuple',), Path('not-coerced')]:
            with self.assertRaises(a.AcceptanceValidationError):
                a.canonical_bytes(value)

    def test_destruction_truth_table_never_changes_record_authority(self):
        for index, values in enumerate(itertools.product((False, True), repeat=3)):
            with self.subTest(values=values):
                allowed = a.destruction_prerequisites_met(
                    investigation_closed=values[0], minimum_retention_elapsed=values[1],
                    separate_explicit_decision=values[2])
                self.assertEqual(allowed, index == 7)
                self.assertFalse(self.record['destruction']['authorized'])
        with self.assertRaisesRegex(a.AcceptanceValidationError, 'DESTRUCTION_BOOL_REQUIRED'):
            a.destruction_prerequisites_met(investigation_closed=True,
                                            minimum_retention_elapsed=True,
                                            separate_explicit_decision='authorized')

    def test_canonicalization_matches_exact_repository_functions_on_native_subset(self):
        source = (ROOT / 'src/predictions/provenance.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        names = {'canonical_json', 'canonical_bytes', 'sha256_bytes', 'sha256_json'}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual(len(nodes), 4)
        # Approved native inputs need no third-party coercions from _json_safe.
        namespace = {'json': json, 'hashlib': hashlib, 'Any': Any, '_json_safe': lambda value: value}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), '<repository-canonical-functions>', 'exec'), namespace)
        for value in [self.record, synthetic_draft(), {'z': [None, True, 1, 1.5], 'a': 'é'}]:
            self.assertEqual(a.canonical_bytes(value), namespace['canonical_bytes'](value))
        self.assertEqual(a.canonical_bytes({'z': 1, 'a': 'é'}), '{"a":"é","z":1}'.encode('utf-8'))
        reversed_record = dict(reversed(list(self.record.items())))
        self.assertEqual(a.internal_record_sha256(reversed_record), self.record['record_sha256'])

    def test_validator_has_no_writer_clock_or_scientific_import(self):
        tree = ast.parse((ROOT / 'src/evaluation/phase20_local_acceptance.py').read_text(encoding='utf-8'))
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse(any(name and name.startswith('src') for name in imports))
        forbidden = {'open', 'write_text', 'write_bytes', 'mkdir', 'unlink', 'rename', 'replace',
                     'persist_new', 'now', 'today', 'utcnow', 'time', 'touch', 'chmod'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
                # str.replace is the sole permitted replace: pure ISO timestamp parsing.
                if name == 'replace':
                    self.assertEqual(ast.unparse(node), "value.replace('Z', '+00:00')")
                else:
                    self.assertNotIn(name, forbidden)


def _paths(value, prefix=()):
    for key, child in value.items():
        path = prefix + (key,)
        yield path, child
        if type(child) is dict:
            yield from _paths(child, path)


def _missing_test(path):
    def test(self):
        record = changed(self.record, path, delete=True)
        if path == ('recording',):
            code = 'FIELDS:recording'
        elif path == ('recording', 'record_status'):
            code = 'RECORD_STATUS'
        else:
            code = 'FIELDS' + (':' + '.'.join(path[:-1]) if len(path) > 1 else '')
        self.assert_invalid(record, code)
    return test


def _constant_test(path, value):
    def test(self):
        replacement = not value if type(value) is bool else value + 1 if type(value) is int else 'UNAUTHORIZED_CHANGE'
        code = 'VALUE:' + '.'.join(path)
        if path == ('recording', 'record_status'):
            code = 'RECORD_STATUS'
        self.assert_invalid(changed(self.record, path, replacement), code)
    return test


def _unknown_test(path):
    def test(self):
        self.assert_invalid(changed(self.record, path + ('unexpected',), False),
                            'FIELDS' + (':' + '.'.join(path) if path else ''))
    return test


# Each generated method is a separate discoverable unittest, with its path in
# the name. Semantic mutations are resealed so rejection cannot rely on a stale seal.
_fixture, _ = synthetic_fixture()
_variable = {
    ('decision', 'decision_id'), ('decision', 'owner_stated_decision_date'),
    ('decision', 'write_authorization', 'reference'), ('decision', 'write_authorization', 'sha256'),
    ('recording', 'created_at_utc'), ('recording', 'acceptance_date'),
    ('retention', 'start'), ('retention', 'minimum_end'), ('retention', 'review_date'), ('record_sha256',),
}
setattr(AcceptanceTests, 'test_reject_unknown_root', _unknown_test(()))
for _path, _value in _paths(_fixture):
    _suffix = '_'.join(_path)
    setattr(AcceptanceTests, 'test_reject_missing_' + _suffix, _missing_test(_path))
    if type(_value) is dict:
        setattr(AcceptanceTests, 'test_reject_unknown_' + _suffix, _unknown_test(_path))
    elif _path not in _variable and _value is not None:
        setattr(AcceptanceTests, 'test_reject_changed_' + _suffix, _constant_test(_path, _value))


if __name__ == '__main__':
    unittest.main(verbosity=2)

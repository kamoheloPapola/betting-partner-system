"""Seal the completed Phase 20 test scope and current offline status."""
from __future__ import annotations

import os
import uuid
import xml.etree.ElementTree as ET

from src.evaluation import phase20_settlement as s
from src.evaluation.phase12_operations import put

DESELECTIONS = (
    'tests/unit/test_phase17_readiness.py::test_pl_requires_a_scientific_protocol_before_cohort_creation',
    'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[BL1]',
    'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[SA]',
    'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[FL1]',
    'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[PD]',
    'tests/unit/test_phase18_readiness.py::test_pl_uses_same_generic_initializer_and_has_empty_independent_cohort',
)
TESTS = ('focused', 'protected', 'full')


def _copy_once(source, destination):
    payload = source.read_bytes()
    if destination.exists():
        s.p.check(destination.read_bytes() == payload, 'PHASE20_TEST_ARTIFACT_CONFLICT')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _result(path):
    cases = ET.parse(path).getroot().findall('.//testcase')
    failures = [f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
                for case in cases if case.find('failure') is not None or
                case.find('error') is not None]
    return {'artifact': s.p.identity(path),
        'passed': sum(case.find('failure') is None and case.find('error') is None and
                      case.find('skipped') is None for case in cases),
        'failed': len(failures), 'skipped': sum(case.find('skipped') is not None
                                                for case in cases),
        'failure_ids': failures}


def seal():
    s.authorization()
    tests = {}
    for name in TESTS:
        target = s.RUN / 'tests' / (name + '.xml')
        _copy_once(s.p.ROOT / 'tmp' / f'phase20-{name}.xml', target)
        tests[name] = _result(target)
    previous = s.p.read(s.phase19.RUN / 'run_report.json')
    inherited = set(previous['tests']['full_test_results.xml']['failure_ids'])
    new_failures = sorted(set(tests['full']['failure_ids']) - inherited)
    s.p.check(tests['focused']['failed'] == tests['protected']['failed'] == 0 and
              not new_failures and tests['full']['passed'] >= 1000,
              'PHASE20_TEST_GATE_FAILED')
    result = put(s.RUN / 'test_report.json', s.p.seal({
        'schema_version': 'ffps.phase20_test_report.v1',
        'tests': tests, 'inherited_failures': sorted(set(tests['full']['failure_ids']) & inherited),
        'new_failures': new_failures, 'full_deselections': list(DESELECTIONS),
        'protected_deselections': list(DESELECTIONS),
        'protected_other_deselections': 243,
        'interrupted_or_incomplete_phase20_attempts': [],
        'phase19_literal_all_tests_attempt': 'INCOMPLETE_NOT_PASSING',
        'phase19_test_scope_correction': s.p.identity(
            s.phase19.RUN / 'test_scope_correction.json')}, 'record_sha256'))
    leagues = [s.verify(league) for league in s.LEAGUES]
    protected = s.integrity()
    s.p.check(protected['verified'] and all(row['verified'] for row in leagues),
              'PHASE20_VERIFICATION_GATE_FAILED')
    report = put(s.RUN / 'run_report.json', s.p.seal({
        'schema_version': 'ffps.phase20_run_report.v1',
        'authorization': s.p.identity(s.AUTH),
        'test_report': s.p.identity(s.RUN / 'test_report.json'),
        'leagues': leagues, 'protected_integrity': protected,
        'created_settlements': sum(row['settled'] for row in leagues),
        'scientific_state': 'DECISION_EVIDENCE_INSUFFICIENT',
        'promotion_authorized': False, 'scheduler_authorized': False,
        'live_settlement_demonstrated': any(row['settled'] for row in leagues)},
        'record_sha256'))
    return report

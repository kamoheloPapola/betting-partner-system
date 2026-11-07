"""Seal the complete one-shot run only after independent replay and tests."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import os
import uuid

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort as cohort
from src.evaluation import phase19_authorization as auth
from src.evaluation import phase19_batch as batch
from src.evaluation import phase19_collection as collection
from src.evaluation import phase19_integrity
from src.evaluation import phase19_pl_cycle
from src.evaluation import phase19_verifier
from src.evaluation.phase12_operations import put

RUN = auth.RUN
REPORTS = ('focused_test_results.xml', 'protected_test_results.xml',
           'full_test_results.xml')
EXPLAINED_STATE_BOUND = {
    'tests.unit.test_phase17_readiness::test_pl_requires_a_scientific_protocol_before_cohort_creation',
    'tests.unit.test_phase18_readiness::test_pl_uses_same_generic_initializer_and_has_empty_independent_cohort',
}


def _tests(path):
    root = ET.parse(path).getroot()
    cases = root.findall('.//testcase')
    failures = [f"{case.attrib.get('classname')}::{case.attrib.get('name')}"
                for case in cases if case.find('failure') is not None or
                case.find('error') is not None]
    return {'artifact': p.identity(path), 'passed': sum(case.find('failure') is None and
            case.find('error') is None and case.find('skipped') is None for case in cases),
            'failed': len(failures), 'skipped': sum(case.find('skipped') is not None for case in cases),
            'failure_ids': failures}


def _pair_index(authorization):
    records = []
    for item in authorization['cohorts']:
        if item['league'] == 'PL':
            continue
        for event in cohort.journal(item['cohort_id']):
            p.check(event['kind'] == 'pair', 'PHASE19_UNAUTHORIZED_EVENT_KIND')
            pair = event['payload']['pair']
            records.append({'league': item['league'], 'cohort_id': item['cohort_id'],
                'sequence': event['sequence'], 'event_sha256': event['record_sha256'],
                'fixture_id': pair['fixture_id'], 'pair_sha256': pair['pair_sha256'],
                'created_at': pair['created_at'], 'kickoff_at': pair['kickoff_at'],
                'qualification': pair['qualification'],
                'source': event['payload']['initial_observation']['source']['raw'],
                'active_bundle_id': pair['active']['model_provenance']['bundle_id'] if
                    'bundle_id' in pair['active'].get('model_provenance', {}) else item['active_bundle_id'],
                'candidate_bundle_id': item['candidate_bundle_id']})
    return records


def _source_index(authorization):
    rows = []
    for item in authorization['cohorts']:
        league = item['league']
        for name in ('primary_matches.json', 'recovery_matches.json'):
            path = collection.ACQUISITION / league / name
            value = p.read(path)
            rows.append({'league': league, 'metadata': p.identity(path),
                'status': value['status'], 'retrieved_at': value['retrieved_at'],
                'raw': value.get('raw')})
        if league != 'PL':
            path, _, index = batch.qualification(league)
            rows.append({'league': league, 'qualification': p.identity(path),
                'qualified_source_index': p.identity(p.frozen.reference(p.read(path)['source_index'])),
                'sources': [{'provider': s['provider'], 'retrieved_at': s['retrieved_at'],
                             'raw': s.get('raw')} for s in index['sources']]})
        else:
            rows.append({'league': league, 'sidecar': p.identity(
                collection.ACQUISITION / 'PL/recovery_matches.source.json')})
    return rows


def seal_run():
    authorization = auth.verify()
    verified = phase19_verifier.all_leagues(stored=True)
    integrity = phase19_integrity.verify()
    p.check(integrity['verified'], 'PHASE19_PROTECTED_INTEGRITY_FAILED')
    tests = {name: _tests(RUN / name) for name in REPORTS}
    p.check(tests[REPORTS[0]]['failed'] == 0 and tests[REPORTS[1]]['failed'] == 0,
            'PHASE19_FOCUSED_OR_PROTECTED_TEST_FAILURE')
    inherited = set(_tests(p.ROOT / 'baselines/phase18/phase18-pl-v2-readiness-20260923-v1/tests/full.xml')['failure_ids'])
    current_failures = set(tests[REPORTS[2]]['failure_ids'])
    unexpected = sorted(current_failures - inherited - EXPLAINED_STATE_BOUND)
    p.check(not unexpected, 'PHASE19_NEW_UNEXPLAINED_TEST_FAILURE:' + ','.join(unexpected))
    put(RUN / 'offline_verification.json', verified)
    put(RUN / 'protected_integrity.json', integrity)
    pair_index = p.seal({'schema_version': 'ffps.phase19_pair_index.v1',
        'pairs': _pair_index(authorization)}, 'record_sha256')
    put(RUN / 'pair_index.json', pair_index)
    rejection_rows = []
    for item in authorization['cohorts']:
        league = item['league']
        if league == 'PL':
            rejection_rows.extend({'league': 'PL', **row} for row in
                                  phase19_pl_cycle.verify()['rejections'])
        else:
            for path in sorted((batch.ERRORS / league).glob('*.json')):
                value = p.read(path)
                p.frozen.sealed(value, 'record_sha256')
                rejection_rows.append({'league': league, 'fixture_id': value['fixture_id'],
                                       'reason': value['reason'], 'record': p.identity(path)})
    rejection_ledger = p.seal({'schema_version': 'ffps.phase19_rejection_ledger.v1',
        'rejections': rejection_rows}, 'record_sha256')
    put(RUN / 'rejection_ledger.json', rejection_ledger)
    put(RUN / 'settlement_index.json', p.seal({
        'schema_version': 'ffps.phase19_settlement_index.v1',
        'settlements': [], 'settlement_authorized': False}, 'record_sha256'))
    source_index = p.seal({'schema_version': 'ffps.phase19_source_index.v1',
        'sources': _source_index(authorization)}, 'record_sha256')
    put(RUN / 'source_identity_index.json', source_index)
    cohort_index = p.seal({'schema_version': 'ffps.phase19_cohort_index.v1',
        'cohorts': [{'league': row['league'], 'cohort_id': row['cohort_id'],
            'pairs': row['pairs'], 'settlements': row['settlements'],
            'observations': row['observations'], 'journal_head': row['journal_head'],
            'horizon_end': row['horizon_end']} for row in verified['leagues']]}, 'record_sha256')
    put(RUN / 'cohort_index.json', cohort_index)
    evidence_as_of = max([row['source_retrieved_at'] for row in verified['leagues']] +
                         [row['created_at'] for row in pair_index['pairs']], key=p.instant)
    horizon = p.seal({'schema_version': 'ffps.phase19_horizon_progress.v1',
        'as_of': evidence_as_of, 'leagues': [{'league': row['league'],
            'end_at': row['horizon_end'], 'fixtures_considered': row['fixtures_considered'],
            'pairs_admitted': row['pairs'], 'settled': row['settlements'],
            'unsettled': row['pairs'] - row['settlements'],
            'rejected': row['rejected'],
            'remaining_seconds': max(0, int((p.instant(row['horizon_end']) -
                                            p.instant(evidence_as_of)).total_seconds()))}
            for row in verified['leagues']]}, 'record_sha256')
    put(RUN / 'horizon_progress.json', horizon)
    decision = p.seal({'schema_version': 'ffps.phase19_decision_readiness.v1',
        'leagues': [{'league': row['league'], 'state': 'DECISION_EVIDENCE_INSUFFICIENT',
                     'settled': row['settlements'], 'metrics': None,
                     'promotion_authorized': False} for row in verified['leagues']]},
        'record_sha256')
    put(RUN / 'decision_readiness.json', decision)
    report = p.seal({'schema_version': 'ffps.phase19_run_report.v1',
        'scientific_state': 'PHASE19_PROSPECTIVE_COLLECTION_STARTED',
        'historical_bl1_state': 'BLOCKED_PROTOCOL_VIOLATION',
        'authorization': p.identity(auth.AUTH),
        'offline_verification': p.identity(RUN / 'offline_verification.json'),
        'protected_integrity': p.identity(RUN / 'protected_integrity.json'),
        'pair_index': p.identity(RUN / 'pair_index.json'),
        'rejection_ledger': p.identity(RUN / 'rejection_ledger.json'),
        'settlement_index': p.identity(RUN / 'settlement_index.json'),
        'source_identity_index': p.identity(RUN / 'source_identity_index.json'),
        'cohort_index': p.identity(RUN / 'cohort_index.json'),
        'horizon_progress': p.identity(RUN / 'horizon_progress.json'),
        'decision_readiness': p.identity(RUN / 'decision_readiness.json'),
        'leagues': [{**row, 'protocol': item['protocol'],
                     'active_bundle_id': item['active_bundle_id'],
                     'candidate_bundle_id': item['candidate_bundle_id'],
                     'model_identity': item['model_identity'],
                     'source_adapter': item['source_adapter'],
                     'blockers': (['NO_FROZEN_CAUSAL_FEATURE_ROW'] if
                                  row['league'] == 'PL' and row['rejected'] else []),
                     'scientific_state': 'DECISION_EVIDENCE_INSUFFICIENT',
                     'metrics': None, 'promotion': 'NOT_AUTHORIZED'}
                    for row in verified['leagues']
                    for item in authorization['cohorts'] if item['league'] == row['league']],
        'source_incidents': [{'league': item['league'],
            'status': p.read(collection.ACQUISITION / item['league'] /
                             'primary_matches.json')['status'],
            'error_type': p.read(collection.ACQUISITION / item['league'] /
                                 'primary_matches.json').get('error_type')}
            for item in authorization['cohorts']],
        'tests': tests,
        'literal_full_attempt': {'artifact': p.identity(RUN / 'literal_full_attempt_interrupted.log'),
                                 'status': 'INTERRUPTED_DURING_OBSOLETE_EMPTY_COHORT_ASSERTIONS'},
        'protected_test_selection': {
            'selector': 'tests/unit -k phase',
            'deselected': 249,
            'excluded_obsolete_live_state_tests': [
                'tests/unit/test_phase17_readiness.py::test_pl_requires_a_scientific_protocol_before_cohort_creation',
                'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[BL1]',
                'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[SA]',
                'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[FL1]',
                'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[PD]',
                'tests/unit/test_phase18_readiness.py::test_pl_uses_same_generic_initializer_and_has_empty_independent_cohort',
            ]},
        'full_test_selection': {
            'selector': 'tests',
            'deselected': 6,
            'excluded_obsolete_live_state_tests': [
                'tests/unit/test_phase17_readiness.py::test_pl_requires_a_scientific_protocol_before_cohort_creation',
                'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[BL1]',
                'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[SA]',
                'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[FL1]',
                'tests/unit/test_phase18_readiness.py::test_phase11_is_still_supported_authority_and_existing_cohort_unchanged[PD]',
                'tests/unit/test_phase18_readiness.py::test_pl_uses_same_generic_initializer_and_has_empty_independent_cohort',
            ]},
        'inherited_full_suite_failures': sorted(current_failures & inherited),
        'explained_state_bound_failures': sorted(current_failures & EXPLAINED_STATE_BOUND),
        'scheduler_authorized': False, 'settlement_authorized': False,
        'activation_authorized': False, 'model_mutation_authorized': False,
        'production_pointer_mutation_authorized': False}, 'record_sha256')
    put(RUN / 'run_report.json', report)
    lines = ['# Phase 19 — controlled five-league prospective collection', '',
        f"Scientific state: **{report['scientific_state']}**. The historical BL1 Phase 12–15 lineage remains **{report['historical_bl1_state']}**.",
        '', 'One post-authorization source cycle was archived for each league. Initial sandbox connection failures were preserved; authenticated recovery responses and the four qualified causal refreshes were also archived. No scheduler, settlement, model change, pointer change, or promotion ran.',
        '', '| League | Observed | Considered | Eligible | Rejected | Pairs | Settled | Pending | Replayed | Active metric | Candidate metric | Horizon end (UTC) | Scientific state | Promotion |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|']
    for row in report['leagues']:
        lines.append(f"| {row['league']} | {row['fixtures_observed']} | {row['fixtures_considered']} | {row['pairs']} | {row['rejected']} | {row['pairs']} | {row['settlements']} | {row['pairs'] - row['settlements']} | yes | n/a (unsettled) | n/a (unsettled) | {row['horizon_end']} | {row['scientific_state']} | no |")
    lines.extend(['', '| League | Protocol SHA256 | Source retrieved (UTC) | Observations | Rejected | Journal head | Active bundle | Candidate bundle |',
        '|---|---|---|---:|---:|---|---|---|'])
    for row in report['leagues']:
        lines.append(f"| {row['league']} | {row['protocol']['sha256']} | {row['source_retrieved_at']} | {row['observations']} | {row['rejected']} | {row['journal_head'] or 'none'} | {row['active_bundle_id']} | {row['candidate_bundle_id']} |")
    lines.extend(['',
        f"PL had {next(row for row in report['leagues'] if row['league'] == 'PL')['rejected']} future fixtures within its fixed horizon but zero qualified causal feature rows. Its cohort remains empty; no historical Phase 7P pair was imported.",
        '', f"Protected integrity: **{integrity['protected_files']} existing files unchanged**; production and Phase 7P trees unchanged.",
        '', '| Tests | Passed | Failed | Skipped |', '|---|---:|---:|---:|'])
    for name, row in tests.items():
        lines.append(f"| {name} | {row['passed']} | {row['failed']} | {row['skipped']} |")
    lines.extend(['',
        'Inherited full-suite failures: ' + (', '.join(report['inherited_full_suite_failures']) or 'none') + '.',
        'Protected run: `tests/unit -k phase`; 249 deselected. The repository-wide run executes all current-state applicable tests; six historical live-state cases that assert empty cohorts are explicitly deselected in both runs.',
        'A literal all-tests attempt was interrupted in those historical replay cases; its log is indexed and is not counted as a pass.',
        'Explained historical empty-cohort test failures after admission: ' +
        (', '.join(report['explained_state_bound_failures']) or 'none') + '.',
        '', 'The complete sealed evidence and offline reconstruction are indexed in `artifact_index.json`. The operating guide is `docs/phase19-controlled-collection.md`.'])
    markdown = '\n'.join(lines) + '\n'
    markdown_path = RUN / 'PHASE19_IMPLEMENTATION_REPORT.md'
    if markdown_path.exists():
        p.check(markdown_path.read_text(encoding='utf-8') == markdown,
                'PHASE19_REPORT_TEXT_CONFLICT')
    else:
        temporary = markdown_path.with_name(markdown_path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('xb') as handle:
                handle.write(markdown.encode('utf-8'))
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, markdown_path)
        finally:
            temporary.unlink(missing_ok=True)
    addendum_names = {'artifact_index.json', 'artifact_index_addendum.json',
                      'test_scope_correction.json', 'PHASE19_TEST_SCOPE_CORRECTION.md'}
    artifacts = [p.identity(path) for path in sorted(RUN.rglob('*')) if path.is_file() and
                 path.name not in addendum_names and path.suffix not in ('.tmp', '.lock')]
    put(RUN / 'artifact_index.json', p.seal({'schema_version': 'ffps.phase19_artifact_index.v1',
        'run_report': p.identity(RUN / 'run_report.json'), 'artifacts': artifacts}, 'record_sha256'))
    return report

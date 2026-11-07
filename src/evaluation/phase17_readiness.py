"""Phase 17 offline qualification and empty-cohort readiness audit."""
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort, phase16_verifier
from src.evaluation import phase17_pl_adapter as adapter
from src.evaluation.phase12_operations import put

RUN = p.ROOT / 'baselines/phase17/phase17-all-league-v2-readiness-20260923-v1'
PHASE7_SPEC = p.ROOT / 'baselines/phase7/phase7-active-vs-candidate-shadow-79110f9a-20260829-v2/evaluation_specification.json'
PHASE16_INDEX = phase16_cohort.RUN / 'artifact_index.json'
COHORT_IDS = {'BL1': 'BL1-clean-001', 'SA': 'SA-clean-001',
              'FL1': 'FL1-clean-001', 'PD': 'PD-clean-001'}


def _pl_horizon():
    specification = p.read(PHASE7_SPEC)
    eligibility = specification['eligibility']
    p.check(eligibility['league'] == 'PL' and
            specification['specification_fingerprint'] ==
            '272fc86be883dbf43d41b71485236f44cad8975925ef6a97d0877cff442c7775',
            'PL_PHASE7P_SPECIFICATION_DRIFT')
    return (date.fromisoformat(eligibility['fixture_date_max_inclusive']) + timedelta(days=1)).isoformat() + 'T00:00:00+00:00'


def qualification_facts():
    source, rows = adapter.normalize_snapshot()
    second_source, second_rows = adapter.normalize_snapshot()
    p.check(source == second_source and rows == second_rows, 'PL_NORMALIZATION_NONDETERMINISTIC')
    aliases = p.read(adapter.ALIAS)
    p.check(any(row['home_team'] in ('COVENTRY', 'HULL') or row['away_team'] in
                ('COVENTRY', 'HULL') for row in rows), 'PL_ALIAS_EXAMPLES_MISSING')
    p.check(any(row['status'] == 'FINISHED' for row in rows) and
            any(row['status'] == 'TIMED' for row in rows), 'PL_STATUS_EXAMPLES_MISSING')
    return p.seal({'schema_version': 'ffps.phase17_pl_source_qualification.v1',
        'classification': 'PL_ADAPTER_OFFLINE_QUALIFIED',
        'source_lineage': 'Phase7P authorized PL fixture source via Phase7D raw capture and Phase7E exact alias/field authority',
        'source_metadata': p.identity(adapter.SOURCE_2026), 'raw_response': p.identity(adapter.RAW_2026),
        'alias_contract': p.identity(adapter.ALIAS), 'field_authority': p.identity(adapter.AUTHORITY),
        'adapter_implementation': p.identity(p.ROOT / 'src/evaluation/phase17_pl_adapter.py'),
        'phase7p_specification': p.identity(PHASE7_SPEC),
        'retrieved_at': source['retrieved_at'], 'provider': source['provider'],
        'competition': 'PL', 'season': source['season'],
        'fixture_count': len(rows), 'statuses': dict(sorted(Counter(row['status'] for row in rows).items())),
        'normalized_sha256': p.digest(rows),
        'replay_equal': True,
        'archived_kickoff_revision_count': 0,
        'archived_terminal_status_count': sum(row['status'] not in ('TIMED','SCHEDULED','FINISHED') for row in rows),
        'future_versions_require_immutable_raw_archive': True,
        'phase7p_fixed_horizon_end_exclusive': _pl_horizon(),
        'source_authority_extended': False,
        'production_mapping_changed': aliases['production_mapping_changed']}, 'record_sha256')


def qualify_pl():
    facts = qualification_facts()
    return put(RUN / 'pl_source_qualification.json', facts)


def verify_pl():
    stored = p.read(RUN / 'pl_source_qualification.json')
    p.frozen.sealed(stored, 'record_sha256')
    p.check(stored == qualification_facts(), 'PL_QUALIFICATION_RECONSTRUCTION_MISMATCH')
    for field in ('source_metadata', 'raw_response', 'alias_contract', 'field_authority',
                  'adapter_implementation', 'phase7p_specification'):
        p.frozen.reference(stored[field])
    return stored


def status():
    rule = phase16_cohort.load_rule()
    rows = []
    for league, cohort_id in COHORT_IDS.items():
        result = phase16_verifier.verify_cohort(cohort_id)
        spec, protocol = phase16_cohort.binding(cohort_id)
        p.check(result['verified'] and result['league'] == league and
                result['pairs'] == result['settlements'] == 0 and
                result['journal_head'] is None and
                spec['revision_rule_sha256'] == rule['protocol_sha256'] and
                protocol['end_at'] == spec['end_at'], 'PHASE17_NONPL_COHORT_INVALID')
        rows.append({'league': league, 'cohort_id': cohort_id, 'cohort': result['cohort'],
            'protocol': result['protocol'], 'source_qualification': spec['source_qualification'],
            'source_adapter': 'Phase11 qualified current-source path',
            'season': spec['season'], 'start_at': spec['start_at'], 'end_at': spec['end_at'],
            'rule_sha256': spec['revision_rule_sha256'], 'active_bundle_id': spec['active_bundle_id'],
            'candidate_bundle_id': spec['candidate_bundle_id'], 'pairs': 0, 'settlements': 0,
            'journal_head': None, 'state': 'V2_CLEAN_COHORT_READY'})
    pl_qualification = verify_pl()
    pl_cohorts = sorted((phase16_cohort.RUN / 'cohorts').glob('PL-*/cohort_specification.json'))
    p.check(not pl_cohorts, 'UNQUALIFIED_PL_COHORT_PRESENT')
    rows.append({'league': 'PL', 'cohort_id': None, 'cohort': None, 'protocol': None,
        'source_qualification': p.identity(RUN / 'pl_source_qualification.json'),
        'source_adapter': pl_qualification['adapter_implementation'], 'season': 2026,
        'start_at': None, 'end_at': pl_qualification['phase7p_fixed_horizon_end_exclusive'],
        'rule_sha256': rule['protocol_sha256'], 'active_bundle_id': None,
        'candidate_bundle_id': None, 'pairs': 0, 'settlements': 0,
        'journal_head': None, 'state': 'PL_PHASE11_HORIZON_AND_GENERIC_INITIALIZER_UNAVAILABLE'})
    return p.seal({'schema_version': 'ffps.phase17_readiness.v1',
        'scientific_state': 'PHASE17_PARTIAL_READY_PL_PROTOCOL_BLOCKED',
        'historical_bl1_state': 'BLOCKED_PROTOCOL_VIOLATION',
        'v2_rule': p.identity(phase16_cohort.RULE_FILE),
        'v2_rule_sha256': rule['protocol_sha256'],
        'sealed_phase16_index': p.identity(PHASE16_INDEX),
        'leagues': rows, 'pairs_admitted': 0, 'settlements': 0,
        'collection_scheduled': False, 'activation_authorized': False,
        'pl_blocker': 'No PL Phase11 prospective protocol exists. The frozen Phase16 initializer and verifier require a Phase11 protocol and accept only SA/FL1/PD/BL1. Phase7P has a Dec 31 fixed window, but substituting it for the requested Phase11 inherited horizon would author a new scientific contract.'}, 'record_sha256')


def verify_integrity():
    before = p.read(RUN / 'protected_before.json')
    changed = [name for name, sha in before.items() if not (p.ROOT / name).is_file() or
               p.source.sha256_file(p.ROOT / name) != sha]
    p.check(not changed, 'PHASE17_PROTECTED_DRIFT:' + ','.join(changed))
    production = {base:p.source._tree_snapshot(p.ROOT/base)
                  for base in ('data','src/ml/models','baselines/phase7p')}
    p.check(production == p.read(RUN / 'production_before.json'),
            'PHASE17_PRODUCTION_OR_PHASE7P_DRIFT')
    return {'verified': True, 'protected_count': len(before), 'changed': changed,
            'production_and_phase7p_unchanged': True}


def verify():
    computed = status()
    stored = p.read(RUN / 'readiness.json')
    p.frozen.sealed(stored, 'record_sha256')
    p.check(stored == computed, 'PHASE17_READINESS_RECONSTRUCTION_MISMATCH')
    integrity = verify_integrity()
    return {'verified': True, 'scientific_state': computed['scientific_state'],
            'cohorts_ready': 4, 'pl_source_qualified': True,
            'pl_cohort_initialized': False, 'integrity': integrity}


def record_status():
    return put(RUN / 'readiness.json', status())

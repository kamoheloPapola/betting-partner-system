"""Validated horizon authorities for new league-scoped v2 cohorts.

Phase 11 and the explicit PL Phase 18 contract are the only supported authorities.
The Phase 16 rule and its original publisher remain sealed.
"""
from datetime import timedelta

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort, phase17_readiness as prior
from src.evaluation.phase12_operations import put

RUN = p.ROOT / 'baselines/phase18/phase18-pl-v2-readiness-20260923-v1'
PL_CONTRACT = RUN / 'pl_prospective_scientific_contract.json'
PHASE7P_RUN = p.ROOT / 'baselines/phase7p/phase7p-prospective-shadow-79110f9a-20260829-v1'
PHASE7P_REFERENCE = PHASE7P_RUN / 'frozen_specification_reference.json'
PHASE7P_OFFLINE = PHASE7P_RUN / 'offline_verification.json'
PHASE17_OFFLINE = prior.RUN / 'offline_verification.json'
END = '2027-01-01T00:00:00+00:00'


def _frozen_pl_inputs():
    reference = p.read(PHASE7P_REFERENCE)
    p.check(reference['consumed_unchanged'] is True and
            reference['specification_fingerprint'] ==
            '272fc86be883dbf43d41b71485236f44cad8975925ef6a97d0877cff442c7775' and
            reference['file_sha256'] == p.identity(prior.PHASE7_SPEC)['sha256'] and
            reference['path'] == p.identity(prior.PHASE7_SPEC)['path'],
            'PHASE7P_FROZEN_REFERENCE_DRIFT')
    phase7p = p.read(PHASE7P_OFFLINE)
    phase17 = p.read(PHASE17_OFFLINE)
    p.check(phase7p['status'] == 'VERIFIED' and phase7p['errors'] == [] and
            phase17['verified'] is True and
            phase17['state'] == 'PHASE17_PARTIAL_READY_PL_PROTOCOL_BLOCKED' and
            phase17['pl_cohort_initialized'] is False,
            'PREREQUISITE_OFFLINE_VERIFICATION_FAILED')
    qualification = prior.verify_pl()
    spec = p.read(prior.PHASE7_SPEC)
    p.check(prior._pl_horizon() == END and
            qualification['phase7p_fixed_horizon_end_exclusive'] == END and
            spec['eligibility']['fixture_date_min_inclusive'] == '2026-08-30' and
            spec['eligibility']['fixture_date_max_inclusive'] == '2026-12-31',
            'PL_FROZEN_HORIZON_MISMATCH')
    for label in ('active', 'candidate'):
        for member in spec[label]['members'].values():
            p.check(p.source.sha256_file(p.ROOT / member['path']) == member['sha256'],
                    'PL_MODEL_ARTIFACT_DRIFT')
    rule = phase16_cohort.load_rule()
    return spec, qualification, rule


def freeze_pl_contract():
    """Publish a new decision; do not mutate the Phase 7P/16/17 authorities."""
    spec, qualification, rule = _frozen_pl_inputs()
    frozen_at = p.now()
    value = p.seal({'schema_version': 'ffps.phase18_pl_prospective_contract.v1',
        'league': 'PL', 'season': 2026, 'frozen_at': frozen_at,
        'authority_type': 'phase7p_frozen_window_approved_for_v2',
        'horizon_authority_phase': 'Phase7P', 'phase11_pl_protocol_exists': False,
        'authorization_phase': 'Phase18', 'end_at': END,
        'phase7p_specification': p.identity(prior.PHASE7_SPEC),
        'phase7p_frozen_reference': p.identity(PHASE7P_REFERENCE),
        'phase7p_offline_verification': p.identity(PHASE7P_OFFLINE),
        'phase17_offline_verification': p.identity(PHASE17_OFFLINE),
        'phase17_source_qualification': p.identity(prior.RUN / 'pl_source_qualification.json'),
        'source_adapter': qualification['adapter_implementation'],
        'revision_rule': p.identity(phase16_cohort.RULE_FILE),
        'revision_rule_sha256': rule['protocol_sha256'],
        'active_bundle_id': spec['active']['bundle_id'],
        'active_bundle_sha256': spec['active']['model_bundle_sha256'],
        'candidate_bundle_id': spec['candidate']['bundle_id'],
        'candidate_bundle_sha256': spec['candidate']['model_bundle_sha256'],
        'prospective_start_rule': 'cohort start fixed at initialization, strictly after contract freeze and initialization; never retrospective',
        'horizon_policy': 'exact Phase7P end, no extension, shortening, or replacement within this cohort',
        'initial_pair_prefix': [], 'initial_settlement_prefix': [],
        'historical_pl_pair_import_authorized': False,
        'result_aware_selection_authorized': False,
        'collection_authorized_now': False, 'scheduler_authorized': False,
        'activation_authorized': False, 'production_pointer_mutation_authorized': False},
        'contract_sha256')
    return put(PL_CONTRACT, value)


def verify_pl_contract():
    p.check(PL_CONTRACT.is_file(), 'PL_CONTRACT_NOT_FROZEN')
    value = p.read(PL_CONTRACT)
    p.frozen.sealed(value, 'contract_sha256')
    spec, qualification, rule = _frozen_pl_inputs()
    expected = {key: item for key, item in freeze_payload(value['frozen_at'], spec, qualification, rule).items()}
    p.check(value == expected, 'PL_CONTRACT_RECONSTRUCTION_MISMATCH')
    for field in ('phase7p_specification','phase7p_frozen_reference',
                  'phase7p_offline_verification','phase17_offline_verification',
                  'phase17_source_qualification','source_adapter','revision_rule'):
        p.frozen.reference(value[field])
    p.check(p.instant(value['frozen_at']) < p.instant(value['end_at']),
            'PL_CONTRACT_FROZEN_AFTER_END')
    return value


def freeze_payload(frozen_at, spec, qualification, rule):
    """Pure reconstruction of the approved contract, with no new clock reading."""
    return p.seal({'schema_version': 'ffps.phase18_pl_prospective_contract.v1',
        'league': 'PL', 'season': 2026, 'frozen_at': frozen_at,
        'authority_type': 'phase7p_frozen_window_approved_for_v2',
        'horizon_authority_phase': 'Phase7P', 'phase11_pl_protocol_exists': False,
        'authorization_phase': 'Phase18', 'end_at': END,
        'phase7p_specification': p.identity(prior.PHASE7_SPEC),
        'phase7p_frozen_reference': p.identity(PHASE7P_REFERENCE),
        'phase7p_offline_verification': p.identity(PHASE7P_OFFLINE),
        'phase17_offline_verification': p.identity(PHASE17_OFFLINE),
        'phase17_source_qualification': p.identity(prior.RUN / 'pl_source_qualification.json'),
        'source_adapter': qualification['adapter_implementation'],
        'revision_rule': p.identity(phase16_cohort.RULE_FILE),
        'revision_rule_sha256': rule['protocol_sha256'],
        'active_bundle_id': spec['active']['bundle_id'],
        'active_bundle_sha256': spec['active']['model_bundle_sha256'],
        'candidate_bundle_id': spec['candidate']['bundle_id'],
        'candidate_bundle_sha256': spec['candidate']['model_bundle_sha256'],
        'prospective_start_rule': 'cohort start fixed at initialization, strictly after contract freeze and initialization; never retrospective',
        'horizon_policy': 'exact Phase7P end, no extension, shortening, or replacement within this cohort',
        'initial_pair_prefix': [], 'initial_settlement_prefix': [],
        'historical_pl_pair_import_authorized': False,
        'result_aware_selection_authorized': False,
        'collection_authorized_now': False, 'scheduler_authorized': False,
        'activation_authorized': False, 'production_pointer_mutation_authorized': False},
        'contract_sha256')


def resolve(ref):
    """Return one common validated authority descriptor, never an arbitrary horizon."""
    path = p.frozen.reference(ref)
    if path == PL_CONTRACT:
        contract = verify_pl_contract()
        return {'authority_type': 'phase18_explicit_pl_contract',
            'authority': p.identity(PL_CONTRACT), 'league': contract['league'],
            'season': contract['season'], 'end_at': contract['end_at'],
            'frozen_at': contract['frozen_at'],
            'source_qualification': contract['phase17_source_qualification'],
            'source_adapter': contract['source_adapter'],
            'active_bundle_id': contract['active_bundle_id'],
            'candidate_bundle_id': contract['candidate_bundle_id'],
            'active_bundle_sha256': contract['active_bundle_sha256'],
            'candidate_bundle_sha256': contract['candidate_bundle_sha256']}
    for league in p.source.LEAGUES:
        expected = p.PHASE11 / league / 'prospective_protocol.json'
        if path == expected:
            old, _, controller, _, _ = p.frozen.load_protocol(p.PHASE11, league)
            p.check(old['league'] == league and old['end_at'] == p.read(path)['end_at'],
                    'PHASE11_AUTHORITY_DRIFT')
            return {'authority_type': 'phase11_frozen_protocol',
                'authority': p.identity(path), 'league': league, 'season': old['season'],
                'end_at': old['end_at'], 'frozen_at': old['frozen_at'],
                'source_qualification': old['qualification'], 'source_adapter': None,
                'active_bundle_id': controller.legacy_id,
                'candidate_bundle_id': controller.candidate_id}
    raise ValueError('UNSUPPORTED_PROSPECTIVE_HORIZON_AUTHORITY')

"""Authority-driven, empty v2 cohort initializer; no prediction capture API."""
import re
from datetime import timedelta

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort
from src.evaluation import phase18_authority as authority
from src.evaluation.phase12_operations import put
from src.ml.bundle_activation import state_lock


def _directory(cohort_id):
    p.check(bool(re.fullmatch(r'[A-Z0-9][A-Za-z0-9_-]{2,79}', str(cohort_id))),
            'INVALID_COHORT_ID')
    directory = (authority.RUN / 'cohorts' / cohort_id).resolve()
    p.check(directory.parent == authority.RUN / 'cohorts', 'COHORT_PATH_ESCAPE')
    return directory


def initialize(cohort_id, horizon_authority):
    """Use a verified authority; retain the frozen Phase16 publisher for Phase11."""
    descriptor = authority.resolve(horizon_authority)
    if descriptor['authority_type'] == 'phase11_frozen_protocol':
        return phase16_cohort.initialize(cohort_id, descriptor['league'])
    p.check(descriptor['authority_type'] == 'phase18_explicit_pl_contract',
            'UNSUPPORTED_PROSPECTIVE_HORIZON_AUTHORITY')
    p.check(cohort_id == 'PL-clean-001', 'PL_CONTRACT_BOUND_TO_SINGLE_COHORT')
    directory = _directory(cohort_id)
    with state_lock(directory / 'operation.lock'):
        if (directory / 'creation_intent.json').exists():
            intent = p.read(directory / 'creation_intent.json')
            p.frozen.sealed(intent, 'record_sha256')
            p.check(intent['cohort_id'] == cohort_id and
                    intent['horizon_authority'] == descriptor['authority'],
                    'COHORT_CREATION_INTENT_CONFLICT')
        else:
            created = p.now()
            start = (p.instant(created) + timedelta(seconds=2)).isoformat()
            p.check(p.instant(descriptor['frozen_at']) < p.instant(created) <
                    p.instant(start) < p.instant(descriptor['end_at']),
                    'COHORT_START_OR_HORIZON_INVALID')
            intent = p.seal({'schema_version': 'ffps.phase18_cohort_creation_intent.v1',
                'cohort_id': cohort_id, 'league': descriptor['league'],
                'created_at': created, 'start_at': start,
                'end_at': descriptor['end_at'],
                'horizon_authority': descriptor['authority'],
                'initial_pair_prefix': [], 'initial_settlement_prefix': []})
            put(directory / 'creation_intent.json', intent)
        protocol = protocol_payload(directory, descriptor, intent)
        put(directory / 'prospective_protocol.json', protocol)
        spec = specification_payload(directory, descriptor, intent, protocol)
        put(directory / 'cohort_specification.json', spec)
        from src.evaluation.phase18_verifier import verify_pl_cohort
        verify_pl_cohort(cohort_id)
        return spec


def protocol_payload(directory, descriptor, intent):
    rule = phase16_cohort.load_rule()
    return p.seal({'schema_version': 'ffps.league_prospective_protocol.v2',
        'phase': 18, 'league': descriptor['league'], 'season': descriptor['season'],
        'frozen_at': intent['created_at'], 'start_at': intent['start_at'],
        'end_at': descriptor['end_at'],
        'cohort_path': directory.relative_to(p.ROOT).as_posix(),
        'horizon_authority': descriptor['authority'],
        'horizon_authority_type': descriptor['authority_type'],
        'source_qualification': descriptor['source_qualification'],
        'source_adapter': descriptor['source_adapter'],
        'revision_rule': p.identity(phase16_cohort.RULE_FILE),
        'revision_rule_sha256': rule['protocol_sha256'],
        'active_bundle_id': descriptor['active_bundle_id'],
        'active_bundle_sha256': descriptor['active_bundle_sha256'],
        'candidate_bundle_id': descriptor['candidate_bundle_id'],
        'candidate_bundle_sha256': descriptor['candidate_bundle_sha256'],
        'first_collection_policy': rule['selection_policy'],
        'eligible_fixture_policy': 'all future PL fixtures with qualified causal features within frozen Phase7P boundary',
        'initial_pair_prefix': [], 'initial_settlement_prefix': [],
        'promotion_authorized': False, 'collection_scheduled': False},
        'protocol_sha256')


def specification_payload(directory, descriptor, intent, protocol):
    rule = phase16_cohort.load_rule()
    return p.seal({'schema_version': 'ffps.clean_cohort.v2',
        'cohort_id': intent['cohort_id'], 'league': descriptor['league'],
        'season': descriptor['season'], 'created_at': intent['created_at'],
        'start_at': intent['start_at'], 'end_at': descriptor['end_at'],
        'creation_intent': p.identity(directory / 'creation_intent.json'),
        'protocol': p.identity(directory / 'prospective_protocol.json'),
        'protocol_sha256': protocol['protocol_sha256'],
        'horizon_authority': descriptor['authority'],
        'source_qualification': descriptor['source_qualification'],
        'source_adapter': descriptor['source_adapter'],
        'revision_rule': p.identity(phase16_cohort.RULE_FILE),
        'revision_rule_sha256': rule['protocol_sha256'],
        'active_bundle_id': descriptor['active_bundle_id'],
        'active_bundle_sha256': descriptor['active_bundle_sha256'],
        'candidate_bundle_id': descriptor['candidate_bundle_id'],
        'candidate_bundle_sha256': descriptor['candidate_bundle_sha256'],
        'initial_pair_prefix': [], 'initial_settlement_prefix': [],
        'initial_journal_head': None,
        'admission_policy': 'only predictions created under this v2 protocol after start_at',
        'horizon_extension_authorized': False, 'historical_pair_import_authorized': False,
        'activation_authorized': False, 'production_pointer_mutation_authorized': False},
        'record_sha256')


def validate_pair_origin(spec, pair):
    """Read-only preflight; older Phase7P predictions cannot enter this cohort."""
    p.check(pair.get('league') == spec['league'] and
            pair.get('protocol_sha256') == spec['protocol_sha256'] and
            pair.get('cohort_id') == spec['cohort_id'] and
            pair.get('created_at') is not None and pair.get('kickoff_at') is not None,
            'CROSS_PROTOCOL_OR_HISTORICAL_PAIR')
    p.check(p.instant(spec['start_at']) <= p.instant(pair['created_at']) <
            p.instant(pair['kickoff_at']) < p.instant(spec['end_at']),
            'RETROSPECTIVE_OR_OUT_OF_HORIZON_PAIR')
    return True

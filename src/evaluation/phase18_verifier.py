"""Independent offline reconstruction of the PL contract and five empty cohorts."""
from datetime import timedelta

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort, phase16_verifier
from src.evaluation import phase17_readiness as prior
from src.evaluation import phase18_authority as authority


def verify_pl_cohort(cohort_id='PL-clean-001'):
    p.check(cohort_id == 'PL-clean-001', 'UNKNOWN_PL_COHORT_ID')
    directory = (authority.RUN / 'cohorts' / cohort_id).resolve()
    p.check(directory.parent == authority.RUN / 'cohorts', 'COHORT_PATH_ESCAPE')
    contract = authority.verify_pl_contract()
    descriptor = authority.resolve(p.identity(authority.PL_CONTRACT))
    rule = phase16_cohort.load_rule()
    intent = p.read(directory / 'creation_intent.json')
    p.frozen.sealed(intent, 'record_sha256')
    p.check(intent == p.seal({'schema_version': 'ffps.phase18_cohort_creation_intent.v1',
        'cohort_id': cohort_id, 'league': 'PL', 'created_at': intent['created_at'],
        'start_at': intent['start_at'], 'end_at': authority.END,
        'horizon_authority': p.identity(authority.PL_CONTRACT),
        'initial_pair_prefix': [], 'initial_settlement_prefix': []}) and
        p.instant(contract['frozen_at']) < p.instant(intent['created_at']) and
        intent['start_at'] == (p.instant(intent['created_at']) + timedelta(seconds=2)).isoformat() and
        p.instant(intent['created_at']) < p.instant(intent['start_at']) < p.instant(authority.END),
        'PL_CREATION_INTENT_OR_START_INVALID')
    protocol = p.read(directory / 'prospective_protocol.json')
    p.frozen.sealed(protocol, 'protocol_sha256')
    expected_protocol = p.seal({'schema_version': 'ffps.league_prospective_protocol.v2',
        'phase': 18, 'league': 'PL', 'season': 2026,
        'frozen_at': intent['created_at'], 'start_at': intent['start_at'],
        'end_at': authority.END, 'cohort_path': directory.relative_to(p.ROOT).as_posix(),
        'horizon_authority': p.identity(authority.PL_CONTRACT),
        'horizon_authority_type': 'phase18_explicit_pl_contract',
        'source_qualification': contract['phase17_source_qualification'],
        'source_adapter': contract['source_adapter'],
        'revision_rule': p.identity(phase16_cohort.RULE_FILE),
        'revision_rule_sha256': rule['protocol_sha256'],
        'active_bundle_id': contract['active_bundle_id'],
        'active_bundle_sha256': contract['active_bundle_sha256'],
        'candidate_bundle_id': contract['candidate_bundle_id'],
        'candidate_bundle_sha256': contract['candidate_bundle_sha256'],
        'first_collection_policy': rule['selection_policy'],
        'eligible_fixture_policy': 'all future PL fixtures with qualified causal features within frozen Phase7P boundary',
        'initial_pair_prefix': [], 'initial_settlement_prefix': [],
        'promotion_authorized': False, 'collection_scheduled': False}, 'protocol_sha256')
    p.check(protocol == expected_protocol, 'PL_PROTOCOL_RECONSTRUCTION_MISMATCH')
    spec = p.read(directory / 'cohort_specification.json')
    p.frozen.sealed(spec, 'record_sha256')
    expected_spec = p.seal({'schema_version': 'ffps.clean_cohort.v2',
        'cohort_id': cohort_id, 'league': 'PL', 'season': 2026,
        'created_at': intent['created_at'], 'start_at': intent['start_at'],
        'end_at': authority.END, 'creation_intent': p.identity(directory / 'creation_intent.json'),
        'protocol': p.identity(directory / 'prospective_protocol.json'),
        'protocol_sha256': protocol['protocol_sha256'],
        'horizon_authority': p.identity(authority.PL_CONTRACT),
        'source_qualification': contract['phase17_source_qualification'],
        'source_adapter': contract['source_adapter'],
        'revision_rule': p.identity(phase16_cohort.RULE_FILE),
        'revision_rule_sha256': rule['protocol_sha256'],
        'active_bundle_id': contract['active_bundle_id'],
        'active_bundle_sha256': contract['active_bundle_sha256'],
        'candidate_bundle_id': contract['candidate_bundle_id'],
        'candidate_bundle_sha256': contract['candidate_bundle_sha256'],
        'initial_pair_prefix': [], 'initial_settlement_prefix': [],
        'initial_journal_head': None,
        'admission_policy': 'only predictions created under this v2 protocol after start_at',
        'horizon_extension_authorized': False, 'historical_pair_import_authorized': False,
        'activation_authorized': False, 'production_pointer_mutation_authorized': False},
        'record_sha256')
    p.check(spec == expected_spec, 'PL_SPECIFICATION_RECONSTRUCTION_MISMATCH')
    allowed = {'creation_intent.json', 'prospective_protocol.json',
               'cohort_specification.json', 'operation.lock'}
    unexpected = [path.relative_to(directory).as_posix() for path in directory.rglob('*')
                  if path.is_file() and path.name not in allowed]
    p.check(not unexpected, 'PL_COHORT_NOT_EMPTY:' + ','.join(unexpected))
    return {'verified': True, 'league': 'PL', 'cohort_id': cohort_id,
        'cohort': p.identity(directory / 'cohort_specification.json'),
        'protocol': p.identity(directory / 'prospective_protocol.json'),
        'horizon_authority': p.identity(authority.PL_CONTRACT),
        'source_qualification': contract['phase17_source_qualification'],
        'revision_rule': p.identity(phase16_cohort.RULE_FILE),
        'rule_sha256': rule['protocol_sha256'],
        'season': 2026, 'start_at': intent['start_at'], 'end_at': authority.END,
        'active_bundle_id': descriptor['active_bundle_id'],
        'candidate_bundle_id': descriptor['candidate_bundle_id'],
        'pairs': 0, 'settlements': 0, 'journal_head': None,
        'state': 'V2_CLEAN_COHORT_READY', 'activation_authorized': False}


def all_leagues():
    rule = phase16_cohort.load_rule()
    rows = []
    for league, cohort_id in prior.COHORT_IDS.items():
        result = phase16_verifier.verify_cohort(cohort_id)
        spec, _ = phase16_cohort.binding(cohort_id)
        authority.resolve(spec['prior_protocol'])
        p.check(result['verified'] and result['league'] == league and
                result['pairs'] == result['settlements'] == 0 and
                result['journal_head'] is None and
                spec['revision_rule_sha256'] == rule['protocol_sha256'],
                'EXISTING_CLEAN_COHORT_CHANGED')
        rows.append({'league': league, 'cohort_id': cohort_id, 'cohort': result['cohort'],
                     'protocol': result['protocol'], 'horizon_authority': spec['prior_protocol'],
                     'source_qualification': spec['source_qualification'],
                     'rule_sha256': spec['revision_rule_sha256'], 'season': spec['season'],
                     'start_at': spec['start_at'], 'end_at': spec['end_at'],
                     'active_bundle_id': spec['active_bundle_id'],
                     'candidate_bundle_id': spec['candidate_bundle_id'],
                     'pairs': 0, 'settlements': 0, 'journal_head': None,
                     'state': 'V2_CLEAN_COHORT_READY', 'activation_authorized': False})
    rows.append(verify_pl_cohort())
    p.check(len(rows) == 5 and {row['league'] for row in rows} == {'BL1','SA','FL1','PD','PL'} and
            len({row['cohort']['path'] for row in rows}) == 5,
            'ALL_LEAGUE_COHORT_ISOLATION_FAILED')
    return p.seal({'schema_version': 'ffps.phase18_all_league_readiness.v1',
        'scientific_state': 'PHASE18_ALL_LEAGUES_V2_READY_FOR_PROSPECTIVE_COLLECTION',
        'historical_bl1_state': 'BLOCKED_PROTOCOL_VIOLATION',
        'revision_rule': p.identity(phase16_cohort.RULE_FILE),
        'revision_rule_sha256': rule['protocol_sha256'],
        'leagues': rows, 'pairs_admitted': 0, 'settlements': 0,
        'collection_scheduled': False, 'activation_authorized': False}, 'record_sha256')

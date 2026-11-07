"""One-shot collection authorization bound to five sealed v2 cohorts."""
from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort, phase17_readiness, phase18_authority
from src.evaluation.phase12_operations import put

RUN = p.ROOT / 'baselines/phase19/phase19-controlled-collection-20260923-v1'
AUTH = RUN / 'collection_authorization.json'
PHASE18_READY = phase18_authority.RUN / 'readiness.json'
PHASE17_MODELS = phase17_readiness.RUN / 'artifact_identities.json'


def _facts(frozen_at):
    ready = p.read(PHASE18_READY)
    p.frozen.sealed(ready, 'record_sha256')
    p.check(ready['scientific_state'] == 'PHASE18_ALL_LEAGUES_V2_READY_FOR_PROSPECTIVE_COLLECTION' and
            len(ready['leagues']) == 5 and ready['pairs_admitted'] == ready['settlements'] == 0,
            'PHASE18_PREREQUISITE_NOT_READY')
    models = p.read(PHASE17_MODELS)
    p.frozen.sealed(models, 'record_sha256')
    rule = phase16_cohort.load_rule()
    cohorts = []
    for row in ready['leagues']:
        p.check(row['rule_sha256'] == rule['protocol_sha256'] and
                row['pairs'] == row['settlements'] == 0,
                'PHASE19_COHORT_OR_RULE_DRIFT')
        for field in ('cohort','protocol','source_qualification','horizon_authority'):
            p.frozen.reference(row[field])
        league = row['league']
        adapter = (p.identity(p.ROOT / 'src/evaluation/phase17_pl_adapter.py')
                   if league == 'PL' else
                   p.identity(p.ROOT / 'src/evaluation/phase11_current_source.py'))
        cohorts.append({'league': league, 'cohort_id': row['cohort_id'],
            'cohort': row['cohort'], 'protocol': row['protocol'],
            'horizon_authority': row['horizon_authority'],
            'source_qualification': row['source_qualification'],
            'source_adapter': adapter, 'active_bundle_id': row['active_bundle_id'],
            'candidate_bundle_id': row['candidate_bundle_id'],
            'model_identity': models['leagues'][league],
            'end_at': row['end_at']})
    return p.seal({'schema_version': 'ffps.phase19_collection_authorization.v1',
        'frozen_at': frozen_at, 'collection_mode': 'CONTROLLED_ONE_SHOT',
        'phase18_readiness': p.identity(PHASE18_READY),
        'phase17_model_identities': p.identity(PHASE17_MODELS),
        'revision_rule': p.identity(phase16_cohort.RULE_FILE),
        'revision_rule_sha256': rule['protocol_sha256'],
        'cohorts': cohorts,
        'fresh_source_acquisition_authorized': True,
        'normalization_authorized': True,
        'prospective_pair_capture_authorized': True,
        'append_only_observation_authorized': True,
        'offline_verification_authorized': True,
        'settlement_authorized': False,
        'scheduler_authorized': False, 'activation_authorized': False,
        'training_authorized': False, 'calibration_authorized': False,
        'candidate_replacement_authorized': False,
        'model_mutation_authorized': False,
        'production_pointer_mutation_authorized': False,
        'retrospective_acquisition_authorized': False,
        'historical_pair_import_authorized': False,
        'manual_fixture_exclusion_authorized': False,
        'manual_pair_deletion_authorized': False,
        'horizon_mutation_authorized': False}, 'record_sha256')


def freeze():
    if AUTH.exists():
        return verify()
    return put(AUTH, _facts(p.now()))


def verify():
    row = p.read(AUTH)
    p.frozen.sealed(row, 'record_sha256')
    p.check(row == _facts(row['frozen_at']), 'PHASE19_AUTHORIZATION_RECONSTRUCTION_MISMATCH')
    for item in row['cohorts']:
        p.frozen.reference(item['source_adapter'])
    p.check(p.instant(row['frozen_at']) < p.instant(p.now()), 'PHASE19_AUTHORIZATION_TIME_INVALID')
    return row

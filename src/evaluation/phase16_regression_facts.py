"""Read-only historical regression reconstruction; not a new-cohort admission."""
from src.evaluation import phase12_prospective as p
from src.evaluation import phase15_review_contract as old
from src.evaluation import phase15_review_verifier as old_verifier
from src.evaluation import phase16_revision as revision
from src.evaluation import phase16_verifier as new_verifier

RUN = p.ROOT / 'baselines/phase16/phase16-fixture-revision-20260923-v1'
REVIEW = old.RUN / 'reviews/BL1/565839/review-000001.json'


def reconstruct():
    """Independently replay the sealed old review and apply v2 only hypothetically."""
    p.check(old_verifier.verify(REVIEW)['verified'], 'HISTORICAL_REVIEW_NOT_VERIFIED')
    historical = p.read(REVIEW)
    p.frozen.sealed(historical, 'review_sha256')
    p.check(historical['outcome'] == 'UNRESOLVED_PROTOCOL_CONFLICT' and
            historical['clears_protocol_blocker'] is False, 'HISTORICAL_REVIEW_REINTERPRETED')
    basis = historical['basis']
    protocol, _, qualification, pairs, _, targets, events = p.load_cohort(old.a.COHORT, 'BL1')
    count = len(basis['lineage']['ordered_events'])
    p.check([p.identity(old.a.COHORT / 'BL1' / 'journal' / f"{event['sequence']:08d}.json")
             for event in events[:count]] == basis['lineage']['ordered_events'], 'HISTORICAL_LINEAGE_PREFIX_CHANGED')
    preserved_pairs = [pair for pair in pairs if pair['pair_sha256'] in basis['lineage']['pair_hashes']]
    p.check([pair['pair_sha256'] for pair in preserved_pairs] == basis['lineage']['pair_hashes'],
            'HISTORICAL_PAIR_COHORT_CHANGED')
    old_pair = next(pair for pair in preserved_pairs if pair['fixture_id'] == basis['fixture_id'])
    p.check(basis['admission']['payload'] == old_pair, 'HISTORICAL_ADMISSION_CHANGED')
    versions = []
    for row in basis['observed_versions']:
        source = row['metadata']
        normalized = [event for event in p.current_events(source, protocol, qualification)
                      if event['fixture_id'] == old_pair['fixture_id']]
        p.check(len(normalized) == 1 and normalized[0] == row['fixture'],
                'HISTORICAL_SOURCE_RECONSTRUCTION_MISMATCH')
        versions.append({'source': source, 'fixture': normalized[0]})
    rule = revision.rule()
    p.check(p.read(RUN / 'fixture_revision_rule_v2.json') == rule, 'NEW_RULE_NOT_FROZEN')
    hypothetical = {'cohort_id': 'HISTORICAL_REGRESSION_ONLY_BL1_565839',
        'league': 'BL1', 'season': protocol['season'],
        'protocol_sha256': protocol['protocol_sha256'],
        'revision_rule_sha256': rule['protocol_sha256'],
        'start_at': protocol['start_at'], 'end_at': protocol['end_at']}
    classifier = revision.classify(rule, hypothetical,
        new_verifier._facts(old_pair, targets[old_pair['fixture_id']]), versions,
        [new_verifier._facts(pair, targets[pair['fixture_id']]) for pair in preserved_pairs])
    p.check(classifier['classification'] == 'PERMITTED_KICKOFF_REVISION',
            'V2_FACTUAL_REGRESSION_DID_NOT_CLASSIFY')
    old_consumption_path = old.RUN / 'phase14_consumption.json'
    consumption = p.read(old_consumption_path)
    p.frozen.sealed(consumption, 'record_sha256')
    p.check(consumption['state'] == 'BLOCKED_PROTOCOL_VIOLATION', 'OLD_PHASE14_STATE_CHANGED')
    return p.seal({'schema_version': 'ffps.phase16_historical_regression.v1',
        'historical_review': p.identity(REVIEW),
        'historical_cohort': p.identity(old.a.COHORT / 'run_specification.json'),
        'historical_phase14_consumption': p.identity(old_consumption_path),
        'historical_protocol_sha256': protocol['protocol_sha256'],
        'historical_pair_sha256': old_pair['pair_sha256'],
        'v2_revision_rule': p.identity(RUN / 'fixture_revision_rule_v2.json'),
        'source_versions': [p.identity(p.frozen.reference(row['source']['raw'])) for row in versions],
        'classification': classifier,
        'historical_scientific_state': 'BLOCKED_PROTOCOL_VIOLATION',
        'historical_review_outcome': 'UNRESOLVED_PROTOCOL_CONFLICT',
        'counterfactual_only': True, 'admitted_to_new_cohort': False,
        'activation_authorized': False}, 'record_sha256')

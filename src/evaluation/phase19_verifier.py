"""Offline reconstruction of Phase 19's archived sources and complete v2 journals."""
from __future__ import annotations

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort as cohort
from src.evaluation import phase16_revision as revision
from src.evaluation import phase16_verifier as v16
from src.evaluation import phase19_authorization as auth
from src.evaluation import phase19_batch as batch
from src.evaluation import phase19_collection as collection
from src.evaluation import phase19_pl_cycle as pl
from src.evaluation.phase12_operations import put

VERIFIED_DIR = auth.RUN / 'league_verification'


def _acquisition(league):
    initial = collection.acquire(league)
    p.check(initial['status'] == 'SOURCE_UNAVAILABLE' and
            initial['error_type'] == 'ConnectionError', 'PHASE19_INITIAL_ATTEMPT_DRIFT')
    recovered = collection.recover_network_attempt(league)
    p.check(recovered['status'] == 'RETRIEVED' and recovered['http_status'] == 200 and
            p.instant(initial['retrieved_at']) < p.instant(recovered['started_at']) <
            p.instant(recovered['retrieved_at']) <= p.instant(p.now()),
            'PHASE19_RECOVERY_ORDER_INVALID')
    p.frozen.reference(recovered['raw'])
    return initial, recovered


def nonpl(league, progress=None, *, progress_event=None):
    if progress_event:
        progress_event('league_preparing', league, 0, None, None)
    assignment = next(row for row in auth.verify()['cohorts'] if row['league'] == league)
    _, discovery = _acquisition(league)
    path, q, index = batch.qualification(league)
    p.evidence11.replay_source_evidence(q)
    source = index['sources'][3]
    p.check(p.instant(source['started_at']) > p.instant(auth.verify()['frozen_at']),
            'PHASE19_QUALIFICATION_SOURCE_STALE')
    spec, protocol = cohort.binding(assignment['cohort_id'])
    _, _, controller, original, original_index = p.frozen.load_protocol(p.PHASE11, league)
    p.pair_qualification({'qualification': p.identity(path)},
        {'qualification_ref': spec['source_qualification'], 'value': original}, original_index)
    events = p.current_events(source, protocol, q)
    by_id = {row['fixture_id']: row for row in events}
    p.check(len(by_id) == len(events), 'PHASE19_DUPLICATE_NORMALIZED_FIXTURE')
    discovered = p.current_events(discovery, protocol, q)
    first_by_id = {row['fixture_id']: row for row in discovered}
    stable = ('league', 'season', 'fixture_id', 'provider_fixture_id',
              'home_id', 'away_id', 'home_team', 'away_team',
              'kickoff_at', 'status', 'home_goals', 'away_goals')
    p.check(len(first_by_id) == len(discovered) and set(first_by_id) == set(by_id) and
            all(all(first_by_id[key][field] == by_id[key][field] for field in stable)
                for key in by_id), 'PHASE19_SOURCE_VERSION_MISMATCH')
    journal = cohort.journal(assignment['cohort_id'])
    pairs = [row['payload']['pair'] for row in journal if row['kind'] == 'pair']
    p.check(all(row['kind'] == 'pair' for row in journal) and
            len({pair['fixture_id'] for pair in pairs}) == len(pairs),
            'PHASE19_UNAUTHORIZED_JOURNAL_EVENT')
    facts = []
    if progress_event:
        progress_event('league_started', league, 0, len(journal), None)
    for index_number, row in enumerate(journal, 1):
        pair = row['payload']['pair']
        if progress_event:
            progress_event('pair_started', league, index_number - 1, len(journal),
                           pair.get('fixture_id') if isinstance(pair, dict) else None)
        p.check(pair['league'] == league and pair['protocol_sha256'] == protocol['protocol_sha256'] and
                pair['qualification'] == p.identity(path), 'PHASE19_PAIR_BINDING_DRIFT')
        target = p.verify_pair_features(pair, q, index, protocol, controller)
        observation = row['payload']['initial_observation']
        p.check(observation['source'] == source and observation['fixture'] == target and
                by_id.get(pair['fixture_id']) == target and
                p.instant(source['retrieved_at']) <= p.instant(pair['created_at']) <=
                p.instant(row['created_at']) < p.instant(target['kickoff_at']),
                'PHASE19_PAIR_SOURCE_REPLAY_DRIFT')
        fact = v16._facts(pair, target)
        facts.append(fact)
        classified = revision.classify(cohort.load_rule(), spec, fact,
            [observation], facts)
        p.check(classified == row['payload']['classification'] and
                classified['classification'] == 'UNCHANGED',
                'PHASE19_CLASSIFIER_REPLAY_DRIFT')
        if progress_event:
            progress_event('pair_verified', league, index_number, len(journal), pair['fixture_id'])
        if progress and index_number % 25 == 0:
            progress(league, index_number, len(journal))
    paired = {pair['fixture_id'] for pair in pairs}
    expected = {row['fixture_id'] for row in q['assessment']['upcoming']}
    rejected = {}
    for fixture_id in expected - paired:
        path_rejection = batch.ERRORS / league / (fixture_id + '.json')
        p.check(path_rejection.exists(), 'PHASE19_UNACCOUNTED_FIXTURE:' + fixture_id)
        value = p.read(path_rejection)
        p.frozen.sealed(value, 'record_sha256')
        p.check(value['league'] == league and value['fixture_id'] == fixture_id and
                value['source'] == source['raw'] and value['qualification'] == p.identity(path),
                'PHASE19_REJECTION_SCOPE_DRIFT')
        rejected[fixture_id] = value['reason']
    p.check(paired | set(rejected) == expected and not (paired & set(rejected)),
            'PHASE19_INCOMPLETE_FIXTURE_ACCOUNTING')
    if progress_event:
        progress_event('league_verified', league, len(journal), len(journal), None)
    return {'league': league, 'verified': True, 'cohort_id': assignment['cohort_id'],
        'source': source['raw'], 'source_retrieved_at': source['retrieved_at'],
        'qualification': p.identity(path), 'fixtures_observed': len(events),
        'fixtures_considered': len(expected), 'pairs': len(pairs),
        'settlements': 0, 'observations': 0, 'rejected': len(rejected),
        'source_mismatch_count': 0,
        'rejection_reasons': {reason: sum(x == reason for x in rejected.values())
                              for reason in sorted(set(rejected.values()))},
        'journal_head': journal[-1]['record_sha256'] if journal else None,
        'horizon_end': assignment['end_at']}


def store_nonpl(league, progress=None):
    result = p.seal(nonpl(league, progress=progress), 'record_sha256')
    return put(VERIFIED_DIR / (league + '.json'), result)


def _stored_nonpl(league):
    path = VERIFIED_DIR / (league + '.json')
    row = p.read(path)
    p.frozen.sealed(row, 'record_sha256')
    current = cohort.journal(row['cohort_id'])
    p.check(row['league'] == league and row['verified'] is True and
            row['journal_head'] == (current[-1]['record_sha256'] if current else None) and
            row['pairs'] == sum(event['kind'] == 'pair' for event in current),
            'PHASE19_VERIFIED_PREFIX_CHANGED')
    p.frozen.reference(row['source'])
    p.frozen.reference(row['qualification'])
    _, q, index = batch.qualification(league)
    p.frozen.reference(q['source_index'])
    for source in index['sources']:
        p.frozen.reference(source['raw'])
    return {key: value for key, value in row.items() if key != 'record_sha256'}


def all_leagues(*, stored=False, progress=None, progress_event=None):
    authorization = auth.verify()
    p.check(authorization['scheduler_authorized'] is False and
            authorization['settlement_authorized'] is False and
            authorization['activation_authorized'] is False,
            'PHASE19_AUTHORIZATION_BOUNDARY_DRIFT')
    rows = [(_stored_nonpl(league) if stored else nonpl(
                league, progress=progress, progress_event=progress_event))
            for league in ('BL1', 'SA', 'FL1', 'PD')]
    if progress_event:
        progress_event('league_started', 'PL', 0, 0, None)
    _acquisition('PL')
    pl_cycle = pl.verify()
    rows.append({'league': 'PL', 'verified': True, 'cohort_id': 'PL-clean-001',
        'source': pl_cycle['raw'],
        'source_retrieved_at': p.read(collection.ACQUISITION / 'PL/recovery_matches.json')['retrieved_at'],
        'qualification': assignment['source_qualification'] if
            (assignment := next(row for row in authorization['cohorts'] if row['league'] == 'PL')) else None,
        'fixtures_observed': pl_cycle['fixtures_observed'],
        'fixtures_considered': pl_cycle['future_within_horizon'],
        'pairs': 0, 'settlements': 0, 'observations': 0,
        'source_mismatch_count': 0,
        'rejected': len(pl_cycle['rejections']),
        'rejection_reasons': {'NO_FROZEN_CAUSAL_FEATURE_ROW': len(pl_cycle['rejections'])},
        'journal_head': None, 'horizon_end': pl_cycle['horizon_end']})
    p.check(len(rows) == 5 and len({row['cohort_id'] for row in rows}) == 5 and
            all(row['verified'] for row in rows), 'PHASE19_LEAGUE_ISOLATION_FAILED')
    if progress_event:
        progress_event('league_verified', 'PL', 0, 0, None)
    return p.seal({'schema_version': 'ffps.phase19_offline_verification.v1',
        'authorization': p.identity(auth.AUTH), 'leagues': rows,
        'scheduler_authorized': False, 'activation_authorized': False}, 'record_sha256')

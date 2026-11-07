"""Single-source, append-only Phase 19 batch admission for qualified non-PL leagues.

Expensive source qualification and cohort replay run once at the batch boundary.
Each individual pair still receives the frozen feature, prediction, source,
revision and common-RNG checks before its atomic journal publication.
"""
from __future__ import annotations

from src.evaluation import phase10_current_data as current
from src.evaluation import phase10_shadow as capture
from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort as cohort
from src.evaluation import phase16_revision as revision
from src.evaluation import phase16_verifier as verifier
from src.evaluation import phase19_authorization as authorization
from src.evaluation.phase12_operations import put
from src.ml.bundle_activation import state_lock

REFRESH = p.ROOT / 'baselines/phase11/phase19-qualified-refresh-20260923-v1'
ERRORS = authorization.RUN / 'rejections'


def check_target(target, spec, source, at):
    p.check(target['status'] in ('TIMED', 'SCHEDULED') and
            p.instant(spec['start_at']) <= p.instant(at) <
            p.instant(target['kickoff_at']) < p.instant(spec['end_at']) and
            p.instant(source['retrieved_at']) < p.instant(at),
            'PHASE19_FIXTURE_NOT_CAUSALLY_ELIGIBLE')


def qualification(league):
    path = REFRESH / league / 'assessments/assessment-001/qualification.json'
    q, index, _ = p.frozen.load_qualification(path)
    p.check(q['league'] == league and q['feature_readiness'] and
            q['assessment']['qualified'], 'PHASE19_QUALIFICATION_NOT_READY')
    return path, q, index


def collect_nonpl(league, progress=None):
    auth = authorization.verify()
    p.check(league in ('BL1', 'SA', 'FL1', 'PD'), 'PHASE19_NONPL_LEAGUE_INVALID')
    assigned = next(row for row in auth['cohorts'] if row['league'] == league)
    cohort_id = assigned['cohort_id']
    path, q, index = qualification(league)
    source = index['sources'][3]
    p.check(source['status'] == 'RETRIEVED' and
            p.instant(source['started_at']) > p.instant(auth['frozen_at']),
            'PHASE19_SOURCE_NOT_FRESH')
    p.frozen.reference(source['raw'])
    spec, protocol = cohort.binding(cohort_id)
    p.check(spec['league'] == league and spec['source_qualification'] == assigned['source_qualification'],
            'PHASE19_COHORT_BINDING_DRIFT')
    _, _, controller, original, original_index = p.frozen.load_protocol(p.PHASE11, league)
    p.pair_qualification({'qualification': p.identity(path)},
        {'qualification_ref': spec['source_qualification'], 'value': original}, original_index)
    events = p.current_events(source, protocol, q)
    by_id = {event['fixture_id']: event for event in events}
    p.check(len(by_id) == len(events), 'PHASE19_DUPLICATE_SOURCE_FIXTURE')
    active, candidate = [controller.owner.load(controller.bundle(bundle))
                         for bundle in (controller.legacy_id, controller.candidate_id)]
    rule = cohort.load_rule()
    with state_lock(cohort._directory(cohort_id) / 'operation.lock'):
        before = verifier.verify_cohort(cohort_id)
        rows = cohort.journal(cohort_id)
        existing = {row['payload']['pair']['fixture_id'] for row in rows if row['kind'] == 'pair'}
        facts = [verifier._facts(row['payload']['pair'],
            by_id[row['payload']['pair']['fixture_id']]) for row in rows if row['kind'] == 'pair']
        previous = rows[-1]['record_sha256'] if rows else None
        sequence = len(rows)
        results = {'league': league, 'source': source['raw'],
                   'observed': len(events), 'considered': 0,
                   'already_admitted': len(existing), 'admitted': 0,
                   'rejected': 0, 'journal_head_before': before['journal_head']}
        for target in q['assessment']['upcoming']:
            fixture_id = target['fixture_id']
            results['considered'] += 1
            if fixture_id in existing:
                continue
            reason = None
            try:
                p.check(by_id.get(fixture_id) == target, 'PHASE19_SOURCE_TARGET_MISMATCH')
                at = p.now()
                check_target(target, spec, source, at)
                matrix, fixture = p.source.target_matrix(q['assessment'], target,
                    as_of=at, source_index=index)
                current.validate_features(matrix, fixture, q['ordered_features'], at)
                fixture = fixture.copy()
                fixture['kickoff_utc'] = target['kickoff_at']
                predicted_at = p.now()
                records = [capture.capture(matrix, fixture, suite, league, predicted_at)
                           for suite in (active, candidate)]
                pair = p.seal({'league': league, 'protocol_sha256': protocol['protocol_sha256'],
                    'fixture_id': fixture_id, 'created_at': p.now(),
                    'kickoff_at': target['kickoff_at'], 'qualification': p.identity(path),
                    'active': records[0], 'candidate': records[1]}, 'pair_sha256')
                p.verify_pair_features(pair, q, index, protocol, controller)
                p.check(p.instant(source['retrieved_at']) <= p.instant(pair['created_at']) <
                        p.instant(target['kickoff_at']) and p.instant(p.now()) <
                        p.instant(target['kickoff_at']), 'PHASE19_ADMISSION_TIME_INVALID')
                one = verifier._facts(pair, target)
                classification = revision.classify(rule, spec, one,
                    [{'source': source, 'fixture': target}], facts + [one])
                p.check(classification['classification'] == 'UNCHANGED',
                        'PHASE19_ADMISSION_CLASSIFICATION_BLOCKED')
                sequence += 1
                event = p.seal({'sequence': sequence, 'previous': previous,
                    'league': league, 'kind': 'pair', 'created_at': p.now(),
                    'payload': {'pair': pair,
                        'initial_observation': {'source': source, 'fixture': target},
                        'classification': classification}})
                p.persist_new(cohort._directory(cohort_id) / 'journal' /
                              f'{sequence:08d}.json', event)
                previous = event['record_sha256']
                facts.append(one)
                existing.add(fixture_id)
                results['admitted'] += 1
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                reason = f'{type(exc).__name__}:{exc}'
            if reason:
                put(ERRORS / league / (fixture_id + '.json'), p.seal({
                    'schema_version': 'ffps.phase19_fixture_rejection.v1',
                    'league': league, 'fixture_id': fixture_id,
                    'source': source['raw'], 'qualification': p.identity(path),
                    'reason': reason}, 'record_sha256'))
                results['rejected'] += 1
            if progress and results['considered'] % 10 == 0:
                progress(dict(results))
        results['journal_head_after'] = previous
    return results

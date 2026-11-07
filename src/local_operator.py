"""Read-oriented personal-PC workflow over the frozen Phase 19/20 contracts.

This module owns no feature, prediction, settlement, metric, or integrity rule.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort as cohort
from src.evaluation import phase18_cohort
from src.evaluation import phase19_authorization as phase19
from src.evaluation import phase19_integrity
from src.evaluation import phase19_pl_cycle
from src.evaluation import phase19_verifier
from src.evaluation import phase20_settlement as phase20
from src.evaluation import prospective_collection_next as next_collection

LEAGUES = ('BL1', 'SA', 'FL1', 'PD', 'PL')
HISTORICAL = p.ROOT / 'baselines/phase12/phase12-prospective-20260908-v1'


def _attempt(action):
    try:
        return {'verified': True, 'value': action()}
    except Exception as exc:
        return {'verified': False, 'error': f'{type(exc).__name__}: {exc}'}


def _authority():
    auth = phase19.verify()
    report = p.read(phase19.RUN / 'run_report.json')
    p.frozen.sealed(report, 'record_sha256')
    p.check(report['authorization'] == p.identity(phase19.AUTH),
            'OPERATOR_PHASE19_REPORT_AUTHORIZATION_DRIFT')
    p.frozen.reference(report['offline_verification'])
    return auth, report


def _current(league, auth, report, *, audit=False, full_replay=False):
    assigned = next(row for row in auth['cohorts'] if row['league'] == league)
    sealed_row = next(row for row in report['leagues'] if row['league'] == league)
    if league == 'PL':
        directory = phase18_cohort._directory(assigned['cohort_id'])
        spec = p.read(directory / 'cohort_specification.json')
        p.frozen.sealed(spec, 'record_sha256')
        p.check(spec['initial_pair_prefix'] == [] and
                spec['initial_journal_head'] is None, 'OPERATOR_PL_COHORT_ORIGIN_DRIFT')
    else:
        directory = cohort._directory(assigned['cohort_id'])
        spec, _ = cohort.binding(assigned['cohort_id'])
    p.check(spec['league'] == league and assigned['cohort'] ==
            p.identity(directory / 'cohort_specification.json'),
            'OPERATOR_COHORT_BINDING_DRIFT')
    if league == 'PL':
        if audit:
            result = phase19_pl_cycle.verify()
            p.check(result['pairs_admitted'] == 0, 'OPERATOR_PL_PAIR_DRIFT')
            saved = result
        else:
            saved = p.read(phase19_pl_cycle.OUTPUT)
            p.frozen.sealed(saved, 'record_sha256')
            p.check(saved['pairs_admitted'] == 0 and
                    saved['cohort'] == assigned['cohort'], 'OPERATOR_PL_PAIR_DRIFT')
        source_at = p.read(p.frozen.reference(saved['source']))['retrieved_at']
        count, head = 0, None
    else:
        if audit:
            result = (phase19_verifier.nonpl(league) if full_replay else
                      phase19_verifier._stored_nonpl(league))
            count, head = result['pairs'], result['journal_head']
            source_at = result['source_retrieved_at']
        else:
            certificate = p.read(phase19_verifier.VERIFIED_DIR / (league + '.json'))
            p.frozen.sealed(certificate, 'record_sha256')
            p.check(certificate['verified'] is True and certificate['cohort_id'] ==
                    assigned['cohort_id'], 'OPERATOR_CERTIFICATE_INVALID')
            p.frozen.reference(certificate['source'])
            p.frozen.reference(certificate['qualification'])
            rows = cohort.journal(assigned['cohort_id'])
            p.check(all(event['kind'] == 'pair' for event in rows),
                    'OPERATOR_UNEXPECTED_COHORT_EVENT')
            count, head = len(rows), rows[-1]['record_sha256'] if rows else None
            p.check(certificate['pairs'] == count and certificate['journal_head'] == head,
                    'OPERATOR_CERTIFICATE_HEAD_DRIFT')
            source_at = certificate['source_retrieved_at']
    p.check(sealed_row['verified'] is True and sealed_row['pairs'] == count and
            sealed_row['journal_head'] == head and assigned['cohort_id'] == spec['cohort_id'],
            'OPERATOR_PHASE19_VERIFIED_HEAD_DRIFT')
    return {'cohort_id': assigned['cohort_id'], 'active_model': assigned['active_bundle_id'],
            'candidate_model': assigned['candidate_bundle_id'], 'recorded_predictions': count,
            'journal_head': head, 'source_retrieved_at': source_at, 'replay': 'VERIFIED'}


def _historical(league):
    if league == 'PL':
        return {'state': 'NO_PHASE12_COHORT', 'incidents': []}
    events = p.journal(HISTORICAL, league)
    incidents = [{'sequence': event['sequence'], 'code': event['payload']['code'],
                  'fixture_id': event['payload']['fixture_id'],
                  'record_sha256': event['record_sha256']}
                 for event in events if event['kind'] == 'incident' and
                 event['payload']['protocol_violation']]
    return {'state': ('BLOCKED_PROTOCOL_VIOLATION / UNRESOLVED'
                      if league == 'FL1' and any(row['code'] == 'ARTIFACT_DRIFT'
                                                  for row in incidents) else
                      'BLOCKED_PROTOCOL_VIOLATION' if incidents else
                      'NO RECORDED VIOLATION; NOT CLEARED'),
            'incidents': incidents, 'journal_count': len(events)}


def snapshot(*, audit=False, full_replay=False):
    """A failed league stays local; global integrity failures block shared actions."""
    authority = _attempt(_authority)
    p19_integrity = _attempt(phase19_integrity.verify)
    p20_authority = _attempt(phase20.authorization)
    p20_integrity = (_attempt(phase20.integrity) if p20_authority['verified'] else
                     {'verified': False, 'error': 'PHASE20_AUTHORIZATION_INVALID'})
    rows = []
    for league in LEAGUES:
        historical = _attempt(lambda league=league: _historical(league))
        current = (_attempt(lambda league=league: _current(
            league, *authority['value'], audit=audit, full_replay=full_replay))
            if authority['verified'] else {'verified': False, 'error': authority['error']})
        settlement = (_attempt(lambda league=league: _settlement_status(league))
                      if p20_authority['verified'] else
                      {'verified': False, 'error': p20_authority['error']})
        current_ok = (current['verified'] and p19_integrity['verified'] and
                      p19_integrity['value']['verified'])
        result = settlement.get('value', {})
        pairs = current.get('value', {}).get('recorded_predictions', 0)
        settlement_ok = (current_ok and settlement['verified'] and
                         p20_integrity['verified'] and
                         p20_integrity['value']['verified'] and pairs > 0)
        rows.append({'league': league, 'current': current, 'historical': historical,
                     'settlement': settlement, 'current_lineage_verified': current_ok,
                     'recorded_predictions': pairs if current_ok else None,
                     'settled': result.get('settled') if settlement['verified'] else None,
                     'pending': (result.get('pending') if result.get('pending') is not None else
                                 pairs - result['settled']) if settlement['verified'] and current_ok else None,
                     'settlement_eligibility': ('ELIGIBLE' if settlement_ok else
                         'NO_RECORDED_PREDICTION' if pairs == 0 and current_ok else
                         'BLOCKED_CURRENT_VERIFICATION' if not current_ok else
                         'BLOCKED_PHASE20_INTEGRITY' if not p20_integrity['verified'] or
                             not p20_integrity['value']['verified'] else
                         'BLOCKED_SETTLEMENT_VERIFICATION')})
    return {'phase19_integrity': p19_integrity, 'phase20_integrity': p20_integrity,
            'leagues': rows, 'new_prediction_cycle_authorized': False,
            'scheduling_authorized': False, 'promotion_authorized': False}


def _settlement_status(league):
    """Phase 20 journal owns settled counts; deep score replay belongs to audit."""
    rows = phase20.journal(league)
    if rows:
        return phase20.verify(league)
    return {'verified': True, 'settled': len(rows),
            'pending': None, 'settled_fixture_ids': [row['payload']['fixture_id'] for row in rows]}


def predictions(*, league=None, fixture_id=None, all_rows=False, limit=25, now=None):
    auth_report = _attempt(_authority)
    integrity = _attempt(phase19_integrity.verify)
    p.check(auth_report['verified'], auth_report.get('error', 'PHASE19_AUTHORIZATION_FAILED'))
    p.check(integrity['verified'] and integrity['value']['verified'],
            'PHASE19_PROTECTED_INTEGRITY_FAILED')
    auth, report = auth_report['value']
    at = now or datetime.now(timezone.utc)
    result, errors = [], []
    for name in ((league,) if league else LEAGUES):
        verified = _attempt(lambda name=name: _current(name, auth, report))
        if not verified['verified']:
            errors.append({'league': name, 'error': verified['error']})
            continue
        if name == 'PL':
            continue
        assignment = next(x for x in auth['cohorts'] if x['league'] == name)
        settled = _attempt(lambda name=name: _settlement_status(name))
        settled_ids = set(settled['value']['settled_fixture_ids']) if settled['verified'] else set()
        for event in cohort.journal(assignment['cohort_id']):
            p.check(event['kind'] == 'pair', 'OPERATOR_UNEXPECTED_COHORT_EVENT')
            pair = event['payload']['pair']
            if fixture_id and pair['fixture_id'] != fixture_id:
                continue
            state = ('SETTLEMENT STATUS UNVERIFIED' if not settled['verified'] else
                     'SETTLED' if pair['fixture_id'] in settled_ids else
                     'UPCOMING' if p.instant(pair['kickoff_at']) > at else
                     'KICKED_OFF_UNSETTLED')
            result.append({'league': name, 'cohort_id': assignment['cohort_id'],
                'state': 'RECORDED PROSPECTIVE PREDICTION',
                'fixture_id': pair['fixture_id'], 'home_team': pair['active']['fixture']['home_team'],
                'away_team': pair['active']['fixture']['away_team'],
                'kickoff_at': pair['kickoff_at'], 'recorded_at': pair['created_at'],
                'fixture_state': state, 'pair_sha256': pair['pair_sha256'],
                'event_sha256': event['record_sha256'], 'replay': 'VERIFIED',
                'active': pair['active']['prediction_output'],
                'candidate': pair['candidate']['prediction_output']})
    result.sort(key=lambda row: (row['kickoff_at'], row['league'], row['fixture_id']))
    return {'total_matching': len(result), 'predictions': result if all_rows else result[:limit],
            'errors': errors, 'pl_state': 'NO RECORDED PREDICTION',
            'new_prediction_cycle_authorized': False}


def new_predictions(*, league=None, fixture_id=None, all_rows=False, limit=25, now=None):
    """Show separately authorized prospective records after offline replay."""
    if league == 'PL':
        return {'total_matching': 0, 'predictions': [],
                'errors': [{'league': 'PL', 'error': 'UNSUPPORTED_CAUSAL_INPUTS'}],
                'settlement_authorized': False, 'promotion_authorized': False}
    next_collection.authorization()
    at = now or datetime.now(timezone.utc)
    result, errors = [], []
    for name in ((league,) if league else next_collection.LEAGUES):
        replay = _attempt(lambda name=name: next_collection.verify(name))
        if not replay['verified'] or not replay['value']['verified']:
            errors.append({'league': name, 'error': replay.get('error', 'REPLAY_FAILED')})
            continue
        for event in next_collection.journal(name):
            pair = event['payload']['pair']
            if fixture_id and pair['fixture_id'] != fixture_id:
                continue
            result.append({'league': name, 'lineage': 'prospective-20260924-v1',
                'state': 'RECORDED PROSPECTIVE PREDICTION — NEW LINEAGE',
                'fixture_id': pair['fixture_id'],
                'home_team': pair['active']['fixture']['home_team'],
                'away_team': pair['active']['fixture']['away_team'],
                'kickoff_at': pair['kickoff_at'], 'recorded_at': pair['created_at'],
                'fixture_state': ('UPCOMING' if p.instant(pair['kickoff_at']) > at else
                                  'KICKED_OFF'),
                'pair_sha256': pair['pair_sha256'],
                'event_sha256': event['record_sha256'], 'replay': 'VERIFIED',
                'active': pair['active']['prediction_output'],
                'candidate': pair['candidate']['prediction_output']})
    result.sort(key=lambda row: (row['kickoff_at'], row['league'], row['fixture_id']))
    return {'total_matching': len(result),
            'predictions': result if all_rows else result[:limit],
            'errors': errors, 'settlement_authorized': False,
            'promotion_authorized': False}


def performance():
    authority = _attempt(phase20.authorization)
    rows = []
    for league in LEAGUES:
        result = (_attempt(lambda league=league: _settlement_status(league))
                  if authority['verified'] else
                  {'verified': False, 'error': authority['error']})
        value = result.get('value', {})
        pairs = (value.get('pairs') if value.get('pairs') is not None else
                 authority['value']['pair_counts'][league] if authority['verified'] else None)
        settled = value.get('settled')
        rows.append({'league': league, 'verified': result['verified'],
                     'error': result.get('error'), 'pairs': pairs,
                     'settled': settled, 'pending': (value.get('pending') if value.get('pending') is not None
                                                    else pairs - settled if pairs is not None and
                                                    settled is not None else None),
                     'descriptive_metrics': value.get('descriptive_metrics') if
                         value.get('settled') else None,
                     'evidence': ('NO SETTLED EVIDENCE' if value.get('settled') == 0 else
                                  'DESCRIPTIVE ONLY' if result['verified'] else 'UNVERIFIED')})
    return {'leagues': rows, 'promotion_authorized': False,
            'model_recommendation': None}


def settle(league):
    """Explicit manual command; current integrity refusal occurs before acquisition."""
    p.check(league in LEAGUES, 'UNKNOWN_LEAGUE')
    if league == 'PL':
        return {'status': 'BLOCKED', 'league': league,
                'reason': 'NO RECORDED PREDICTION'}
    authority = _authority()
    _current(league, *authority)
    p.check(phase19_integrity.verify()['verified'], 'PHASE19_PROTECTED_INTEGRITY_FAILED')
    phase20.authorization()
    integrity = phase20.integrity()
    if not integrity['verified']:
        errors = integrity.get('errors', [])
        return {'status': 'BLOCKED', 'league': league,
                'reason': 'SETTLEMENT BLOCKED: INTEGRITY',
                'detail': ('Phase 20 approved boundary failed: ' + ', '.join(errors[:3])
                           if errors else 'Phase 20 approved boundary could not be verified.'),
                'boundary_errors': errors,
                'unapproved_additions': (integrity.get('unauthorized_additions', 0) +
                                         integrity.get('unknown_additions',
                                                       len(integrity.get('added', []))) +
                                         integrity.get('unfinished_additions', 0)),
                'changed': len(integrity.get('changed', [])),
                'missing': len(integrity.get('missing', []))}
    p.check(phase20.verify(league)['verified'], 'PHASE20_SETTLEMENT_REPLAY_FAILED')
    acquired = phase20.acquire(league, authorized=True)
    if acquired['status'] != 'RETRIEVED':
        return {'status': 'BLOCKED', 'league': league,
                'reason': 'RESULT SOURCE UNAVAILABLE', 'source': acquired['source']}
    outcome = phase20.settle(league, p.ROOT / acquired['source']['path'], authorized=True)
    return {'status': 'SETTLED' if outcome['accepted'] else 'NO NEW SETTLEMENT',
            'league': league, 'result': outcome}

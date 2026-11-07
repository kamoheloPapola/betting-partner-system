"""Version 2 fixture-revision rule. Pure, league-generic, and result-blind."""
from src.evaluation import phase12_prospective as p

VERSION = 'ffps.fixture_revision.v2'
IDENTITY = ('league', 'season', 'fixture_id', 'provider_fixture_id',
            'home_id', 'away_id', 'home_team', 'away_team')
SCHEDULED = ('SCHEDULED', 'TIMED')
TERMINAL = ('CANCELLED', 'CANCELED', 'ABANDONED', 'POSTPONED', 'SUSPENDED')


def rule():
    """The static scientific rule, frozen before any new pair is captured."""
    return p.seal({'schema_version': VERSION,
        'implementation': [p.identity(p.ROOT / name) for name in
            ('src/evaluation/phase16_revision.py', 'src/evaluation/phase16_cohort.py')],
        'identity_fields': list(IDENTITY),
        'source_versions': 'immutable authoritative bytes, re-normalized by bound source adapter; sort by retrieval instant; equal-time disagreement blocks',
        'selection_policy': 'consider every qualifying future fixture independently; no earliest-fixture limiter',
        'cohort_order': 'admission order immutable; kickoff-sorted evaluation order must not change',
        'revision_policy': 'identity fixed; each observed kickoff must preserve pair causality, selection order and fixed horizon',
        'unknown_policy': 'fail closed; no manual override, result-aware readmission or fixture substitution',
        'statuses': {'pre_match': list(SCHEDULED), 'finished': 'FINISHED', 'terminal': list(TERMINAL)},
        'classification_precedence': ['INSUFFICIENT_EVIDENCE', 'MATERIAL_FIXTURE_REVISION',
            'CAUSALITY_VIOLATION', 'ORDERING_VIOLATION', 'HORIZON_VIOLATION',
            'PERMITTED_KICKOFF_REVISION', 'UNCHANGED'],
        'historical_protocols_unchanged': True, 'activation_authorized': False}, 'protocol_sha256')


def _before(left, right):
    return p.instant(left) < p.instant(right)


def _order(members, fixture_id, kickoff):
    values = [(row['fixture_id'], kickoff if row['fixture_id'] == fixture_id else row['kickoff_at'])
              for row in members]
    p.check(len(values) == len({name for name, _ in values}), 'DUPLICATE_COHORT_FIXTURE')
    return [name for name, _ in sorted(values, key=lambda row: (p.instant(row[1]), row[0]))]


def classify(protocol, cohort, pair, versions, members):
    """Classify authenticated normalized versions; never mutates a cohort.

    The caller must reconstruct each normalized fixture from its archived raw
    source. The classifier deliberately ignores all scores and result metrics.
    """
    p.check(protocol == rule(), 'UNKNOWN_REVISION_PROTOCOL')
    p.check(cohort['revision_rule_sha256'] == protocol['protocol_sha256'], 'COHORT_PROTOCOL_SUBSTITUTION')
    p.check(pair['fixture_id'] in {row['fixture_id'] for row in members}, 'PAIR_NOT_IN_COHORT')
    p.check(pair['league'] == cohort['league'] and pair['season'] == cohort['season'], 'CROSS_LEAGUE_PAIR')
    p.check(pair['protocol_sha256'] == cohort['protocol_sha256'], 'PAIR_PROTOCOL_SUBSTITUTION')
    p.check(p.instant(cohort['start_at']) <= p.instant(pair['admission_at']) and
            _before(pair['admission_at'], pair['kickoff_at']), 'RETROSPECTIVE_OR_PRE_PROTOCOL_PAIR')
    p.check(pair['prediction_at'] == pair['active_prediction_at'] == pair['candidate_prediction_at'],
            'PAIRED_PREDICTION_TIME_MISMATCH')
    p.check(pair['causal_cutoff'] == pair['active_causal_cutoff'] == pair['candidate_causal_cutoff'],
            'PAIRED_CAUSAL_CUTOFF_MISMATCH')
    ordered = sorted(versions, key=lambda row: (p.instant(row['source'].get('retrieved_at') or '0001-01-01T00:00:00+00:00'),
                                               p.digest(row['source'])))
    reasons, checks, accepted = [], [], []
    if not ordered:
        reasons.append('MISSING_ORIGINAL_SOURCE_VERSION')
    else:
        first = ordered[0]
        first_fixture = first.get('fixture') or {}
        if any(first_fixture.get(key) != pair.get(key) for key in IDENTITY) or \
                first_fixture.get('kickoff_at') is None or not first['source'].get('retrieved_at') or \
                p.instant(first_fixture['kickoff_at']) != p.instant(pair['kickoff_at']) or \
                not p.instant(first['source']['retrieved_at']) <= p.instant(pair['prediction_at']) <= p.instant(pair['admission_at']):
            reasons.append('MISSING_OR_INCOMPATIBLE_ADMISSION_SOURCE')
    preceding = pair['kickoff_at']
    initial_order = _order(members, pair['fixture_id'], pair['kickoff_at'])
    for i, row in enumerate(ordered):
        event, source = row.get('fixture') or {}, row['source']
        observed = source.get('retrieved_at')
        status = event.get('status')
        identity_ok = all(event.get(key) == pair.get(key) for key in IDENTITY)
        kickoff = event.get('kickoff_at')
        known = kickoff is not None
        source_ok = bool(observed and isinstance(source.get('raw'), dict) and
                         {'path', 'sha256', 'size_bytes'} <= source['raw'].keys())
        simultaneous_conflict = bool(i > 0 and observed and ordered[i-1]['source'].get('retrieved_at') and \
            p.instant(observed) == p.instant(ordered[i-1]['source']['retrieved_at']) and \
            event != ordered[i-1]['fixture'])
        duplicate = i > 0 and event == ordered[i-1]['fixture']
        material = not identity_ok or status in TERMINAL
        is_changed = known and p.instant(kickoff) != p.instant(preceding)
        observation_timing = bool(observed and known and (
            (_before(observed, kickoff) if status in SCHEDULED else
             (status == 'FINISHED' and not is_changed and _before(kickoff, observed)))))
        timing = bool(known and observed and _before(pair['prediction_at'], preceding) and \
            _before(pair['prediction_at'], kickoff) and \
            _before(pair['causal_cutoff'], preceding) and _before(pair['causal_cutoff'], kickoff) and \
            _before(pair['maximum_source_event_at'], pair['causal_cutoff']) and \
            all(p.instant(cutoff) <= p.instant(pair['prediction_at']) and
                _before(cutoff, preceding) and _before(cutoff, kickoff)
                for cutoff in pair['feature_cutoffs']) and \
            _before(pair['admission_at'], kickoff) and observation_timing)
        horizon = known and p.instant(cohort['start_at']) <= p.instant(kickoff) < p.instant(cohort['end_at'])
        order_now = _order(members, pair['fixture_id'], kickoff) if known else None
        ordering = order_now == initial_order
        stale_at_admission = bool(i > 0 and observed and p.instant(observed) <= p.instant(pair['admission_at']) and \
            known and p.instant(kickoff) != p.instant(pair['kickoff_at']))
        finished_valid = bool(status != 'FINISHED' or (known and observed and not is_changed and _before(kickoff, observed)))
        checks.append({'source': source, 'fixture': event, 'identity_stable': identity_ok,
            'known_kickoff': known, 'source_identity_present': source_ok,
            'simultaneous_conflict': simultaneous_conflict, 'duplicate_identical': duplicate,
            'status_admissible': status in SCHEDULED or status == 'FINISHED',
            'causality_preserved': bool(timing), 'ordering_preserved': ordering,
            'horizon_preserved': bool(horizon), 'stale_at_admission': stale_at_admission,
            'finished_after_kickoff': finished_valid,
            'kickoff_changed': is_changed,
            'ordered_fixture_ids': order_now})
        if not source_ok or not known or simultaneous_conflict or stale_at_admission or not finished_valid or \
                (status not in SCHEDULED and status != 'FINISHED' and status not in TERMINAL):
            reasons.append('INSUFFICIENT_SOURCE_OR_TIMING_EVIDENCE')
        if material:
            reasons.append('FIXTURE_IDENTITY_OR_TERMINAL_STATUS_CHANGED')
        if not timing and known:
            reasons.append('CAUSALITY_NOT_PRESERVED')
        if not ordering and known:
            reasons.append('COHORT_ORDER_CHANGED')
        if not horizon and known:
            reasons.append('OUTSIDE_FROZEN_HORIZON')
        if known:
            preceding = kickoff
            if not duplicate:
                accepted.append(kickoff)
    if any(x in reasons for x in ('MISSING_ORIGINAL_SOURCE_VERSION', 'MISSING_OR_INCOMPATIBLE_ADMISSION_SOURCE',
                                   'INSUFFICIENT_SOURCE_OR_TIMING_EVIDENCE')):
        outcome = 'INSUFFICIENT_EVIDENCE'
    elif 'FIXTURE_IDENTITY_OR_TERMINAL_STATUS_CHANGED' in reasons:
        outcome = 'MATERIAL_FIXTURE_REVISION'
    elif 'CAUSALITY_NOT_PRESERVED' in reasons:
        outcome = 'CAUSALITY_VIOLATION'
    elif 'COHORT_ORDER_CHANGED' in reasons:
        outcome = 'ORDERING_VIOLATION'
    elif 'OUTSIDE_FROZEN_HORIZON' in reasons:
        outcome = 'HORIZON_VIOLATION'
    else:
        outcome = ('PERMITTED_KICKOFF_REVISION' if any(
            p.instant(value) != p.instant(pair['kickoff_at']) for value in accepted) else 'UNCHANGED')
    return p.seal({'schema_version': 'ffps.fixture_revision_classification.v2',
        'revision_rule_sha256': protocol['protocol_sha256'],
        'cohort_protocol_sha256': cohort['protocol_sha256'], 'cohort_id': cohort['cohort_id'],
        'league': cohort['league'], 'fixture_id': pair['fixture_id'],
        'pair_identity': pair, 'evaluated_source_through': ordered[-1]['source'].get('retrieved_at') if ordered else None,
        'observed_versions': checks, 'original_order': initial_order,
        'classification': outcome, 'reasons': sorted(set(reasons)),
        'admissible_for_new_cohort': outcome in ('UNCHANGED', 'PERMITTED_KICKOFF_REVISION'),
        'activation_authorized': False}, 'classification_sha256')

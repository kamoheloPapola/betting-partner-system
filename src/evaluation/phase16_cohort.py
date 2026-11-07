"""Isolated v2 prospective cohort. Historical Phase12-15 consumers are untouched."""
from datetime import timedelta
from pathlib import Path
import re

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_revision as revision
from src.evaluation.phase12_operations import put
from src.ml.bundle_activation import state_lock

RUN = p.ROOT / 'baselines/phase16/phase16-fixture-revision-20260923-v1'
RULE_FILE = RUN / 'fixture_revision_rule_v2.json'


def _directory(cohort_id):
    p.check(bool(re.fullmatch(r'[A-Z0-9][A-Za-z0-9_-]{2,79}', str(cohort_id))), 'INVALID_COHORT_ID')
    target = (RUN / 'cohorts' / cohort_id).resolve()
    p.check(target.parent == RUN / 'cohorts', 'COHORT_PATH_ESCAPE')
    return target


def freeze_rule():
    return put(RULE_FILE, revision.rule())


def load_rule():
    row = p.read(RULE_FILE)
    p.frozen.sealed(row, 'protocol_sha256')
    p.check(row == revision.rule(), 'REVISION_RULE_IMPLEMENTATION_DRIFT')
    return row


def initialize(cohort_id, league):
    """Publish an empty new cohort, with no Phase11/12 pair prefix."""
    p.check(league in p.source.LEAGUES, 'UNQUALIFIED_CURRENT_SOURCE_ADAPTER')
    directory = _directory(cohort_id)
    with state_lock(directory / 'operation.lock'):
        if (directory / 'cohort_specification.json').exists():
            spec, _ = binding(cohort_id)
            p.check(spec['league'] == league, 'COHORT_LEAGUE_SUBSTITUTION')
            from src.evaluation.phase16_verifier import verify_cohort
            verify_cohort(cohort_id)
            return spec
        rule = load_rule()
        old, _, controller, _, _ = p.frozen.load_protocol(p.PHASE11, league)
        frozen_at = p.now()
        start_at = (p.instant(frozen_at) + timedelta(seconds=2)).isoformat()
        p.check(p.instant(start_at) < p.instant(old['end_at']), 'NO_FUTURE_FIXED_HORIZON')
        old_ref = p.identity(p.PHASE11 / league / 'prospective_protocol.json')
        protocol = p.seal({**{k: v for k, v in old.items() if k != 'protocol_sha256'},
            'schema_version': 'ffps.league_prospective_protocol.v2', 'phase': 16,
            'frozen_at': frozen_at, 'start_at': start_at,
            'cohort_path': directory.relative_to(p.ROOT).as_posix(),
            'first_collection_policy': rule['selection_policy'],
            'revision_rule': p.identity(RULE_FILE), 'supersedes_without_reinterpretation': old_ref,
            'promotion_authorized': False}, 'protocol_sha256')
        protocol_path = directory / 'prospective_protocol.json'
        put(protocol_path, protocol)
        spec = p.seal({'schema_version': 'ffps.phase16_clean_cohort.v1', 'cohort_id': cohort_id,
            'league': league, 'season': old['season'], 'created_at': frozen_at,
            'start_at': start_at, 'end_at': old['end_at'],
            'protocol': p.identity(protocol_path), 'protocol_sha256': protocol['protocol_sha256'],
            'revision_rule': p.identity(RULE_FILE), 'revision_rule_sha256': rule['protocol_sha256'],
            'prior_protocol': old_ref, 'source_qualification': old['qualification'],
            'active_bundle_id': controller.legacy_id, 'candidate_bundle_id': controller.candidate_id,
            'historical_phase12_cohort': p.identity(p.ROOT / 'baselines/phase12/phase12-prospective-20260908-v1/run_specification.json'),
            'initial_pair_prefix': [], 'initial_journal_head': None,
            'admission_policy': 'only predictions created under this v2 protocol after start_at',
            'activation_authorized': False, 'production_pointer_mutation_authorized': False}, 'record_sha256')
        put(directory / 'cohort_specification.json', spec)
        return spec


def binding(cohort_id):
    directory = _directory(cohort_id)
    spec = p.read(directory / 'cohort_specification.json')
    p.frozen.sealed(spec, 'record_sha256')
    p.check(spec['schema_version'] == 'ffps.phase16_clean_cohort.v1' and
            spec['cohort_id'] == cohort_id and spec['league'] in p.source.LEAGUES and
            spec['initial_pair_prefix'] == [] and spec['initial_journal_head'] is None,
            'UNKNOWN_OR_NONEMPTY_COHORT_ORIGIN')
    rule = load_rule()
    p.check(spec['revision_rule'] == p.identity(RULE_FILE) and
            spec['revision_rule_sha256'] == rule['protocol_sha256'], 'COHORT_RULE_SUBSTITUTION')
    protocol_path = p.frozen.reference(spec['protocol'])
    p.check(protocol_path == directory / 'prospective_protocol.json', 'COHORT_PROTOCOL_PATH_MISMATCH')
    protocol = p.read(protocol_path)
    p.frozen.sealed(protocol, 'protocol_sha256')
    p.check(protocol['schema_version'] == 'ffps.league_prospective_protocol.v2' and
            protocol['protocol_sha256'] == spec['protocol_sha256'] and
            protocol['revision_rule'] == spec['revision_rule'] and
            protocol['cohort_path'] == directory.relative_to(p.ROOT).as_posix() and
            protocol['league'] == spec['league'] and protocol['season'] == spec['season'] and
            protocol['start_at'] == spec['start_at'] and protocol['end_at'] == spec['end_at'] and
            protocol['active_bundle_id'] == spec['active_bundle_id'] and
            protocol['candidate_bundle_id'] == spec['candidate_bundle_id'], 'COHORT_PROTOCOL_SUBSTITUTION')
    p.frozen.reference(spec['prior_protocol'])
    p.frozen.reference(spec['source_qualification'])
    p.frozen.reference(spec['historical_phase12_cohort'])
    p.check(p.instant(spec['created_at']) < p.instant(spec['start_at']) < p.instant(spec['end_at']),
            'INVALID_CLEAN_HORIZON')
    return spec, protocol


def journal(cohort_id):
    directory = _directory(cohort_id) / 'journal'
    rows, previous = [], None
    for sequence, path in enumerate(sorted(directory.glob('*.json')), 1):
        p.check(path.name == f'{sequence:08d}.json', 'V2_JOURNAL_GAP_OR_REORDER')
        row = p.read(path)
        p.frozen.sealed(row, 'record_sha256')
        p.check(row['sequence'] == sequence and row['previous'] == previous and
                row['kind'] in ('pair', 'observation', 'settlement'), 'V2_JOURNAL_CHAIN_MISMATCH')
        rows.append(row)
        previous = row['record_sha256']
    return rows


def _append(cohort_id, league, kind, payload):
    rows = journal(cohort_id)
    row = p.seal({'sequence': len(rows)+1,
        'previous': rows[-1]['record_sha256'] if rows else None,
        'league': league, 'kind': kind, 'created_at': p.now(), 'payload': payload})
    p.persist_new(_directory(cohort_id) / 'journal' / f'{len(rows)+1:08d}.json', row)
    return row


def pair_facts(pair, protocol):
    """Map a replayed non-PL pair into the generic revision rule interface."""
    q = p.read(p.frozen.reference(pair['qualification']))
    target = next((row for row in q['assessment']['upcoming'] if row['fixture_id'] == pair['fixture_id']), None)
    p.check(target is not None, 'PAIR_TARGET_MISSING')
    active, candidate = pair['active'], pair['candidate']
    result = {key: target[key] for key in revision.IDENTITY}
    result.update({'protocol_sha256': pair['protocol_sha256'], 'kickoff_at': pair['kickoff_at'],
        'admission_at': pair['created_at'],
        'prediction_at': active['fixture']['prediction_timestamp'],
        'active_prediction_at': active['fixture']['prediction_timestamp'],
        'candidate_prediction_at': candidate['fixture']['prediction_timestamp'],
        'causal_cutoff': active['data_provenance']['causal_cutoff'],
        'active_causal_cutoff': active['data_provenance']['causal_cutoff'],
        'candidate_causal_cutoff': candidate['data_provenance']['causal_cutoff'],
        'maximum_source_event_at': max((active['data_provenance']['maximum_source_event_timestamp'],
                                       candidate['data_provenance']['maximum_source_event_timestamp']), key=p.instant),
        'feature_cutoffs': [item['causal_cutoff'] for side in (active, candidate)
                            for item in side['feature_provenance']['model_inputs']]})
    return result


def _normalize(source, protocol, qualification, fixture_id, original=None):
    p.check(source.get('status') == 'RETRIEVED', 'SOURCE_NOT_RETRIEVED')
    events = p.current_events(source, protocol, qualification)
    matches = [event for event in events if event['fixture_id'] == fixture_id]
    if len(matches) == 1:
        return matches[0]
    p.check(not matches, 'DUPLICATE_PROVIDER_FIXTURE')
    if original is not None:
        replacements = [event for event in events if all(event.get(key) == original.get(key)
                        for key in ('league', 'season', 'home_id', 'away_id'))]
        p.check(len(replacements) <= 1, 'AMBIGUOUS_PROVIDER_REPLACEMENT')
        if replacements:
            return replacements[0]
    return None


def capture_pair(cohort_id, qualification_ref, fixture_id, source):
    """Create a *new* prospective prediction using archived qualified source bytes.

    Caller must use admit_pair before kickoff. No old prediction can pass the
    new protocol hash and start time. No scheduler or network work occurs here.
    """
    from src.evaluation import phase10_shadow as capture
    from src.evaluation import phase10_current_data as current
    from src.evaluation.phase16_verifier import verify_cohort
    verify_cohort(cohort_id)
    spec, protocol = binding(cohort_id)
    _, _, controller, original, original_index = p.frozen.load_protocol(p.PHASE11, spec['league'])
    q, index, _ = p.frozen.load_qualification(p.frozen.reference(qualification_ref))
    p.evidence11.replay_source_evidence(q)
    p.pair_qualification({'qualification': qualification_ref},
        {'qualification_ref': spec['source_qualification'], 'value': original}, original_index)
    target = next((event for event in q['assessment']['upcoming'] if event['fixture_id'] == fixture_id), None)
    p.check(target is not None, 'UNKNOWN_TARGET_FIXTURE')
    p.check(_normalize(source, protocol, q, fixture_id) == target, 'CURRENT_SOURCE_TARGET_MISMATCH')
    at = p.now()
    p.check(p.instant(protocol['start_at']) <= p.instant(at) < p.instant(target['kickoff_at']) <
            p.instant(protocol['end_at']), 'V2_CAPTURE_OUTSIDE_HORIZON')
    matrix, fixture = p.source.target_matrix(q['assessment'], target, as_of=at, source_index=index)
    current.validate_features(matrix, fixture, q['ordered_features'], at)
    fixture = fixture.copy()
    fixture['kickoff_utc'] = target['kickoff_at']
    active, candidate = [controller.owner.load(controller.bundle(bundle))
                         for bundle in (controller.legacy_id, controller.candidate_id)]
    prediction_at = p.now()
    records = [capture.capture(matrix, fixture, suite, spec['league'], prediction_at)
               for suite in (active, candidate)]
    pair = p.seal({'league': spec['league'], 'protocol_sha256': protocol['protocol_sha256'],
        'fixture_id': fixture_id, 'created_at': p.now(), 'kickoff_at': target['kickoff_at'],
        'qualification': qualification_ref, 'active': records[0], 'candidate': records[1]}, 'pair_sha256')
    p.verify_pair_features(pair, q, index, protocol, controller)
    return pair


def admit_pair(cohort_id, pair, source):
    spec, protocol = binding(cohort_id)
    with state_lock(_directory(cohort_id) / 'operation.lock'):
        from src.evaluation.phase16_verifier import verify_cohort
        verify_cohort(cohort_id)
        rows = journal(cohort_id)
        existing = [row for row in rows if row['kind'] == 'pair' and
                    row['payload']['pair']['fixture_id'] == pair['fixture_id']]
        if existing:
            p.check(existing[0]['payload']['pair'] == pair, 'CONFLICTING_V2_PAIR')
            return existing[0]
        p.check(pair['league'] == spec['league'] and pair['protocol_sha256'] == spec['protocol_sha256'] and
                p.instant(spec['start_at']) <= p.instant(pair['created_at']) < p.instant(pair['kickoff_at']) and
                p.instant(p.now()) < p.instant(pair['kickoff_at']), 'RETROSPECTIVE_OR_CROSS_PROTOCOL_PAIR')
        _, _, controller, original, original_index = p.frozen.load_protocol(p.PHASE11, spec['league'])
        q, index, _ = p.frozen.load_qualification(p.frozen.reference(pair['qualification']))
        p.pair_qualification({'qualification': pair['qualification']},
            {'qualification_ref': spec['source_qualification'], 'value': original}, original_index)
        p.verify_pair_features(pair, q, index, protocol, controller)
        target = _normalize(source, protocol, q, pair['fixture_id'])
        facts = pair_facts(pair, protocol)
        p.check(target is not None and all(target.get(key) == facts[key] for key in revision.IDENTITY) and
                p.instant(target['kickoff_at']) == p.instant(pair['kickoff_at']) and
                p.instant(source['retrieved_at']) <= p.instant(pair['created_at']), 'ADMISSION_SOURCE_MISMATCH')
        members = [pair_facts(row['payload']['pair'], protocol) for row in rows if row['kind'] == 'pair'] + [facts]
        classification = revision.classify(load_rule(), spec, facts,
            [{'source': source, 'fixture': target}], members)
        p.check(classification['classification'] == 'UNCHANGED', 'ADMISSION_CLASSIFICATION_BLOCKED')
        return _append(cohort_id, spec['league'], 'pair',
            {'pair': pair, 'initial_observation': {'source': source, 'fixture': target},
             'classification': classification})


def append_observation(cohort_id, fixture_id, source):
    spec, protocol = binding(cohort_id)
    with state_lock(_directory(cohort_id) / 'operation.lock'):
        from src.evaluation.phase16_verifier import verify_cohort
        verify_cohort(cohort_id)
        rows = journal(cohort_id)
        admissions = [row for row in rows if row['kind'] == 'pair' and
                      row['payload']['pair']['fixture_id'] == fixture_id]
        p.check(len(admissions) == 1, 'UNKNOWN_V2_ADMISSION')
        pair = admissions[0]['payload']['pair']
        q = p.read(p.frozen.reference(pair['qualification']))
        try:
            target = _normalize(source, protocol, q, fixture_id,
                                pair_facts(pair, protocol))
            error = None
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            target, error = None, str(exc)
        versions = [admissions[0]['payload']['initial_observation']] + [row['payload']['observation']
            for row in rows if row['kind'] == 'observation' and row['payload']['fixture_id'] == fixture_id]
        observation = {'source': source, 'fixture': target, 'normalization_error': error}
        if any(row['source'] == source and row['fixture'] == target for row in versions):
            return {'idempotent': True, 'classification': revision.classify(load_rule(), spec,
                pair_facts(pair, protocol), versions,
                [pair_facts(row['payload']['pair'], protocol) for row in rows if row['kind'] == 'pair'])}
        versions.append(observation)
        result = revision.classify(load_rule(), spec, pair_facts(pair, protocol), versions,
            [pair_facts(row['payload']['pair'], protocol) for row in rows if row['kind'] == 'pair'])
        _append(cohort_id, spec['league'], 'observation',
            {'fixture_id': fixture_id, 'observation': observation, 'classification': result})
        return {'idempotent': False, 'classification': result}


def settle(cohort_id, fixture_id, source):
    """Settle only after a complete permitted version history and finished raw result."""
    append_observation(cohort_id, fixture_id, source)
    spec, protocol = binding(cohort_id)
    with state_lock(_directory(cohort_id) / 'operation.lock'):
        from src.evaluation.phase16_verifier import verify_cohort
        verify_cohort(cohort_id)
        rows = journal(cohort_id)
        pair = next(row['payload']['pair'] for row in rows if row['kind'] == 'pair' and
                    row['payload']['pair']['fixture_id'] == fixture_id)
        versions = [row['payload']['initial_observation'] for row in rows if row['kind'] == 'pair' and
                    row['payload']['pair']['fixture_id'] == fixture_id]
        versions += [row['payload']['observation'] for row in rows if row['kind'] == 'observation' and
                     row['payload']['fixture_id'] == fixture_id]
        result = revision.classify(load_rule(), spec, pair_facts(pair, protocol), versions,
            [pair_facts(row['payload']['pair'], protocol) for row in rows if row['kind'] == 'pair'])
        p.check(result['admissible_for_new_cohort'], 'V2_REVISION_BLOCKS_SETTLEMENT')
        event = _normalize(source, protocol, p.read(p.frozen.reference(pair['qualification'])),
                           fixture_id, pair_facts(pair, protocol))
        p.check(event is not None and event['status'] == 'FINISHED' and
                all(type(event[key]) is int and event[key] >= 0 for key in ('home_goals', 'away_goals')) and
                p.instant(event['kickoff_at']) < p.instant(source['retrieved_at']) <= p.instant(p.now()),
                'INVALID_POST_MATCH_SETTLEMENT')
        existing = [row for row in rows if row['kind'] == 'settlement' and row['payload']['fixture_id'] == fixture_id]
        if existing:
            p.check([existing[0]['payload'][key] for key in ('home_goals', 'away_goals')] ==
                    [event[key] for key in ('home_goals', 'away_goals')], 'CONFLICTING_V2_SETTLEMENT')
            return existing[0]
        payload = p.seal({'fixture_id': fixture_id, 'league': spec['league'],
            'protocol_sha256': spec['protocol_sha256'], 'pair_sha256': pair['pair_sha256'],
            'classification_sha256': result['classification_sha256'],
            'home_goals': event['home_goals'], 'away_goals': event['away_goals'],
            'source': source, 'normalized_result': event,
            'created_at': p.now()}, 'settlement_sha256')
        return _append(cohort_id, spec['league'], 'settlement', payload)

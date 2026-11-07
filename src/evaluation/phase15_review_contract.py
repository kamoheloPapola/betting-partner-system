"""Read-only incident facts and frozen-rule adjudication. No publication or override."""
from src.evaluation import phase13_authorization as a
from src.evaluation import phase14_decision as d14
from src.evaluation import phase14_verifier as v14

RUN = a.p.ROOT / 'baselines/phase15/phase15-scientific-review-20260921-v1'
RULE = 'ffps.frozen_admitted_kickoff_review.v1'
RULE_SOURCES = ('src/evaluation/phase12_prospective.py', 'src/evaluation/phase13_authorization.py',
                'src/evaluation/phase14_decision.py', 'docs/phase12-prospective-operations.md')


def identity(league, fixture):
    a.p.check(league in a.p.source.LEAGUES and str(fixture).isdigit(), 'INVALID_REVIEW_IDENTITY')
    return f'{league}_football_data_org_{int(fixture)}'


def source_references(value):
    refs = {}
    def visit(node):
        if isinstance(node, dict):
            if {'path', 'sha256', 'size_bytes'} <= node.keys():
                a.p.frozen.reference(node)
                refs[node['path']] = {key: node[key] for key in ('path', 'sha256', 'size_bytes')}
            for item in node.values():
                visit(item)
        elif isinstance(node, list):
            for item in node:
                visit(item)
    visit(value)
    return [refs[key] for key in sorted(refs)]


def adjudicate(facts):
    """No new permission: instant equality is the existing Phase12 predicate.

    Factual identity can resolve a supposed revision, but no frozen consumer grants
    review-based clearance of a recorded protocol violation.
    """
    reasons = []
    for field, reason in [('identity_matches', 'FIXTURE_IDENTITY_CHANGED'),
                          ('feature_times_compatible', 'FEATURE_TIMING_INCOMPATIBLE'),
                          ('prediction_times_compatible', 'PREDICTION_TIMING_INCOMPATIBLE'),
                          ('source_retrievals_compatible', 'SOURCE_RETRIEVAL_TIMING_INCOMPATIBLE')]:
        if not facts[field]:
            reasons.append(reason)
    if facts['cohort_order_changed']:
        reasons.append('COHORT_ORDER_CHANGED')
    if reasons:
        outcome = 'BLOCKED_SCIENTIFIC_VIOLATION'
    elif not facts['kickoff_instant_equal']:
        outcome = 'UNRESOLVED_PROTOCOL_CONFLICT'
        reasons = ['ADMITTED_KICKOFF_EQUALITY_VIOLATED', 'NO_FROZEN_KICKOFF_REVISION_RESOLUTION_RULE']
    else:
        outcome = 'RESOLVED_FACTUAL_IDENTITY'
        reasons = ['EXISTING_INSTANT_EQUALITY_PREDICATE_SATISFIED']
    return {'outcome': outcome, 'reason_codes': reasons + ['NO_FROZEN_REVIEW_CLEARANCE_AUTHORITY'],
            'clears_protocol_blocker': False, 'fixture_exclusion_authorized': False,
            'activation_authorized': False}


def reconstruct(league, fixture, *, event_count=None):
    """Caller holds the league operation lock. Replays the entire cohort."""
    fixture_id = identity(league, fixture)
    controller, contract, installed = d14.binding(league)
    before = d14.authority_snapshot(league)
    context = a.p.load_cohort(a.COHORT, league)
    protocol, _, original, pairs, settlements, targets, events = context
    v14.verify_incident_sources(events)
    if event_count is not None:
        a.p.check(type(event_count) is int and 0 <= event_count <= len(events), 'HISTORICAL_REVIEW_PREFIX_MISSING')
        prefix_count = len(a.p.specification(a.COHORT)['leagues'][league]['prefix'])
        events = events[:event_count]
        pairs = pairs[:prefix_count] + [event['payload'] for event in events if event['kind'] == 'pair']
        settlements = [event['payload'] for event in events if event['kind'] == 'settlement']
    matching = [pair for pair in pairs if pair['fixture_id'] == fixture_id]
    a.p.check(len(matching) == 1, 'REVIEW_REQUIRES_EXACT_ADMITTED_PAIR')
    pair = matching[0]
    incidents = [event for event in events if event['kind'] == 'incident' and
                 event['payload']['fixture_id'] == fixture_id and
                 event['payload']['code'] == 'ADMITTED_KICKOFF_REVISION']
    a.p.check(incidents, 'NO_ADMITTED_KICKOFF_INCIDENT')
    admission = [event for event in events if event['kind'] == 'pair' and event['payload']['fixture_id'] == fixture_id]
    a.p.check(len(admission) == 1, 'ADMISSION_EVENT_MISSING')
    q = a.p.read(a.p.frozen.reference(pair['qualification']))
    index = a.p.read(a.p.frozen.reference(q['source_index']))
    original_target = targets[fixture_id]
    versions = []
    metadata = list(index['sources']) + [event['payload']['source'] for event in incidents]
    # Also preserve all authoritative result observations retained in this league journal.
    metadata += [event['payload']['source'] for event in events if event['kind'] in ('incident', 'settlement')
                 and isinstance(event['payload'].get('source'), dict)]
    seen = set()
    for source in metadata:
        if source.get('provider') != 'football-data.org:v4' or source.get('status') != 'RETRIEVED':
            continue
        if source.get('request') != {'url': f'https://api.football-data.org/v4/competitions/{league}/matches',
                                     'params': {'season': protocol['season']}}:
            continue
        key = a.p.digest(source)
        if key in seen:
            continue
        seen.add(key)
        current = a.p.current_events(source, protocol, original)
        rows = [row for row in current if row['fixture_id'] == fixture_id]
        a.p.check(len(rows) == 1, 'AUTHORITATIVE_FIXTURE_MISSING_OR_DUPLICATED')
        versions.append({'metadata': source, 'fixture': rows[0]})
    versions.sort(key=lambda row: (row['metadata']['retrieved_at'], a.p.digest(row['metadata'])))
    a.p.check(versions, 'AUTHORITATIVE_REVIEW_SOURCES_MISSING')
    a.p.check(any(row['fixture'] == original_target for row in versions), 'ORIGINAL_KICKOFF_SOURCE_MISMATCH')
    revisions = []
    for incident in incidents:
        source = incident['payload']['source']
        rows = [row for row in versions if row['metadata'] == source]
        a.p.check(len(rows) == 1, 'INCIDENT_SOURCE_NOT_RECONSTRUCTED')
        revised = rows[0]['fixture']
        timing = {}
        for label in ('active', 'candidate'):
            record = pair[label]
            data = record['data_provenance']
            timing[label] = {'prediction_at': record['fixture']['prediction_timestamp'],
                'recorded_at': record['recorded_at'], 'causal_cutoff': data['causal_cutoff'],
                'maximum_source_event_timestamp': data['maximum_source_event_timestamp'],
                'feature_cutoffs': [model['causal_cutoff'] for model in record['feature_provenance']['model_inputs']],
                'feature_bundle_sha256': record['feature_provenance']['feature_bundle_sha256'],
                'source_events': data['source_events']}
        original_order = [row['fixture_id'] for row in sorted(pairs, key=lambda row: (a.p.instant(row['kickoff_at']), row['fixture_id']))]
        revised_order = [row['fixture_id'] for row in sorted(pairs, key=lambda row:
            (a.p.instant(revised['kickoff_at'] if row['fixture_id'] == fixture_id else row['kickoff_at']), row['fixture_id']))]
        facts = {'original': original_target, 'revised': revised, 'observed_revision_at': source['retrieved_at'],
            'incident_at': incident['created_at'], 'publication_at': admission[0]['created_at'],
            'pair_created_at': pair['created_at'], 'timing': timing,
            'identity_matches': all(original_target[key] == revised[key] for key in
                ('league', 'season', 'fixture_id', 'provider_fixture_id', 'home_id', 'away_id', 'home_team', 'away_team')),
            'kickoff_instant_equal': a.p.instant(original_target['kickoff_at']) == a.p.instant(revised['kickoff_at']),
            'prediction_times_compatible': all(a.p.instant(row['prediction_at']) <= a.p.instant(row['recorded_at']) <=
                a.p.instant(admission[0]['created_at']) < a.p.instant(revised['kickoff_at']) for row in timing.values()),
            'feature_times_compatible': all(a.p.instant(row['maximum_source_event_timestamp']) < a.p.instant(row['causal_cutoff']) <=
                a.p.instant(row['prediction_at']) < a.p.instant(revised['kickoff_at']) and
                all(a.p.instant(cutoff) <= a.p.instant(row['prediction_at']) for cutoff in row['feature_cutoffs']) and
                all(a.p.instant(event['event_timestamp']) < a.p.instant(row['causal_cutoff']) for event in row['source_events'])
                for row in timing.values()),
            'source_retrievals_compatible': all(a.p.instant(raw['retrieved_at']) <= a.p.instant(pair['created_at']) <
                a.p.instant(revised['kickoff_at']) for raw in index['sources']),
            'original_chronological_order': original_order, 'revised_chronological_order': revised_order,
            'cohort_order_changed': original_order != revised_order,
            'admission_order': [row['fixture_id'] for row in pairs],
            'frozen_predicate': 'phase12_prospective.same_fixture(admitted=True): identical kickoff instant required',
            'no_retraining_or_feature_reconstruction': True}
        revisions.append({'incident': incident, 'facts': facts, 'assessment': adjudicate(facts)})
    snapshot = a.p.specification(a.COHORT)['leagues'][league]
    a.p.check(before == d14.authority_snapshot(league), 'LINEAGE_CHANGED_DURING_REVIEW')
    sources = source_references([pair['qualification'], q['source_index'], index, metadata,
                                 protocol['qualification'], protocol['current_data_contract']])
    outcomes = [row['assessment']['outcome'] for row in revisions]
    outcome = ('BLOCKED_SCIENTIFIC_VIOLATION' if 'BLOCKED_SCIENTIFIC_VIOLATION' in outcomes else
               'UNRESOLVED_PROTOCOL_CONFLICT' if 'UNRESOLVED_PROTOCOL_CONFLICT' in outcomes else 'RESOLVED_FACTUAL_IDENTITY')
    return {'league': league, 'fixture_id': fixture_id, 'provider_fixture_id': int(fixture),
        'protocol': contract['protocol'], 'protocol_sha256': protocol['protocol_sha256'],
        'installation': installed, 'horizon_end': protocol['end_at'],
        'lineage': {'run': a.p.identity(a.COHORT / 'run_specification.json'), 'prefix': snapshot['prefix'],
                    'ordered_events': [a.p.identity(a.COHORT / league / 'journal' / f"{event['sequence']:08d}.json") for event in events],
                    'pair_hashes': [row['pair_sha256'] for row in pairs],
                    'settlement_hashes': [row['settlement_sha256'] for row in settlements]},
        'admission': admission[0], 'subsequent_events': [event for event in events if event['sequence'] > admission[0]['sequence']],
        'qualification': pair['qualification'], 'source_index': q['source_index'], 'sources': sources,
        'observed_versions': versions, 'revisions': revisions, 'replay_verified': True,
        'rule': {'id': RULE, 'frozen_authorities': [a.p.identity(a.p.ROOT / name) for name in RULE_SOURCES],
                 'review_implementation': a.p.identity(a.p.ROOT / 'src/evaluation/phase15_review_contract.py'),
                 'clearance_rule': None, 'fixture_exclusion_rule': None},
        'outcome': outcome, 'clears_protocol_blocker': False, 'activation_authorized': False}


def material(basis, revision, predecessor):
    return a.p.seal({'schema_version': 'ffps.scientific_incident_review.v1',
        'revision': revision, 'predecessor': predecessor, 'basis_sha256': a.p.digest(basis),
        'basis': basis, 'outcome': basis['outcome'], 'clears_protocol_blocker': False,
        'activation_authorized': False}, 'review_sha256')

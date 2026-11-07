"""Read-only v2 verifier; does not import the cohort publisher."""
from pathlib import Path
from datetime import timedelta
from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_revision as revision

RUN = p.ROOT / 'baselines/phase16/phase16-fixture-revision-20260923-v1'


def _rule():
    row = p.read(RUN / 'fixture_revision_rule_v2.json')
    p.frozen.sealed(row, 'protocol_sha256')
    p.check(row == revision.rule(), 'V2_RULE_RECONSTRUCTION_MISMATCH')
    return row


def _facts(pair, target):
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
    p.check(source.get('status') == 'RETRIEVED', 'V2_SOURCE_UNAVAILABLE')
    events = p.current_events(source, protocol, qualification)
    rows = [row for row in events if row['fixture_id'] == fixture_id]
    p.check(len(rows) <= 1, 'V2_DUPLICATE_PROVIDER_FIXTURE')
    if rows:
        return rows[0]
    if original is not None:
        replacements = [row for row in events if all(row.get(key) == original.get(key)
                        for key in ('league', 'season', 'home_id', 'away_id'))]
        p.check(len(replacements) <= 1, 'V2_AMBIGUOUS_PROVIDER_REPLACEMENT')
        if replacements:
            return replacements[0]
    return None


def verify_cohort(cohort_id):
    """Reconstruct the clean protocol, journal, source history and every classifier."""
    p.check(isinstance(cohort_id, str) and cohort_id and all(c.isalnum() or c in '_-' for c in cohort_id),
            'INVALID_V2_COHORT_ID')
    directory = (RUN / 'cohorts' / cohort_id).resolve()
    p.check(directory.parent == RUN / 'cohorts', 'V2_COHORT_PATH_ESCAPE')
    rule = _rule()
    spec = p.read(directory / 'cohort_specification.json')
    p.frozen.sealed(spec, 'record_sha256')
    p.check(spec['schema_version'] == 'ffps.phase16_clean_cohort.v1' and
            spec['cohort_id'] == cohort_id and spec['league'] in p.source.LEAGUES and
            spec['initial_pair_prefix'] == [] and spec['initial_journal_head'] is None and
            spec['revision_rule'] == p.identity(RUN / 'fixture_revision_rule_v2.json') and
            spec['revision_rule_sha256'] == rule['protocol_sha256'] and
            spec['activation_authorized'] is False, 'V2_COHORT_BINDING_INVALID')
    old, _, controller, original, original_index = p.frozen.load_protocol(p.PHASE11, spec['league'])
    old_ref = p.identity(p.PHASE11 / spec['league'] / 'prospective_protocol.json')
    p.check(spec['prior_protocol'] == old_ref and spec['source_qualification'] == old['qualification'] and
            spec['season'] == old['season'] and spec['end_at'] == old['end_at'] and
            spec['start_at'] == (p.instant(spec['created_at']) + timedelta(seconds=2)).isoformat() and
            spec['active_bundle_id'] == controller.legacy_id and
            spec['candidate_bundle_id'] == controller.candidate_id and
            spec['historical_phase12_cohort'] == p.identity(p.ROOT / 'baselines/phase12/phase12-prospective-20260908-v1/run_specification.json'),
            'V2_COHORT_ORIGIN_MISMATCH')
    protocol = p.read(p.frozen.reference(spec['protocol']))
    p.frozen.sealed(protocol, 'protocol_sha256')
    expected = p.seal({**{k: v for k, v in old.items() if k != 'protocol_sha256'},
        'schema_version': 'ffps.league_prospective_protocol.v2', 'phase': 16,
        'frozen_at': spec['created_at'], 'start_at': spec['start_at'],
        'cohort_path': directory.relative_to(p.ROOT).as_posix(),
        'first_collection_policy': rule['selection_policy'],
        'revision_rule': spec['revision_rule'], 'supersedes_without_reinterpretation': old_ref,
        'promotion_authorized': False}, 'protocol_sha256')
    p.check(protocol == expected and spec['protocol_sha256'] == protocol['protocol_sha256'] and
            spec['protocol'] == p.identity(directory / 'prospective_protocol.json'),
            'V2_PROTOCOL_RECONSTRUCTION_MISMATCH')
    events, previous = [], None
    for sequence, path in enumerate(sorted((directory / 'journal').glob('*.json')), 1):
        p.check(path.name == f'{sequence:08d}.json', 'V2_JOURNAL_MEMBER_MISSING')
        event = p.read(path)
        p.frozen.sealed(event, 'record_sha256')
        p.check(event['sequence'] == sequence and event['previous'] == previous and
                event['league'] == spec['league'] and event['kind'] in ('pair', 'observation', 'settlement'),
                'V2_JOURNAL_CHAIN_MISMATCH')
        events.append(event)
        previous = event['record_sha256']
    pairs, histories, settlements = {}, {}, {}
    for event in events:
        value = event['payload']
        if event['kind'] == 'pair':
            pair = value['pair']
            fixture_id = pair['fixture_id']
            p.check(fixture_id not in pairs and pair['protocol_sha256'] == spec['protocol_sha256'] and
                    pair['league'] == spec['league'] and
                    p.instant(spec['start_at']) <= p.instant(pair['created_at']) < p.instant(pair['kickoff_at']),
                    'V2_DUPLICATE_OR_RETROSPECTIVE_PAIR')
            q, index, _ = p.frozen.load_qualification(p.frozen.reference(pair['qualification']))
            p.pair_qualification({'qualification': pair['qualification']},
                {'qualification_ref': spec['source_qualification'], 'value': original}, original_index)
            target = p.verify_pair_features(pair, q, index, protocol, controller)
            observation = value['initial_observation']
            normalized = _normalize(observation['source'], protocol, q, fixture_id)
            p.check(normalized == observation['fixture'] == target and
                    p.instant(observation['source']['retrieved_at']) <= p.instant(pair['created_at']),
                    'V2_PAIR_SOURCE_RECONSTRUCTION_MISMATCH')
            pairs[fixture_id] = (pair, target)
            histories[fixture_id] = [observation]
            members = [_facts(pair, target) for pair, target in pairs.values()]
            computed = revision.classify(rule, spec, _facts(pair, target), histories[fixture_id], members)
            p.check(computed == value['classification'] and computed['classification'] == 'UNCHANGED',
                    'V2_ADMISSION_CLASSIFICATION_MISMATCH')
        elif event['kind'] == 'observation':
            fixture_id = value['fixture_id']
            p.check(fixture_id in pairs, 'V2_ORPHAN_OBSERVATION')
            pair, target = pairs[fixture_id]
            q = p.read(p.frozen.reference(pair['qualification']))
            observation = value['observation']
            try:
                reconstructed = _normalize(observation['source'], protocol, q, fixture_id,
                                           _facts(pair, target))
                error = None
            except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                reconstructed, error = None, str(exc)
            p.check(reconstructed == observation['fixture'] and
                    error == observation.get('normalization_error') and
                    not any(row['source'] == observation['source'] and
                            row['fixture'] == observation['fixture'] for row in histories[fixture_id]),
                    'V2_OBSERVATION_SOURCE_OR_DUPLICATE')
            histories[fixture_id].append(observation)
            members = [_facts(pair, target) for pair, target in pairs.values()]
            computed = revision.classify(rule, spec, _facts(pair, target), histories[fixture_id], members)
            p.check(computed == value['classification'], 'V2_OBSERVATION_CLASSIFICATION_MISMATCH')
        else:
            fixture_id = value['fixture_id']
            p.frozen.sealed(value, 'settlement_sha256')
            p.check(fixture_id in pairs and fixture_id not in settlements and
                    value['league'] == spec['league'] and value['protocol_sha256'] == spec['protocol_sha256'] and
                    value['pair_sha256'] == pairs[fixture_id][0]['pair_sha256'],
                    'V2_SETTLEMENT_SCOPE_MISMATCH')
            pair, target = pairs[fixture_id]
            members = [_facts(pair, target) for pair, target in pairs.values()]
            computed = revision.classify(rule, spec, _facts(pair, target), histories[fixture_id], members)
            normalized = _normalize(value['source'], protocol,
                                   p.read(p.frozen.reference(pair['qualification'])),
                                   fixture_id, _facts(pair, target))
            p.check(computed['admissible_for_new_cohort'] and
                    computed['classification_sha256'] == value['classification_sha256'] and
                    normalized == value['normalized_result'] and normalized['status'] == 'FINISHED' and
                    [normalized[key] for key in ('home_goals', 'away_goals')] ==
                    [value[key] for key in ('home_goals', 'away_goals')] and
                    p.instant(normalized['kickoff_at']) < p.instant(value['source']['retrieved_at']) <=
                    p.instant(value['created_at']), 'V2_SETTLEMENT_RECONSTRUCTION_MISMATCH')
            settlements[fixture_id] = value
    members = [_facts(pair, target) for pair, target in pairs.values()]
    classifications = {fixture_id: revision.classify(rule, spec, _facts(pair, target),
        histories[fixture_id], members) for fixture_id, (pair, target) in pairs.items()}
    score_conflicts = [fixture_id for fixture_id, history in histories.items() if len({
        (row['fixture'].get('home_goals'), row['fixture'].get('away_goals'))
        for row in history if row['fixture'] is not None and row['fixture'].get('status') == 'FINISHED'}) > 1]
    blocked = bool(score_conflicts or any(not row['admissible_for_new_cohort']
                                         for row in classifications.values()))
    return {'verified': True, 'cohort': p.identity(directory / 'cohort_specification.json'),
        'protocol': p.identity(directory / 'prospective_protocol.json'),
        'revision_rule': p.identity(RUN / 'fixture_revision_rule_v2.json'),
        'league': spec['league'], 'pairs': len(pairs), 'settlements': len(settlements),
        'journal_head': previous, 'classifications': classifications,
        'score_conflicts': score_conflicts,
        'scientific_state': 'PROTOCOL_VIOLATION_BLOCKED' if blocked else
            'PHASE16_PROTOCOL_V2_READY_FOR_CLEAN_COHORT' if not events else
            'PROSPECTIVE_COLLECTION_ACTIVE', 'activation_authorized': False}


def verify_regression(path=None):
    """Reconstruct from old sealed lineage and raw bytes, without the publisher."""
    from src.evaluation import phase16_regression_facts as facts
    from src.ml.bundle_activation import canonical_bytes
    target = (Path(path) if path is not None else RUN / 'regression/BL1-565839.json').resolve()
    p.check(target == RUN / 'regression/BL1-565839.json', 'UNBOUND_V2_REGRESSION_PATH')
    supplied = p.read(target)
    p.frozen.sealed(supplied, 'record_sha256')
    expected = facts.reconstruct()
    p.check(supplied == expected and target.read_bytes() == canonical_bytes(supplied),
            'V2_REGRESSION_RECONSTRUCTION_MISMATCH')
    return {'verified': True, 'artifact': p.identity(target),
        'historical_state': supplied['historical_scientific_state'],
        'v2_classification': supplied['classification']['classification'],
        'counterfactual_only': True, 'activation_authorized': False}

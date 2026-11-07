"""Independent consumer: never imports or trusts the Phase14 publisher."""
from pathlib import Path
from src.evaluation import phase13_authorization as a
from src.evaluation import phase12_evidence as e12
from src.ml.league_bundle_activation import LeagueBundleController
from src.ml.bundle_activation import canonical_bytes


def verify_incident_sources(events):
    def walk(value):
        if isinstance(value, dict):
            if {'path', 'sha256'} <= value.keys():
                path = a.p.frozen.reference(value)
                if 'size_bytes' in value:
                    a.p.check(path.stat().st_size == value['size_bytes'], 'INCIDENT_SOURCE_SIZE_MISMATCH')
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for event in events:
        if event['kind'] == 'incident':
            walk(event['payload'].get('source'))


def verify_locked(directory, league):
    directory = Path(directory).resolve()
    a.p.check(a.RUN.name == a.NEXT.name == 'phase13-authorization-20260915-v2', 'PHASE13_V2_REQUIRED')
    a.p.check(directory.parent == a.RUN / league / 'decisions' and not directory.name.startswith('.'),
              'UNBOUND_FINAL_DECISION_DIRECTORY')
    a.p.check(sorted(path.name for path in directory.iterdir()) == ['artifact_index.json', 'prospective_result.json'],
              'FINAL_PUBLICATION_INVENTORY_MISMATCH')
    controller = LeagueBundleController(a.NEXT / league / 'deployment_specification.json', league=league, root=a.p.ROOT)
    contract, installation = a.verify_binding(controller)
    value = a.p.read(directory / 'prospective_result.json')
    index = a.p.read(directory / 'artifact_index.json')
    a.p.frozen.sealed(value, 'decision_sha256')
    a.p.frozen.sealed(index, 'record_sha256')
    for name, data in (('prospective_result.json', value), ('artifact_index.json', index)):
        a.p.check((directory / name).read_bytes() == canonical_bytes(data), 'NONCANONICAL_FINAL_BYTES')
    a.p.check(index == a.p.seal({'decision': a.p.identity(directory / 'prospective_result.json')}),
              'FINAL_INDEX_MISMATCH')
    context = a.p.load_cohort(a.COHORT, league)
    verify_incident_sources(context[6])
    prefix_count = len(a.p.specification(a.COHORT)['leagues'][league]['prefix'])
    for path in sorted((a.COHORT / league / 'checkpoints').glob('*.json')):
        e12.verify_checkpoint(a.p.read(path), context, prefix_count)
    # The frozen Phase13 validator independently rebuilds every scientific field and hash.
    gate = a.gate(controller, directory)
    a.p.check(value['ordered_events'] == context[6] and value['pairs'] == context[3] and
              value['settlements'] == context[4], 'COHORT_CHANGED_DURING_VERIFICATION')
    return {'verified': True, 'league': league, 'decision': a.p.identity(directory / 'prospective_result.json'),
            'index': a.p.identity(directory / 'artifact_index.json'), 'classification': gate['classification'],
            'phase8_scientific_review_eligible': gate['promotion_eligible'], 'activation_authorized': False}


def verify(directory):
    directory = Path(directory).resolve()
    league = directory.parent.parent.name
    a.p.check(league in a.p.source.LEAGUES, 'WRONG_LEAGUE')
    with a.p.state_lock(a.COHORT / league / 'operation.lock'):
        return verify_locked(directory, league)


def verify_readiness(row):
    """Reconstruct a dated complete-cohort refusal independently of its publisher.

    Later legitimate appends are allowed, but cannot erase or reorder the dated prefix.
    """
    a.p.frozen.sealed(row, 'record_sha256')
    league = row['league']
    a.p.check(league in a.p.source.LEAGUES and row['reconstructed'] and not row['published'],
              'REAL_READINESS_NOT_RECONSTRUCTED')
    with a.p.state_lock(a.COHORT / league / 'operation.lock'):
        controller = LeagueBundleController(a.NEXT / league / 'deployment_specification.json', league=league, root=a.p.ROOT)
        contract, installed = a.verify_binding(controller)
        a.p.check(a.RUN.name == a.NEXT.name == 'phase13-authorization-20260915-v2', 'PHASE13_V2_REQUIRED')
        context = a.p.load_cohort(a.COHORT, league)
        verify_incident_sources(context[6])
        protocol, _, _, all_pairs, _, _, events = context
        prefix = len(a.p.specification(a.COHORT)['leagues'][league]['prefix'])
        for path in sorted((a.COHORT / league / 'checkpoints').glob('*.json')):
            e12.verify_checkpoint(a.p.read(path), context, prefix)
        count = len(row['ordered_event_hashes'])
        members = events[:count]
        a.p.check(row['ordered_event_hashes'] == [e['record_sha256'] for e in members], 'READINESS_LINEAGE_CHANGED')
        a.p.check(all(a.p.instant(e['created_at']) <= a.p.instant(row['as_of']) for e in members) and
                  all(a.p.instant(e['created_at']) > a.p.instant(row['as_of']) for e in events[count:]),
                  'READINESS_COHORT_CHERRY_PICKED')
        pairs = all_pairs[:prefix] + [e['payload'] for e in members if e['kind'] == 'pair']
        settlements = [e['payload'] for e in members if e['kind'] == 'settlement']
        incidents = [e['payload'] for e in members if e['kind'] == 'incident']
        a.p.check(row['ordered_pair_hashes'] == [p['pair_sha256'] for p in pairs] and
                  row['ordered_settlement_hashes'] == [s['settlement_sha256'] for s in settlements] and
                  row['incidents'] == incidents, 'READINESS_MEMBERS_CHANGED')
        evaluation = a.p.evaluate(protocol, pairs, settlements, row['as_of'])
        a.p.check(row['evaluation'] == evaluation and row['installation'] == installed and
                  row['protocol'] == contract['protocol'], 'READINESS_EVALUATION_OR_BINDING_CHANGED')
        a.p.check(a.p.instant(row['as_of']) <= a.p.instant(a.p.now()) < a.p.instant(protocol['end_at']) and
                  row['horizon_end'] == protocol['end_at'], 'INITIAL_REPORT_HORIZON_NOT_BLOCKED')
        blockers = [{'code': 'AWAITING_HORIZON', 'detail': protocol['end_at']}]
        blockers += [{'code': 'BLOCKED_PROTOCOL_VIOLATION', 'detail': i} for i in incidents
                     if i['protocol_violation'] or a.p.is_violation(i['code'])]
        if not evaluation['cohort_completeness']:
            blockers.append({'code': 'AWAITING_SETTLEMENTS', 'detail': evaluation['pending_fixture_ids']})
        if not evaluation['fixture_count']:
            blockers.append({'code': 'BLOCKED_NO_SETTLED_OBSERVATIONS',
                             'detail': 'frozen Phase7 rule requires observed outcomes'})
        expected_state = 'BLOCKED_PROTOCOL_VIOLATION' if any(b['code'] == 'BLOCKED_PROTOCOL_VIOLATION' for b in blockers) else 'AWAITING_HORIZON'
        a.p.check(row['blockers'] == blockers and row['state'] == expected_state and
                  row['pairs'] == len(pairs) and row['settled'] == len(settlements) and
                  row['pending'] == len(pairs)-len(settlements), 'READINESS_BLOCKERS_OR_COUNTS_CHANGED')
        a.p.check(not list((a.RUN / league).rglob('prospective_result.json')), 'UNEXPECTED_REAL_FINAL_DECISION')
        return {'verified': True, 'league': league, 'pairs': len(pairs), 'settled': len(settlements),
                'state': row['state'], 'zero_real_decisions': True, 'activation_authorized': False}

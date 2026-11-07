"""Crash-resumable operations around the byte-frozen Phase 12 implementation.

No training or activation imports. The journals remain authoritative. An intent
is published before work and a report seal after all independent league results.
"""
from pathlib import Path
import re
import time
import uuid

from src.evaluation import phase12_prospective as p
from src.evaluation import phase12_evidence as evidence

SOURCES = ('src/evaluation/phase12_operations.py', 'src/evaluation/phase12_operations_evidence.py',
           'scripts/run_phase12_operations.py', 'scripts/register_phase12_task.ps1',
           'docs/phase12-scheduled-operations.md', 'tests/unit/test_phase12_operations.py')
RETRIES = 3


def put(path, value):
    """Complete an interrupted publication without ever replacing sealed bytes."""
    if path.exists():
        p.check(p.read(path) == value, 'OPERATION_RECORD_CONFLICT:' + path.name)
    else:
        p.persist_new(path, value)
    return value


def root(path):
    return p.run_path(path) / 'operational'


def initialize(path):
    directory = root(path)
    config = directory / 'configuration.json'
    if config.exists():
        row = p.read(config)
        p.frozen.sealed(row, 'record_sha256')
        return row
    evidence.integrity(path)
    before = p.snapshot()
    for item in p.run_path(path).rglob('*'):
        if item.is_file() and not item.is_relative_to(directory) and item.suffix not in ('.lock', '.tmp'):
            before[item.relative_to(p.ROOT).as_posix()] = p.source.sha256_file(item)
    for name in evidence.SOURCES:
        before[name] = p.source.sha256_file(p.ROOT / name)
    # The snapshot itself is atomic; interrupted setup reuses its original boundary.
    snapshot = directory / 'protected_before.json'
    if not snapshot.exists():
        p.persist_new(snapshot, before)
    return put(config, p.seal({'schema': 'ffps.phase12_operations.v1',
        'protected_before': p.identity(snapshot), 'implementation': [p.identity(p.ROOT / n) for n in SOURCES],
        'leagues': list(p.source.LEAGUES), 'cadence_minutes': 60, 'attempts': RETRIES,
        'promotion_authorized': False}))


def integrity(path):
    config = initialize(path)
    for ref in config['implementation']:
        p.frozen.reference(ref)
    before = p.read(p.frozen.reference(config['protected_before']))
    changed = [name for name, sha in before.items()
               if not (p.ROOT / name).is_file() or p.source.sha256_file(p.ROOT / name) != sha]
    p.check(not changed, 'OPERATION_PROTECTED_MUTATION:' + ','.join(changed))
    return {**evidence.integrity(path), 'operational_boundary_count': len(before),
            'sealed_phase12_unchanged': True}


def authorization(controller):
    bound = controller.owner.frozen.get('prospective_contract') is not None
    return {'prospective_contract_bound': bound,
            'state': 'SEPARATE_PHASE8_REVIEW_REQUIRED' if bound else 'BLOCKED_MISSING_PROSPECTIVE_CONTRACT',
            'activation_authorized_by_this_operation': False}


def transient(metadata):
    return metadata.get('status') == 'SOURCE_UNAVAILABLE' or metadata.get('http_status') in (408, 429, 500, 502, 503, 504)


def retrieve(path, league, protocol):
    """Retry only transport/rate-limit errors; each attempt has its own raw archive."""
    for attempt in range(RETRIES):
        directory = p.run_path(path) / league / 'retrievals' / uuid.uuid4().hex
        key = p.source.credentials()['football_data_org']
        metadata = p.source.fetch(directory, 'matches',
            f'https://api.football-data.org/v4/competitions/{league}/matches',
            provider='football-data.org:v4', params={'season': protocol['season']},
            headers={'X-Auth-Token': key} if key else {})
        if metadata.get('status') == 'RETRIEVED':
            return metadata
        p.incident(path, league, 'SOURCE_RETRIEVAL_FAILED:' + str(metadata.get('http_status', metadata['status'])),
                   source_ref=p.identity(directory / 'matches.json'))
        if not transient(metadata) or attempt == RETRIES - 1:
            raise ValueError('SOURCE_UNAVAILABLE:authoritative fixtures')
        time.sleep((2, 5)[attempt])


def refreshed(path, league, protocol):
    """Retry qualification only when archived transport failures explain failure."""
    for attempt in range(RETRIES):
        start = len(p.journal(path, league))
        try:
            return p.refresh(path, league, protocol)
        except (ValueError, RuntimeError, OSError) as exc:
            if p.is_violation(exc):
                raise
            attempts = [e['payload']['source'] for e in p.journal(path, league)[start:]
                        if e['kind'] == 'incident' and e['payload']['code'] == 'SOURCE_REFRESH_STARTED']
            metadata = []
            for ref in attempts:
                for file in p.frozen.reference(ref).parent.rglob('*.json'):
                    value = p.read(file)
                    if isinstance(value, dict) and value.get('schema_version') == 'ffps.phase11_raw_source.v1':
                        metadata.append(value)
            failures = [m for m in metadata if m.get('status') != 'RETRIEVED']
            p.incident(path, league, 'SOURCE_REFRESH_ATTEMPT_FAILED:' + str(exc))
            if not failures or not all(transient(m) for m in failures) or attempt == RETRIES - 1:
                raise
            time.sleep((2, 5)[attempt])


def counts(context):
    return {'pairs': len(context[3]), 'settled': len(context[4]), 'journal_count': len(context[-1]),
            'journal_head': context[-1][-1]['record_sha256'] if context[-1] else None}


def process_league(path, directory, league):
    outcome = directory / league / 'outcome.json'
    if outcome.exists():
        value = p.read(outcome)
        p.frozen.sealed(value, 'record_sha256')
        return value
    with p.state_lock(p.run_path(path) / league / 'operation.lock'):
        context = p.load_cohort(path, league)
        protocol, controller, q, pairs, _, _, events = context
        claim = directory / league / 'claim.json'
        if claim.exists():
            before = p.read(claim)
            p.frozen.sealed(before, 'record_sha256')
        else:
            before = put(claim, p.seal({'league': league, 'as_of': p.now(), **counts(context)}))
        p.check(before['journal_count'] <= len(events) and
                (not before['journal_count'] or events[before['journal_count'] - 1]['record_sha256'] == before['journal_head']),
                'OPERATION_CLAIM_JOURNAL_MISMATCH')
        discovery, blocked, error = None, False, None
        try:
            if not any(e['kind'] == 'incident' and e['payload']['protocol_violation'] for e in events):
                metadata = retrieve(path, league, protocol)
                discovered = p.settle(path, league, context, metadata)
                existing = {pair['fixture_id'] for pair in pairs}
                eligible = sorted([e for e in discovered if e['status'] in ('TIMED', 'SCHEDULED') and
                    p.instant(protocol['start_at']) <= p.instant(e['kickoff_at']) < p.instant(protocol['end_at']) and
                    p.instant(p.now()) < p.instant(e['kickoff_at']) and e['fixture_id'] not in existing],
                    key=lambda e: (e['kickoff_at'], e['fixture_id']))
                discovery = {'source': metadata, 'eligible_fixture_ids': [e['fixture_id'] for e in eligible]}
                missed = {e['payload']['fixture_id'] for e in p.journal(path, league)
                          if e['kind'] == 'incident' and e['payload']['code'] == 'MISSED_PROSPECTIVE_CUTOFF'}
                for event in discovered:
                    if event['fixture_id'] not in existing | missed and p.instant(protocol['start_at']) <= p.instant(event['kickoff_at']) <= p.instant(p.now()):
                        p.incident(path, league, 'MISSED_PROSPECTIVE_CUTOFF', fixture_id=event['fixture_id'], source_ref=metadata)
                current = p.journal(path, league)
                violations = any(e['kind'] == 'incident' and e['payload']['protocol_violation'] for e in current)
                # A crash after atomic pair publication must never consume a second
                # earliest-fixture admission in the same logical invocation.
                admitted = any(e['kind'] == 'pair' for e in current[before['journal_count']:])
                if eligible and not violations and not admitted:
                    fresh = refreshed(path, league, protocol)
                    p.collect(path, league, p.load_cohort(path, league), fresh)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
            error, blocked = str(exc), True
            p.incident(path, league, error, violation=p.is_violation(exc))
        checkpoint = p.write_checkpoint(path, league, discovery=discovery, temporarily_blocked=blocked)
        return put(outcome, p.seal({'league': league, 'before': before, 'checkpoint': checkpoint,
            'newly_admitted': checkpoint['pairs'] - before['pairs'],
            'newly_settled': checkpoint['settled'] - before['settled'],
            'new_incidents': [e['payload'] for e in p.journal(path, league)[before['journal_count']:]
                              if e['kind'] == 'incident'],
            'authorization': authorization(controller), 'error': error}))


def reports(path):
    result = sorted(root(path).glob('report-[0-9]*'))
    p.check([d.name for d in result] == [f'report-{i:06d}' for i in range(1, len(result) + 1)],
            'OPERATION_REPORT_REMOVED_OR_REORDERED')
    return result


def run(path, key=None):
    path = p.run_path(path)
    key = key or p.instant(p.now()).strftime('%Y%m%dT%H')
    p.check(bool(re.fullmatch(r'[A-Za-z0-9_-]{1,80}', key)), 'INVALID_INVOCATION_KEY')
    # Contention returns an explicit retriable busy error, never a league violation.
    with p.state_lock(root(path) / 'runner.lock'):
        initialize(path)
        preserved = integrity(path)
        directories = reports(path)
        pending = [d for d in directories if not (d / 'report.json').exists()]
        p.check(len(pending) <= 1, 'MULTIPLE_UNFINISHED_OPERATIONS')
        matching = [d for d in directories if (d / 'intent.json').exists() and p.read(d / 'intent.json')['key'] == key]
        if matching and (matching[0] / 'report.json').exists():
            result = p.read(matching[0] / 'report.json')
            p.frozen.sealed(result, 'record_sha256')
            for ref in result['artifacts']:
                p.frozen.reference(ref)
            return {'duplicate': True, 'report': p.identity(matching[0] / 'report.json'),
                    'degraded': result.get('degraded', False)}
        directory = pending[0] if pending else root(path) / f'report-{len(directories) + 1:06d}'
        if not (directory / 'intent.json').exists():
            predecessors = [d for d in directories if d.name < directory.name]
            previous = predecessors[-1] / 'report.json' if predecessors else None
            put(directory / 'intent.json', p.seal({'key': key, 'created_at': p.now(),
                'previous_report': p.identity(previous) if previous else None,
                'configuration': p.identity(root(path) / 'configuration.json')}))
        outcomes = []
        for league in p.source.LEAGUES:
            print('Phase 12 operations: ' + league, flush=True)
            try:
                outcomes.append(process_league(path, directory, league))
            except Exception as exc:
                # Failures are auditable but never represented as successful replay.
                # Process death (BaseException) leaves intent pending for recovery.
                file = directory / league / 'outcome.json'
                outcomes.append(put(file, p.seal({'league': league, 'replay': 'FAILED',
                    'scientific_state': 'PROTOCOL_VIOLATION_BLOCKED', 'error': str(exc),
                    'promotion': 'NOT_AUTHORIZED'})))
        from src.evaluation.phase12_operations_evidence import seal_report
        return seal_report(path, directory, outcomes, preserved)

"""Final scientific publication over frozen Phase12/13 contracts. No activation API."""
from pathlib import Path
import os
import re
import uuid
from src.evaluation import phase13_authorization as a
from src.evaluation import phase12_evidence as e12
from src.ml.league_bundle_activation import LeagueBundleController
from src.ml.bundle_activation import canonical_bytes

RUN = a.p.ROOT / 'baselines/phase14/phase14-final-decision-20260915-v1'


def binding(league):
    a.p.check(league in a.p.source.LEAGUES, 'WRONG_LEAGUE')
    a.p.check(a.RUN.name == 'phase13-authorization-20260915-v2' and
              a.NEXT.name == 'phase13-authorization-20260915-v2', 'PHASE13_V2_REQUIRED')
    controller = LeagueBundleController(a.NEXT / league / 'deployment_specification.json', league=league, root=a.p.ROOT)
    contract, installation = a.verify_binding(controller)
    return controller, contract, installation


def authority_snapshot(league):
    """Detect non-cooperating writers too; normal writers share operation.lock."""
    directory = a.COHORT / league
    return {a.relative(path): a.p.source.sha256_file(path) for path in sorted(directory.rglob('*'))
            if path.is_file() and path.suffix not in ('.lock', '.tmp')}


def failure_state(message):
    upper = message.upper()
    if 'BUSY' in upper:
        return 'BLOCKED_OPERATION_BUSY'
    if 'REPLAY' in upper:
        return 'BLOCKED_REPLAY_FAILURE'
    if any(word in upper for word in ('CONFLICT', 'VIOLATION', 'BASELINE', 'PRODUCTION_BUNDLE_CHANGED')):
        return 'BLOCKED_PROTOCOL_VIOLATION'
    return 'BLOCKED_LINEAGE_INVALID'


def inspect_locked(league, finalized_at=None):
    at = finalized_at or a.p.now()
    report = {'league': league, 'as_of': at, 'state': 'BLOCKED_LINEAGE_INVALID',
        'blockers': [], 'reconstructed': False, 'activation_authorized': False}
    def block(code, detail):
        report['blockers'].append({'code': code, 'detail': detail})
    try:
        controller, contract, installed = binding(league)
        protocol = a.p.read(a.p.frozen.reference(contract['protocol']))
        report.update(protocol=contract['protocol'], installation=installed,
                      horizon_end=protocol['end_at'])
        if a.p.instant(at) > a.p.instant(a.p.now()):
            block('BLOCKED_FUTURE_FINALIZATION_TIME', 'finalized_at exceeds the real clock')
        if a.p.instant(at) < a.p.instant(protocol['end_at']) or a.p.instant(a.p.now()) < a.p.instant(protocol['end_at']):
            block('AWAITING_HORIZON', protocol['end_at'])
        before = authority_snapshot(league)
        context = a.p.load_cohort(a.COHORT, league)
        a.p.check(context[0] == protocol, 'PROTOCOL_RECONSTRUCTION_MISMATCH')
        pairs, settlements, events = context[3], context[4], context[6]
        from src.evaluation.phase14_verifier import verify_incident_sources
        verify_incident_sources(events)
        prefix_count = len(a.p.specification(a.COHORT)['leagues'][league]['prefix'])
        for path in sorted((a.COHORT / league / 'checkpoints').glob('*.json')):
            e12.verify_checkpoint(a.p.read(path), context, prefix_count)
        incidents = [event['payload'] for event in events if event['kind'] == 'incident']
        for incident in incidents:
            if incident['protocol_violation'] or a.p.is_violation(incident['code']):
                block('BLOCKED_PROTOCOL_VIOLATION', incident)
        evaluation = a.p.evaluate(protocol, pairs, settlements, at)
        if not evaluation['cohort_completeness']:
            block('AWAITING_SETTLEMENTS', evaluation['pending_fixture_ids'])
        if not evaluation['fixture_count']:
            block('BLOCKED_NO_SETTLED_OBSERVATIONS', 'frozen Phase7 rule requires observed outcomes')
        a.p.check(before == authority_snapshot(league), 'JOURNAL_MUTATION_DURING_RECONSTRUCTION')
        report.update(reconstructed=True, pairs=len(pairs), settled=len(settlements),
            pending=len(pairs)-len(settlements), incidents=incidents,
            ordered_event_hashes=[event['record_sha256'] for event in events],
            ordered_pair_hashes=[pair['pair_sha256'] for pair in pairs],
            ordered_settlement_hashes=[row['settlement_sha256'] for row in settlements],
            evaluation=evaluation, authority_snapshot_sha256=a.p.digest(before))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        block(failure_state(str(exc)), str(exc))
    severity = ('BLOCKED_OPERATION_BUSY', 'BLOCKED_PROTOCOL_VIOLATION', 'BLOCKED_REPLAY_FAILURE', 'BLOCKED_LINEAGE_INVALID',
                'BLOCKED_FUTURE_FINALIZATION_TIME', 'AWAITING_HORIZON', 'AWAITING_SETTLEMENTS',
                'BLOCKED_NO_SETTLED_OBSERVATIONS')
    report['state'] = next((code for code in severity if any(b['code'] == code for b in report['blockers'])),
                           'READY_TO_FINALIZE')
    return report


def status(league):
    a.p.check(league in a.p.source.LEAGUES, 'WRONG_LEAGUE')
    try:
        with a.p.state_lock(a.COHORT / league / 'operation.lock'):
            report = inspect_locked(league)
            destination = a.RUN / league / 'decisions/final'
            if destination.exists() and not report['blockers']:
                from src.evaluation.phase14_verifier import verify_locked
                report['publication'] = verify_locked(destination, league)
                report['state'] = 'FINALIZED'
            return report
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        return {'league': league, 'state': failure_state(str(exc)), 'reconstructed': False,
                'blockers': [{'code': failure_state(str(exc)), 'detail': str(exc)}],
                'activation_authorized': False}


def publish_directory(stage, destination):
    """Same-filesystem atomic directory rename, refusing an existing target."""
    if os.name == 'nt':
        os.rename(stage, destination)  # Windows rename never replaces an existing destination.
    else:
        import ctypes
        library = ctypes.CDLL(None, use_errno=True)
        rename = getattr(library, 'renameat2', None)
        a.p.check(rename is not None, 'ATOMIC_NO_REPLACE_RENAME_UNSUPPORTED')
        if rename(-100, os.fsencode(stage), -100, os.fsencode(destination), 1) != 0:
            raise OSError(ctypes.get_errno(), 'atomic immutable directory publication failed')
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def finalize(league, *, publication_id='final', finalized_at=None):
    a.p.check(league in a.p.source.LEAGUES, 'WRONG_LEAGUE')
    a.p.check(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', publication_id), 'INVALID_PUBLICATION_ID')
    try:
        with a.p.state_lock(a.COHORT / league / 'operation.lock'):
            destination = a.RUN / league / 'decisions' / publication_id
            if destination.exists() and finalized_at is None:
                finalized_at = a.p.read(destination / 'prospective_result.json')['finalized_at']
            report = inspect_locked(league, finalized_at)
            if report['blockers']:
                return {**report, 'published': False}
            controller, contract, installed = binding(league)
            before = authority_snapshot(league)
            a.p.check(a.p.digest(before) == report['authority_snapshot_sha256'], 'JOURNAL_MUTATION_BEFORE_PUBLICATION')
            material = a.decision_material(contract, installed, report['as_of'])
            a.p.check(material['evaluation'] == report['evaluation'], 'EVALUATION_RECONSTRUCTION_MISMATCH')
            a.p.check(before == authority_snapshot(league), 'JOURNAL_MUTATION_DURING_FINALIZATION')
            from src.evaluation.phase14_verifier import verify_locked
            if destination.exists():
                a.p.check((destination / 'prospective_result.json').read_bytes() == canonical_bytes(material),
                          'CONFLICTING_IMMUTABLE_PUBLICATION')
                return {**report, 'state': 'FINALIZED', 'published': False, 'idempotent': True,
                        'publication': verify_locked(destination, league)}
            # Staging is outside the Phase13-permitted decisions root, so it can never authorize deployment.
            stage = a.RUN / league / ('.phase14-stage-' + uuid.uuid4().hex)
            stage.mkdir(parents=True, exist_ok=False)
            a.p.persist_new(stage / 'prospective_result.json', material)
            reference = {**a.p.identity(stage / 'prospective_result.json'),
                         'path': a.relative(destination / 'prospective_result.json')}
            a.p.persist_new(stage / 'artifact_index.json', a.p.seal({'decision': reference}))
            a.p.check((stage / 'prospective_result.json').read_bytes() == canonical_bytes(material), 'STAGED_BYTES_MISMATCH')
            a.verify_binding(controller)
            a.p.check(before == authority_snapshot(league), 'JOURNAL_MUTATION_BEFORE_RENAME')
            destination.parent.mkdir(parents=True, exist_ok=True)
            publish_directory(stage, destination)
            return {**report, 'state': 'FINALIZED', 'published': True, 'idempotent': False,
                    'publication': verify_locked(destination, league)}
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        return {'league': league, 'state': failure_state(str(exc)), 'published': False,
                'publication_exists': (a.RUN / league / 'decisions' / publication_id).exists(),
                'blockers': [{'code': failure_state(str(exc)), 'detail': str(exc)}],
                'activation_authorized': False}

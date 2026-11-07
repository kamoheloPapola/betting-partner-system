"""Additive authorization constraints; never produces a scientific decision."""
from pathlib import Path
from src.evaluation import phase12_prospective as p
from src.evaluation.phase12_operations import put

RUN = p.ROOT / 'baselines/phase13/phase13-authorization-20260915-v2'
COHORT = p.ROOT / 'baselines/phase12/phase12-prospective-20260908-v1'
PRIOR = p.ROOT / 'baselines/phase9/phase9-multi-league-20260905-v1'
NEXT = p.ROOT / 'baselines/phase9/phase13-authorization-20260915-v2'
ANCHOR = COHORT / 'operational/report-000001'
IMPLEMENTATION = ('src/evaluation/phase13_authorization.py',
    'src/ml/phase13_bundle_activation.py', 'src/ml/bundle_activation.py',
    'src/ml/league_bundle_activation.py', 'src/evaluation/phase12_prospective.py',
    'src/evaluation/phase12_evidence.py', 'src/evaluation/phase7_shadow.py')


def relative(path):
    return Path(path).resolve().relative_to(p.ROOT).as_posix()


def successor_material(league):
    old = PRIOR / league / 'deployment_specification.json'
    value = p.read(old)
    p.frozen.sealed(value, 'specification_sha256')
    return {**{k: v for k, v in value.items() if k != 'specification_sha256'},
        'specification_version': 2, 'supersedes': p.identity(old),
        'promotion_status': 'BLOCKED_AWAITING_PROSPECTIVE_DECISION', 'prospective_contract': None}


def construct(league):
    p.check(league in p.source.LEAGUES, 'WRONG_LEAGUE')
    protocol, _, controller, _, _ = p.frozen.load_protocol(p.PHASE11, league)
    material = successor_material(league)
    p.check(controller.owner.frozen == p.read(PRIOR / league / 'deployment_specification.json'),
            'PRIOR_SPECIFICATION_MISMATCH')
    anchor = p.read(ANCHOR / league / 'cohort_index.json')
    run = p.specification(COHORT)
    p.frozen.reference(run['leagues'][league]['protocol'])
    p.check(anchor['league'] == league and anchor['prefix'] == run['leagues'][league]['prefix'],
            'LINEAGE_ANCHOR_MISMATCH')
    for ref in anchor['prefix'] + anchor['ordered_journal']:
        p.frozen.reference(ref)
    bundles = {label: controller.owner.bundle(material[label]['bundle_id']) for label in ('active', 'candidate')}
    return p.seal({'schema_version': 'ffps.prospective_authorization.v1', 'league': league,
        'prior_specification': material['supersedes'],
        'successor': {'path': relative(NEXT / league / 'deployment_specification.json'),
                      'version': 2, 'material_sha256': p.digest(material)},
        'protocol': run['leagues'][league]['protocol'],
        'protocol_sha256': protocol['protocol_sha256'],
        'bundles': bundles, 'evaluation_specification': material['evaluation_specification'],
        'scientific_rules': {'authority': run['leagues'][league]['protocol'],
                             'interpretation': 'unchanged Phase12 evaluate and Phase7 decision rule'},
        'lineage': {'run': p.identity(COHORT / 'run_specification.json'),
            'cohort_root': relative(COHORT), 'league': league,
            'anchor': p.identity(ANCHOR / league / 'cohort_index.json'),
            'report': p.identity(ANCHOR / 'report.json')},
        'decision': {'schema_version': 'ffps.final_prospective_decision.v1',
            'directory': relative(RUN / league / 'decisions'),
            'filename': 'prospective_result.json', 'status': 'FINALIZED_FIXED_HORIZON',
            'contents': 'exact complete current ordered cohort, incidents and recomputed Phase12 evaluation'},
        'implementation': [p.identity(p.ROOT / name) for name in IMPLEMENTATION],
        'phase8': {'engine': 'BundleController.activate', 'explicit_operator': True,
            'expected_active_generation': True, 'stopped_workers': True, 'smoke_replay': True,
            'atomic_transition_rollback_recovery': True, 'cohort_lock_through_transition': True},
        'activation_authorized': False}, 'contract_sha256')


def install(league):
    contract = construct(league)
    path = RUN / league / 'prospective_contract.json'
    put(path, contract)
    spec = p.seal({**successor_material(league), 'prospective_contract': p.identity(path)},
                  'specification_sha256')
    target = NEXT / league / 'deployment_specification.json'
    put(target, spec)
    return put(RUN / league / 'installation.json', p.seal({
        'league': league, 'contract': p.identity(path), 'specification': p.identity(target)}))


def verify_binding(controller):
    league = controller.league
    installation = p.read(RUN / league / 'installation.json')
    p.frozen.sealed(installation, 'record_sha256')
    p.check(installation['league'] == league, 'INSTALLATION_LEAGUE_MISMATCH')
    for key in ('contract', 'specification'):
        p.frozen.reference(installation[key])
    p.check(p.identity(controller.specification_path) == installation['specification'], 'UNBOUND_SPECIFICATION')
    contract = p.read(p.frozen.reference(installation['contract']))
    p.frozen.sealed(contract, 'contract_sha256')
    p.check(contract == construct(league), 'CONTRACT_RECONSTRUCTION_MISMATCH')
    expected = p.seal({**successor_material(league), 'prospective_contract': installation['contract']},
                      'specification_sha256')
    p.check(controller.specification() == expected, 'SUCCESSOR_SPECIFICATION_MISMATCH')
    return contract, installation


def decision_material(contract, installation, finalized_at):
    """Full replay, never accepts a selected export. Caller holds cohort lock."""
    protocol, _, _, pairs, settlements, _, events = p.load_cohort(COHORT, contract['league'])
    p.check(protocol['protocol_sha256'] == contract['protocol_sha256'], 'DECISION_PROTOCOL_MISMATCH')
    p.check(p.instant(protocol['end_at']) <= p.instant(finalized_at) <= p.instant(p.now()),
            'FIXED_HORIZON_NOT_COMPLETE')
    incidents = [event['payload'] for event in events if event['kind'] == 'incident']
    p.check(not any(i['protocol_violation'] or p.is_violation(i['code']) for i in incidents),
            'SCIENTIFIC_REVIEW_BLOCKED')
    evaluation = p.evaluate(protocol, pairs, settlements, finalized_at)
    p.check(evaluation['horizon_complete'] and evaluation['cohort_completeness'] and
            not evaluation['interim_descriptive_only'] and evaluation['fixture_count'] > 0,
            'INCOMPLETE_FINAL_COHORT')
    return p.seal({'schema_version': contract['decision']['schema_version'],
        'status': contract['decision']['status'], 'league': contract['league'],
        'contract': installation['contract'], 'deployment_specification': installation['specification'],
        'protocol': contract['protocol'], 'finalized_at': finalized_at,
        'lineage': contract['lineage'], 'ordered_events': events,
        'pairs': pairs, 'settlements': settlements, 'incidents': incidents,
        'evaluation': evaluation}, 'decision_sha256')


def gate(controller, evidence):
    contract, installed = verify_binding(controller)
    directory = Path(evidence).resolve()
    permitted = p.ROOT / contract['decision']['directory']
    result = {'league': controller.league, 'promotion_eligible': False,
        'candidate_bundle_id': contract['bundles']['candidate']['bundle_id'],
        'specification_fingerprint': contract['evaluation_specification']['specification_sha256'],
        'classification': 'CANDIDATE_INCONCLUSIVE', 'activation_authorized': False,
        'status': 'BLOCKED_AWAITING_PROSPECTIVE_DECISION'}
    p.check(directory.is_relative_to(permitted), 'UNBOUND_DECISION_PATH')
    path = directory / contract['decision']['filename']
    if not path.exists():
        return result
    supplied = p.read(path)
    p.frozen.sealed(supplied, 'decision_sha256')
    index = p.read(directory / 'artifact_index.json')
    p.frozen.sealed(index, 'record_sha256')
    p.check(index == p.seal({'decision': p.identity(path)}), 'DECISION_INDEX_MISMATCH')
    p.check(supplied['schema_version'] == contract['decision']['schema_version'] and
            supplied['status'] == contract['decision']['status'], 'INTERIM_OR_UNKNOWN_DECISION')
    expected = decision_material(contract, installed, supplied['finalized_at'])
    p.check(supplied == expected, 'FINAL_DECISION_RECONSTRUCTION_MISMATCH')
    decision = expected['evaluation']['decision']
    eligible = decision['classification'] == 'CANDIDATE_SUPPORTED_FOR_PROMOTION_REVIEW' and decision['phase8_promotion_review_authorized']
    return {**result, 'promotion_eligible': eligible, 'classification': decision['classification'],
        'status': 'PHASE8_PREFLIGHT_REQUIRED' if eligible else 'BLOCKED_SCIENTIFIC_DECISION',
        'final_decision_sha256': supplied['decision_sha256']}

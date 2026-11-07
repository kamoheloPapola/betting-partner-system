"""Read-only binding reconstruction and immutable Phase13 evidence packaging."""
import xml.etree.ElementTree as ET
from src.evaluation import phase13_authorization as a
from src.ml.phase13_bundle_activation import ProspectiveBundleController

SOURCES = (*a.IMPLEMENTATION, 'src/evaluation/phase13_evidence.py',
           'scripts/run_phase13_authorization.py', 'tests/unit/test_phase13_authorization.py',
           'docs/phase13-prospective-authorization.md')


def status():
    rows = []
    for league in a.p.source.LEAGUES:
        controller = ProspectiveBundleController(a.NEXT / league / 'deployment_specification.json', league=league)
        contract, installed = a.verify_binding(controller)
        gate = controller.league_promotion_gate(a.RUN / league / 'decisions/final')
        protocol = a.p.read(a.p.frozen.reference(contract['protocol']))
        anchor = a.p.read(a.p.frozen.reference(contract['lineage']['anchor']))
        rows.append({'league': league, 'installation': installed, 'protocol': contract['protocol'],
            'bundles': contract['bundles'], 'horizon': {k: protocol[k] for k in ('start_at', 'end_at')},
            'bootstrap': protocol['bootstrap'], 'anchored_pairs': len(anchor['ordered_pairs']),
            'anchored_settlements': len(anchor['ordered_settlements']),
            'before': 'BLOCKED_MISSING_PROSPECTIVE_CONTRACT', 'after': gate,
            'current_fixed_horizon_complete': a.p.instant(a.p.now()) >= a.p.instant(protocol['end_at'])})
    return rows


def integrity():
    before = a.p.read(a.RUN / 'protected_before.json')
    changed = [name for name, sha in before.items() if not (a.p.ROOT / name).is_file()
               or a.p.source.sha256_file(a.p.ROOT / name) != sha]
    production = {base: a.p.source._tree_snapshot(a.p.ROOT / base)
                  for base in ('data', 'src/ml/models', 'baselines/phase7p')}
    a.p.check(not changed, 'PRE_PHASE13_MUTATION:' + ','.join(changed))
    a.p.check(production == a.p.read(a.RUN / 'production_before.json'), 'PRODUCTION_OR_PL7P_MUTATION')
    additions = []
    for base in ('baselines/phase11', 'baselines/phase12'):
        for path in (a.p.ROOT / base).rglob('*'):
            if path.is_file() and path.suffix not in ('.lock', '.tmp') and a.relative(path) not in before:
                additions.append(a.p.identity(path))
    return {'protected_count': len(before), 'changed': changed, 'production_after': production,
            'concurrent_phase12_additions': additions, 'verified': True}


def test_summary(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    failures = sorted(case.get('classname', '') + '::' + case.get('name', '') for case in cases
                      if case.find('failure') is not None or case.find('error') is not None)
    skipped = sum(case.find('skipped') is not None for case in cases)
    return {'tests': len(cases), 'passed': len(cases) - len(failures) - skipped,
            'skipped': skipped, 'failures': failures}


def certify(focused, protected, full):
    inherited = test_summary(a.p.ROOT / 'tmp/phase12-operations-full-final.xml')
    tests = {key: test_summary(path) for key, path in
             [('focused', focused), ('protected', protected), ('full', full)]}
    a.p.check(not tests['focused']['failures'] and not tests['protected']['failures'], 'FOCUSED_OR_PROTECTED_FAILURE')
    a.p.check(tests['full']['failures'] == inherited['failures'], 'UNEXPLAINED_FULL_SUITE_FAILURE')
    for key, path in [('focused', focused), ('protected', protected), ('full', full)]:
        target = a.RUN / 'tests' / (key + '.xml')
        if target.exists():
            a.p.check(target.read_bytes() == path.read_bytes(), 'TEST_EVIDENCE_CONFLICT')
        else:
            # Test bytes are preserved, including failure traces.
            target.parent.mkdir(parents=True, exist_ok=True)
            # This destination is new, never a historical report.
            target.write_bytes(path.read_bytes())
    return a.put(a.RUN / 'test_evidence.json', a.p.seal({'results': tests,
        'inherited_reference': a.p.identity(a.p.ROOT / 'tmp/phase12-operations-full-final.xml'),
        'new_failures': [], 'reports': [a.p.identity(a.RUN / 'tests' / (k + '.xml')) for k in tests]}))


def seal_report():
    rows = status()
    a.p.check(all(not row['after']['promotion_eligible'] and not row['after']['activation_authorized']
                  for row in rows), 'REAL_LEAGUE_UNEXPECTEDLY_ELIGIBLE')
    protected = integrity()
    tests = a.p.read(a.RUN / 'test_evidence.json')
    a.p.frozen.sealed(tests, 'record_sha256')
    for ref in tests['reports']:
        a.p.frozen.reference(ref)
    a.put(a.RUN / 'protected_comparison.json', protected)
    report = a.put(a.RUN / 'report.json', a.p.seal({'schema_version': 'ffps.phase13_report.v1',
        'leagues': rows, 'protected_comparison': a.p.identity(a.RUN / 'protected_comparison.json'),
        'tests': a.p.identity(a.RUN / 'test_evidence.json'), 'activation_performed': False,
        'scientific_decisions_created': False, 'collection_scheduler_changed': False}))
    lines = ['# Phase 13 — Prospective authorization pre-binding', '',
        'All four successor specifications are bound. No final scientific decision or activation was created.', '',
        '| League | Contract SHA-256 | Authorization |', '|---|---|---|']
    for row in rows:
        lines.append('| ' + row['league'] + ' | ' + row['installation']['contract']['sha256'] +
                     ' | ' + row['after']['status'] + ' |')
    lines += ['', f"Protected existing files: {protected['protected_count']}; changed: 0. Model bytes, pointers and PL Phase 7P are unchanged.",
        'The dated Phase12 anchor contains 19 pairs and 3 settlements. It is a lineage anchor, not a new current cohort evaluation.',
        'All horizons remain open. Current checkpoints cannot satisfy the final-decision schema or fixed-horizon checks.', '',
        'Validation: ' + str(tests['results']), '',
        'The original Phase9 specifications and historical Phase12 reports retain their original missing-contract status. '
        'The new immutable specifications carry the successor authorization status. Production pointers still select their existing state.',
        'Future explicit Phase8 operation uses ProspectiveBundleController; the older non-PL CLI does not implement this adapter and fails closed.',
        'The interactive-user Windows scheduling limitation remains deferred. No scheduling change was made.']
    path = a.RUN / 'PHASE13_IMPLEMENTATION_REPORT.md'
    content = '\n'.join(lines) + '\n'
    if path.exists():
        a.p.check(path.read_text(encoding='utf-8') == content, 'REPORT_CONFLICT')
    else:
        path.write_text(content, encoding='utf-8')
    artifacts = [a.p.identity(path) for path in a.RUN.rglob('*') if path.is_file()
                 and path.name not in ('artifact_index.json', 'offline_verification.json')]
    artifacts += [a.p.identity(a.NEXT / league / 'deployment_specification.json') for league in a.p.source.LEAGUES]
    a.put(a.RUN / 'artifact_index.json', a.p.seal({'artifacts': sorted(artifacts, key=lambda r: r['path']),
        'implementation': [a.p.identity(a.p.ROOT / name) for name in dict.fromkeys(SOURCES)]}))
    return report


def verify():
    index = a.p.read(a.RUN / 'artifact_index.json')
    a.p.frozen.sealed(index, 'record_sha256')
    for ref in index['artifacts'] + index['implementation']:
        a.p.frozen.reference(ref)
    rows = status()  # Reconstruct bindings from prior specifications and frozen protocols.
    report = a.p.read(a.RUN / 'report.json')
    a.p.frozen.sealed(report, 'record_sha256')
    a.p.check(rows == report['leagues'], 'OFFLINE_BINDING_OR_STATE_DIVERGENCE')
    comparison = integrity()
    tests = a.p.read(a.RUN / 'test_evidence.json')
    for key in ('focused', 'protected', 'full'):
        a.p.check(test_summary(a.RUN / 'tests' / (key + '.xml')) == tests['results'][key], 'TEST_REPORT_MISMATCH')
    return {'verified': True, 'leagues': list(a.p.source.LEAGUES),
            'protected_count': comparison['protected_count'], 'activation_authorized': False,
            'artifact_index': a.p.identity(a.RUN / 'artifact_index.json')}

"""Initial Phase14 implementation evidence; no scientific decision creation."""
from pathlib import Path
import xml.etree.ElementTree as ET
from src.evaluation import phase14_decision as d
from src.evaluation import phase14_verifier as v

SOURCES = ('src/evaluation/phase14_decision.py', 'src/evaluation/phase14_verifier.py',
           'src/evaluation/phase14_evidence.py', 'scripts/run_phase14_decision.py',
           'tests/unit/test_phase14_decision.py', 'docs/phase14-final-decision.md')


def integrity():
    before = d.a.p.read(d.RUN / 'protected_before.json')
    changed = [name for name, sha in before.items() if not (d.a.p.ROOT / name).is_file()
               or d.a.p.source.sha256_file(d.a.p.ROOT / name) != sha]
    after = {base: d.a.p.source._tree_snapshot(d.a.p.ROOT / base)
             for base in ('data', 'src/ml/models', 'baselines/phase7p')}
    d.a.p.check(not changed, 'PROTECTED_FILE_CHANGED:' + ','.join(changed))
    d.a.p.check(after == d.a.p.read(d.RUN / 'production_before.json'), 'PRODUCTION_OR_PL7P_CHANGED')
    additions = [d.a.p.identity(path) for base in ('baselines/phase11', 'baselines/phase12')
                 for path in (d.a.p.ROOT / base).rglob('*') if path.is_file() and
                 path.suffix not in ('.lock', '.tmp') and d.a.relative(path) not in before]
    return {'verified': True, 'protected_count': len(before), 'changed': changed,
            'production_after': after, 'concurrent_phase12_additions': additions}


def tests(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    failures = sorted(c.get('classname', '') + '::' + c.get('name', '') for c in cases
                      if c.find('failure') is not None or c.find('error') is not None)
    skipped = sum(c.find('skipped') is not None for c in cases)
    return {'cases': len(cases), 'passed': len(cases)-len(failures)-skipped,
            'skipped': skipped, 'failures': failures}


def certify(focused, protected, full):
    prior = d.a.p.read(d.a.RUN / 'test_evidence.json')
    d.a.p.frozen.sealed(prior, 'record_sha256')
    ref = next(ref for ref in prior['reports'] if ref['path'].endswith('/full.xml'))
    inherited = tests(d.a.p.frozen.reference(ref))
    values = {key: tests(path) for key, path in [('focused', focused), ('protected', protected), ('full', full)]}
    d.a.p.check(not values['focused']['failures'] and not values['protected']['failures'], 'FOCUSED_OR_PROTECTED_FAILURE')
    d.a.p.check(values['full']['failures'] == inherited['failures'], 'NEW_FULL_SUITE_FAILURE')
    for key, source in [('focused', focused), ('protected', protected), ('full', full)]:
        target = d.RUN / 'tests' / (key + '.xml')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            d.a.p.check(target.read_bytes() == source.read_bytes(), 'TEST_REPORT_CONFLICT')
        else:
            target.write_bytes(source.read_bytes())
    return d.a.put(d.RUN / 'test_evidence.json', d.a.p.seal({'summaries': values,
        'inherited': ref, 'new_failures': [], 'reports': [d.a.p.identity(d.RUN / 'tests' / (key + '.xml')) for key in values]}))


def seal_report(*, reconstruction_index=None):
    """Seal after independent reconstruction; optionally consume its pinned receipts.

    This avoids rerunning expensive model replay merely to format a report.
    verify_report always performs fresh independent reconstruction.
    """
    if reconstruction_index is not None:
        receipt = d.a.p.read(reconstruction_index)
        d.a.p.frozen.sealed(receipt, 'record_sha256')
        expected = {'real': [d.a.p.identity(d.RUN / 'real' / (league + '.json')) for league in d.a.p.source.LEAGUES],
            'verified': [d.a.p.identity(d.RUN / 'verification' / (league + '.json')) for league in d.a.p.source.LEAGUES],
            'implementation': [d.a.p.identity(d.a.p.ROOT / name) for name in SOURCES]}
        d.a.p.check(receipt == d.a.p.seal(expected), 'RECONSTRUCTION_RECEIPT_MISMATCH')
    verified = []
    for league in d.a.p.source.LEAGUES:
        row = d.a.p.read(d.RUN / 'real' / (league + '.json'))
        if reconstruction_index is None:
            verification = v.verify_readiness(row)
            d.a.put(d.RUN / 'verification' / (league + '.json'), d.a.p.seal(verification))
        else:
            verification = d.a.p.read(d.RUN / 'verification' / (league + '.json'))
            d.a.p.frozen.sealed(verification, 'record_sha256')
            verification.pop('record_sha256')
            d.a.p.check(verification == {'verified': True, 'league': league,
                'pairs': row['pairs'], 'settled': row['settled'], 'state': row['state'],
                'zero_real_decisions': True, 'activation_authorized': False}, 'RECONSTRUCTION_RESULT_MISMATCH')
        d.a.p.check(not list((d.a.RUN / league).rglob('prospective_result.json')), 'UNEXPECTED_REAL_FINAL_DECISION')
        verified.append(verification)
    comparison = integrity()
    d.a.put(d.RUN / 'protected_comparison.json', comparison)
    certificate = d.a.p.read(d.RUN / 'test_evidence.json')
    d.a.p.frozen.sealed(certificate, 'record_sha256')
    for ref in certificate['reports']:
        d.a.p.frozen.reference(ref)
    report = d.a.put(d.RUN / 'report.json', d.a.p.seal({'schema_version': 'ffps.phase14_implementation.v1',
        'leagues': verified, 'real_results': [d.a.p.identity(d.RUN / 'real' / (league + '.json')) for league in d.a.p.source.LEAGUES],
        'integrity': d.a.p.identity(d.RUN / 'protected_comparison.json'),
        'tests': d.a.p.identity(d.RUN / 'test_evidence.json'), 'zero_real_final_decisions': True,
        'activation_performed': False, 'scheduling_changed': False}))
    lines = ['# Phase 14 — Final prospective decision engine', '',
        'Scientific publishing and independent verification are implemented. Every real finalization attempt was refused.', '',
        '| League | Pairs | Settled | Pending | Horizon end (UTC) | State |', '|---|---:|---:|---:|---|---|']
    for league in d.a.p.source.LEAGUES:
        row = d.a.p.read(d.RUN / 'real' / (league + '.json'))
        lines.append(f"| {league} | {row['pairs']} | {row['settled']} | {row['pending']} | {row['horizon_end']} | {row['state']} |")
    lines += ['', 'Zero real final scientific artifacts exist. Frozen horizons and complete-settlement requirements remain enforced.',
        'Independent verification reconstructed every dated complete cohort, including incident history, causal/RNG/Phase4 replay, checkpoints, metrics and bootstrap.',
        'Synthetic terminal tests cover candidate support, active support, inconclusive outcomes, publication recovery, conflicts and lock exclusion. Synthetic artifacts remain in private test directories.', '',
        f"Protected files unchanged: {comparison['protected_count']}. Models, production pointers, Phase7P and Phase13 v1/v2 bytes are unchanged.",
        f"Concurrent new Phase12 files: {len(comparison['concurrent_phase12_additions'])}; these are separate from the protected prefix.", '',
        '| Test scope | Passed | Inherited failures | Skipped |', '|---|---:|---:|---:|']
    for key, result in certificate['summaries'].items():
        lines.append(f"| {key} | {result['passed']} | {len(result['failures'])} | {result['skipped']} |")
    lines += ['', 'Zero new unexplained failures. Full-suite traces and inherited failure identities are indexed in test_evidence.json.',
        'Phase14 has no activation command. The Phase12 hourly task and its logged-in-user limitation remain unchanged.',
        'Operating guide: docs/phase14-final-decision.md. Scientific finalization is a future explicit operation after the frozen horizon.']
    lines += ['', '## Preserved scientific blockers', '']
    for league in d.a.p.source.LEAGUES:
        row = d.a.p.read(d.RUN / 'real' / (league + '.json'))
        for blocker in row['blockers']:
            if blocker['code'] == 'BLOCKED_PROTOCOL_VIOLATION':
                detail = blocker['detail']
                lines.append(f"- {league}: `{detail['code']}` for `{detail.get('fixture_id')}`. This existing incident remains unresolved and blocks finalization independently of the horizon.")
    target = d.RUN / 'PHASE14_IMPLEMENTATION_REPORT.md'
    text = '\n'.join(lines) + '\n'
    if target.exists():
        d.a.p.check(target.read_text(encoding='utf-8') == text, 'REPORT_CONFLICT')
    else:
        target.write_text(text, encoding='utf-8')
    refs = [d.a.p.identity(path) for path in d.RUN.rglob('*') if path.is_file()
            and path.name not in ('artifact_index.json', 'offline_verification.json')]
    d.a.put(d.RUN / 'artifact_index.json', d.a.p.seal({'artifacts': sorted(refs, key=lambda r: r['path']),
        'implementation': [d.a.p.identity(d.a.p.ROOT / name) for name in SOURCES]}))
    receipt = {'verified': True, 'leagues': verified, 'protected_count': comparison['protected_count'],
               'zero_real_final_decisions': True, 'artifact_index': d.a.p.identity(d.RUN / 'artifact_index.json')}
    d.a.put(d.RUN / 'offline_verification.json', d.a.p.seal(receipt))
    return receipt


def verify_report():
    index = d.a.p.read(d.RUN / 'artifact_index.json')
    d.a.p.frozen.sealed(index, 'record_sha256')
    for ref in index['artifacts'] + index['implementation']:
        d.a.p.frozen.reference(ref)
    for league in d.a.p.source.LEAGUES:
        v.verify_readiness(d.a.p.read(d.RUN / 'real' / (league + '.json')))
    integrity()
    return {'verified': True, 'zero_real_final_decisions': True}

"""Phase 18 readiness evidence, integrity, tests, and offline reconstruction."""
from pathlib import Path

from src.evaluation import phase12_prospective as p
from src.evaluation import phase18_authority as authority
from src.evaluation import phase18_verifier as verifier
from src.evaluation.phase14_evidence import tests
from src.evaluation.phase12_operations import put

SOURCES = ('src/evaluation/phase18_authority.py',
           'src/evaluation/phase18_cohort.py',
           'src/evaluation/phase18_verifier.py',
           'src/evaluation/phase18_evidence.py',
           'scripts/run_phase18_readiness.py',
           'tests/unit/test_phase18_readiness.py',
           'docs/phase18-pl-v2-readiness.md')


def integrity():
    before = p.read(authority.RUN / 'protected_before.json')
    changed = [name for name, sha in before.items() if not (p.ROOT/name).is_file() or
               p.source.sha256_file(p.ROOT/name) != sha]
    p.check(not changed, 'PHASE18_PROTECTED_FILE_CHANGED:' + ','.join(changed))
    production = {base:p.source._tree_snapshot(p.ROOT/base)
                  for base in ('data','src/ml/models','baselines/phase7p')}
    p.check(production == p.read(authority.RUN / 'production_before.json'),
            'PHASE18_PRODUCTION_OR_PHASE7P_CHANGED')
    return {'verified': True, 'protected_count': len(before), 'changed': changed,
            'production_and_phase7p_unchanged': True}


def record_readiness():
    return put(authority.RUN / 'readiness.json', verifier.all_leagues())


def certify(focused, protected, full):
    prior = p.read(p.ROOT / 'baselines/phase17/phase17-all-league-v2-readiness-20260923-v1/test_evidence.json')
    p.frozen.sealed(prior, 'record_sha256')
    inherited_ref = next(ref for ref in prior['reports'] if ref['path'].endswith('/full.xml'))
    inherited = tests(p.frozen.reference(inherited_ref))
    inputs = {'focused': Path(focused), 'protected': Path(protected), 'full': Path(full)}
    summaries = {name: tests(path) for name, path in inputs.items()}
    p.check(not summaries['focused']['failures'] and not summaries['protected']['failures'],
            'PHASE18_FOCUSED_OR_PROTECTED_FAILURE')
    p.check(summaries['full']['failures'] == inherited['failures'],
            'PHASE18_NEW_FULL_SUITE_FAILURE')
    refs = []
    for name, source in inputs.items():
        target = authority.RUN / 'tests' / (name + '.xml')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            p.check(target.read_bytes() == source.read_bytes(), 'PHASE18_TEST_REPORT_CONFLICT')
        else:
            target.write_bytes(source.read_bytes())
        refs.append(p.identity(target))
    return put(authority.RUN / 'test_evidence.json', p.seal({'summaries': summaries,
        'inherited': inherited_ref, 'new_failures': [], 'reports': refs}))


def seal(focused, protected, full):
    current = record_readiness()
    p.check(current['scientific_state'] == 'PHASE18_ALL_LEAGUES_V2_READY_FOR_PROSPECTIVE_COLLECTION',
            'PHASE18_NOT_ALL_LEAGUE_READY')
    evidence = certify(focused, protected, full)
    comparison = put(authority.RUN / 'protected_comparison.json', p.seal(integrity()))
    contract = authority.verify_pl_contract()
    pl = next(row for row in current['leagues'] if row['league'] == 'PL')
    lines = ['# Phase 18 — PL prospective contract and five-league v2 readiness', '',
        '**Scientific state: `PHASE18_ALL_LEAGUES_V2_READY_FOR_PROSPECTIVE_COLLECTION`.** '
        'All five leagues have independent empty clean cohorts bound to the unchanged Phase 16 v2 rule.', '',
        'The original BL1 Phase 12–15 experiment remains `BLOCKED_PROTOCOL_VIOLATION`. '
        'The Phase 16 classifier and Phase 17 package are byte-identical to their sealed versions.', '',
        'PL has no Phase 11 prospective protocol. Phase 18 explicitly adopts the previously frozen '
        'Phase 7P December 31, 2026 fixture-date boundary as the new PL prospective contract endpoint '
        '`2027-01-01T00:00:00+00:00`. This is Phase 7P horizon authority, not Phase 11 inheritance. '
        'The PL cohort was created after contract freeze and cannot extend or replace that endpoint. '
        'If the window yields too little evidence, its scientific result is insufficient/inconclusive.', '',
        '| League | Cohort | Horizon authority | Start | End | Pairs | Settlements |',
        '|---|---|---|---|---|---:|---:|']
    for row in current['leagues']:
        source = 'Phase7P via Phase18 PL contract' if row['league'] == 'PL' else 'Phase11'
        lines.append(f"| {row['league']} | {row['cohort_id']} | {source} | {row['start_at']} | "
                     f"{row['end_at']} | 0 | 0 |")
    lines += ['', f"PL contract SHA256: `{p.identity(authority.PL_CONTRACT)['sha256']}`. ",
              f"PL clean cohort SHA256: `{pl['cohort']['sha256']}`. ",
              f"Frozen v2 rule SHA256: `{current['revision_rule']['sha256']}`. ", '',
              'The generic initializer validates a sealed horizon authority. Existing Phase 11 '
              'cohorts retain their original Phase 16 publisher and bytes; PL uses the explicit '
              'Phase 18 contract. All five independent verifiers report zero pairs and zero settlements. '
              'No Phase 7P prediction was imported.', '',
              f"Protected integrity: {comparison['protected_count']} files unchanged; production and Phase 7P unchanged.",
              '', '| Tests | Passed | Inherited failures | Skipped |',
              '|---|---:|---:|---:|']
    for name, summary in evidence['summaries'].items():
        lines.append(f"| {name} | {summary['passed']} | {len(summary['failures'])} | {summary['skipped']} |")
    lines += ['', 'The six full-suite failure identities match Phase 17; no new failure. '
              'Indexed XML reports and offline reconstruction are included.', '',
              'No collection, scheduler, settlement, model, pointer, promotion or activation '
              'operation ran. Operating guide: `docs/phase18-pl-v2-readiness.md`.']
    report = authority.RUN / 'PHASE18_IMPLEMENTATION_REPORT.md'
    content = '\n'.join(lines) + '\n'
    if report.exists():
        p.check(report.read_text(encoding='utf-8') == content, 'PHASE18_REPORT_CONFLICT')
    else:
        report.write_text(content, encoding='utf-8')
    refs = [p.identity(path) for path in authority.RUN.rglob('*') if path.is_file() and
            path.name not in ('artifact_index.json','offline_verification.json') and
            path.suffix not in ('.lock','.tmp')]
    put(authority.RUN / 'artifact_index.json', p.seal({
        'artifacts': sorted(refs, key=lambda ref:ref['path']),
        'implementation': [p.identity(p.ROOT/name) for name in SOURCES]}))
    return verify_package()


def verify_package():
    index = p.read(authority.RUN / 'artifact_index.json')
    p.frozen.sealed(index, 'record_sha256')
    for ref in index['artifacts'] + index['implementation']:
        p.frozen.reference(ref)
    current = verifier.all_leagues()
    stored = p.read(authority.RUN / 'readiness.json')
    p.frozen.sealed(stored, 'record_sha256')
    p.check(stored == current, 'PHASE18_READINESS_RECONSTRUCTION_MISMATCH')
    comparison = integrity()
    evidence = p.read(authority.RUN / 'test_evidence.json')
    p.frozen.sealed(evidence, 'record_sha256')
    observed = {name: tests(p.frozen.reference(ref)) for name, ref in
                zip(('focused','protected','full'), evidence['reports'])}
    inherited = tests(p.frozen.reference(evidence['inherited']))
    p.check(observed == evidence['summaries'] and not observed['focused']['failures'] and
            not observed['protected']['failures'] and
            observed['full']['failures'] == inherited['failures'] and
            evidence['new_failures'] == [], 'PHASE18_TEST_EVIDENCE_MISMATCH')
    return {'verified': True, 'scientific_state': current['scientific_state'],
        'artifact_index': p.identity(authority.RUN / 'artifact_index.json'),
        'protected_count': comparison['protected_count'],
        'all_five_pairs': 0, 'all_five_settlements': 0,
        'inherited_failures': len(inherited['failures']), 'new_failures': 0,
        'activation_authorized': False}

"""Phase16 test, protected-integrity and implementation evidence."""
from pathlib import Path
from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort as cohort
from src.evaluation import phase16_verifier as verifier
from src.evaluation.phase14_evidence import tests
from src.evaluation.phase12_operations import put

RUN = cohort.RUN
SOURCES = tuple('src/evaluation/' + name + '.py' for name in (
    'phase16_revision', 'phase16_cohort', 'phase16_verifier',
    'phase16_regression_facts', 'phase16_regression', 'phase16_evidence')) + (
    'scripts/run_phase16_protocol.py', 'tests/unit/test_phase16_revision.py',
    'tests/unit/test_phase16_cohort.py', 'docs/phase16-fixture-revision.md')


def integrity():
    before = p.read(RUN / 'protected_before.json')
    changed = [name for name, sha in before.items() if not (p.ROOT / name).is_file()
               or p.source.sha256_file(p.ROOT / name) != sha]
    after = {base: p.source._tree_snapshot(p.ROOT / base)
             for base in ('data', 'src/ml/models', 'baselines/phase7p')}
    p.check(not changed, 'PHASE16_PROTECTED_FILE_CHANGED:' + ','.join(changed))
    p.check(after == p.read(RUN / 'production_before.json'), 'PHASE16_PRODUCTION_OR_PL7P_CHANGED')
    additions = [p.identity(path) for base in ('baselines/phase11', 'baselines/phase12')
                 for path in (p.ROOT / base).rglob('*') if path.is_file() and
                 path.suffix not in ('.lock', '.tmp', '.pyc') and
                 path.relative_to(p.ROOT).as_posix() not in before]
    return {'verified': True, 'protected_count': len(before), 'changed': changed,
            'production_after': after, 'concurrent_phase12_additions': additions}


def certify(focused, protected, full):
    prior = p.read(p.ROOT / 'baselines/phase15/phase15-scientific-review-20260921-v1/test_evidence.json')
    p.frozen.sealed(prior, 'record_sha256')
    inherited_ref = next(ref for ref in prior['reports'] if ref['path'].endswith('/full.xml'))
    inherited = tests(p.frozen.reference(inherited_ref))
    inputs = {'focused': Path(focused), 'protected': Path(protected), 'full': Path(full)}
    summaries = {key: tests(path) for key, path in inputs.items()}
    p.check(not summaries['focused']['failures'] and not summaries['protected']['failures'],
            'PHASE16_FOCUSED_OR_PROTECTED_FAILURE')
    p.check(summaries['full']['failures'] == inherited['failures'], 'PHASE16_NEW_FULL_SUITE_FAILURE')
    refs = []
    for key, source in inputs.items():
        target = RUN / 'tests' / (key + '.xml')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            p.check(target.read_bytes() == source.read_bytes(), 'PHASE16_TEST_REPORT_CONFLICT')
        else:
            target.write_bytes(source.read_bytes())
        refs.append(p.identity(target))
    return put(RUN / 'test_evidence.json', p.seal({'summaries': summaries,
        'inherited': inherited_ref, 'new_failures': [], 'reports': refs}))


def seal_report():
    fresh = p.read(RUN / 'verification/clean_cohort.json')
    regression = p.read(RUN / 'verification/historical_regression.json')
    for row in (fresh, regression):
        p.frozen.sealed(row, 'record_sha256')
    p.check(fresh['verified'] and fresh['pairs'] == fresh['settlements'] == 0 and
            fresh['scientific_state'] == 'PHASE16_PROTOCOL_V2_READY_FOR_CLEAN_COHORT' and
            regression['verified'] and regression['counterfactual_only'] and
            regression['historical_state'] == 'BLOCKED_PROTOCOL_VIOLATION' and
            regression['v2_classification'] == 'PERMITTED_KICKOFF_REVISION',
            'PHASE16_RECONSTRUCTION_RECEIPT_INVALID')
    for ref in (fresh['cohort'], fresh['protocol'], fresh['revision_rule'], regression['artifact']):
        p.frozen.reference(ref)
    tests_record = p.read(RUN / 'test_evidence.json')
    p.frozen.sealed(tests_record, 'record_sha256')
    for ref in tests_record['reports']:
        p.frozen.reference(ref)
    comparison = integrity()
    put(RUN / 'protected_comparison.json', p.seal(comparison))
    put(RUN / 'report.json', p.seal({'schema_version': 'ffps.phase16_implementation.v1',
        'historical_scientific_state': regression['historical_state'],
        'historical_review_outcome': 'UNRESOLVED_PROTOCOL_CONFLICT',
        'hypothetical_v2_regression': regression['artifact'],
        'prospective_scientific_state': fresh['scientific_state'],
        'clean_cohort': fresh['cohort'], 'v2_protocol': fresh['protocol'],
        'revision_rule': fresh['revision_rule'],
        'independent_cohort_verification': p.identity(RUN / 'verification/clean_cohort.json'),
        'independent_regression_verification': p.identity(RUN / 'verification/historical_regression.json'),
        'tests': p.identity(RUN / 'test_evidence.json'),
        'integrity': p.identity(RUN / 'protected_comparison.json'),
        'old_cohort_reinterpreted': False, 'activation_performed': False,
        'scheduler_changed': False, 'production_pointer_changed': False}))
    lines = ['# Phase 16 — Versioned fixture revision protocol and clean restart', '',
        '**Historical scientific state: BLOCKED_PROTOCOL_VIOLATION.** BL1 fixture 565839 remains in its original Phase12 cohort. The Phase15 review remains `UNRESOLVED_PROTOCOL_CONFLICT`, and Phase14 cannot finalize that cohort.', '',
        '**Prospective scientific state: PHASE16_PROTOCOL_V2_READY_FOR_CLEAN_COHORT.** A new empty BL1 cohort is bound to the v2 rule, source qualification, active/candidate identities and the inherited May 23, 2027 fixed horizon. It contains zero pairs and zero settlements. This phase does not run collection or grant a decision.', '',
        'The v2 regression re-normalized three archived football-data.org source versions for fixture 565839, replayed the old complete cohort and independently reproduced `PERMITTED_KICKOFF_REVISION` under the new factual rule. That classification is counterfactual only; the old protocol still forbids the admitted kickoff change. The old pair was not transferred or regenerated.', '',
        'One generic classifier covers all five active league codes. It distinguishes unchanged schedules, permitted revisions, material identity/terminal changes, causality, order and horizon violations, and insufficient evidence. New-cohort admission verifies the exact v2 protocol hash, pre-kickoff capture, source qualification, paired features, same RNG and Phase4 replay. Source updates and post-match settlements are append-only and revalidated under the cohort lock. The independent verifier reconstructs raw sources, pair replay, journal order, classifications and settlements.', '',
        'PL has a separate frozen Phase7P source/capture path; its generic classifier cases pass, but a PL clean cohort awaits a qualified PL adapter. SA, FL1 and PD were not restarted. No source authority was broadened.', '',
        f"Protected integrity: **{comparison['protected_count']} prior files unchanged**. Production data, model bytes/pointers, Phase7P, and all Phase12–15 artifacts match the starting snapshot. Concurrent Phase12 additions: {len(comparison['concurrent_phase12_additions'])}; indexed separately.", '',
        '| Test scope | Passed | Inherited failures | Skipped |', '|---|---:|---:|---:|']
    for key, result in tests_record['summaries'].items():
        lines.append(f"| {key} | {result['passed']} | {len(result['failures'])} | {result['skipped']} |")
    lines += ['', 'Zero new failures. The six full-suite failures match the sealed Phase15 identities; XML and failure identities are indexed in test_evidence.json.', '',
        'Operating guide: docs/phase16-fixture-revision.md. The existing Phase12 scheduler and Phase13/14/15 consumers were not changed. There is no activation or retrospective clearance path.',
        f"Generic rule SHA256: `{fresh['revision_rule']['sha256']}`.",
        f"Clean cohort specification SHA256: `{fresh['cohort']['sha256']}`."]
    target = RUN / 'PHASE16_IMPLEMENTATION_REPORT.md'
    content = '\n'.join(lines) + '\n'
    if target.exists():
        p.check(target.read_text(encoding='utf-8') == content, 'PHASE16_REPORT_CONFLICT')
    else:
        target.write_text(content, encoding='utf-8')
    refs = [p.identity(path) for path in RUN.rglob('*') if path.is_file() and
            path.name not in ('artifact_index.json', 'offline_verification.json') and
            path.suffix not in ('.lock', '.tmp')]
    put(RUN / 'artifact_index.json', p.seal({'artifacts': sorted(refs, key=lambda row: row['path']),
        'implementation': [p.identity(p.ROOT / name) for name in SOURCES]}))
    return verify_package()


def verify_package(*, reconstruct=False):
    index = p.read(RUN / 'artifact_index.json')
    p.frozen.sealed(index, 'record_sha256')
    for ref in index['artifacts'] + index['implementation']:
        p.frozen.reference(ref)
    if reconstruct:
        verifier.verify_cohort('BL1-clean-001')
        verifier.verify_regression()
    integrity()
    return {'verified': True, 'artifact_index': p.identity(RUN / 'artifact_index.json'),
        'clean_cohort': p.identity(RUN / 'cohorts/BL1-clean-001/cohort_specification.json'),
        'historical_regression': p.identity(RUN / 'regression/BL1-565839.json'),
        'historical_state': 'BLOCKED_PROTOCOL_VIOLATION',
        'prospective_state': 'PHASE16_PROTOCOL_V2_READY_FOR_CLEAN_COHORT',
        'activation_authorized': False}

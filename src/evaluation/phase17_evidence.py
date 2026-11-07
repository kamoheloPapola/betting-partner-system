"""Seal and independently verify the Phase 17 partial-readiness package."""
from pathlib import Path

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_evidence, phase17_readiness as r
from src.evaluation.phase14_evidence import tests
from src.evaluation.phase12_operations import put

SOURCES = ('src/evaluation/phase17_pl_adapter.py',
           'src/evaluation/phase17_readiness.py', 'src/evaluation/phase17_evidence.py',
           'scripts/run_phase17_readiness.py', 'tests/unit/test_phase17_readiness.py',
           'docs/phase17-v2-readiness.md')


def artifact_identities():
    rows = {}
    for league in r.COHORT_IDS:
        _, _, controller, _, _ = p.frozen.load_protocol(p.PHASE11, league)
        frozen = controller.owner.specification()
        rows[league] = {'deployment_specification': p.identity(controller.owner.specification_path),
                        'active': frozen['active'], 'candidate': frozen['candidate']}
        for key in ('active', 'candidate'):
            p.frozen.reference(frozen[key])
    pl = p.read(r.PHASE7_SPEC)
    rows['PL'] = {'evaluation_specification': p.identity(r.PHASE7_SPEC),
                  'active_bundle_id': pl['active']['bundle_id'],
                  'active_model_bundle_sha256': pl['active']['model_bundle_sha256'],
                  'candidate_bundle_id': pl['candidate']['bundle_id'],
                  'candidate_model_bundle_sha256': pl['candidate']['model_bundle_sha256']}
    return p.seal({'schema_version': 'ffps.phase17_artifact_identities.v1',
                   'leagues': rows}, 'record_sha256')


def pl_source_lineage():
    roots = (
        p.ROOT / 'baselines/phase7d/phase7d-approved-provider-pl-continuity-79110f9a-20260829-v1',
        p.ROOT / 'baselines/phase7e/phase7e-football-data-co-uk-corners-79110f9a-20260830-v1',
        p.ROOT / 'baselines/phase7er/phase7er-qualified-source-refresh-79110f9a-20260905-v2')
    records = []
    for root in roots:
        receipt = p.read(root / 'offline_verification.json')
        p.check(receipt['status'] == 'VERIFIED' and receipt['errors'] == [] and
                receipt['phase8_authorized'] is False and
                receipt['production_path_changed'] is False,
                'PL_PRIOR_SOURCE_LINEAGE_NOT_VERIFIED')
        records.append({'offline_verification': p.identity(root / 'offline_verification.json'),
                        'classification': receipt['classification']})
    return p.seal({'schema_version': 'ffps.phase17_pl_source_lineage.v1',
                   'prior_receipts': records,
                   'phase7p_specification': p.identity(r.PHASE7_SPEC),
                   'raw_source': p.identity(r.adapter.RAW_2026),
                   'source_metadata': p.identity(r.adapter.SOURCE_2026),
                   'qualified_adapter': p.identity(r.RUN / 'pl_source_qualification.json')},
                  'record_sha256')


def certify(focused, protected, full):
    prior = p.read(r.PHASE16_INDEX.parent / 'test_evidence.json')
    p.frozen.sealed(prior, 'record_sha256')
    inherited_ref = next(ref for ref in prior['reports'] if ref['path'].endswith('/full.xml'))
    inherited = tests(p.frozen.reference(inherited_ref))
    inputs = {'focused': Path(focused), 'protected': Path(protected), 'full': Path(full)}
    summaries = {name: tests(path) for name, path in inputs.items()}
    p.check(not summaries['focused']['failures'] and not summaries['protected']['failures'],
            'PHASE17_FOCUSED_OR_PROTECTED_FAILURE')
    p.check(summaries['full']['failures'] == inherited['failures'],
            'PHASE17_NEW_FULL_SUITE_FAILURE')
    refs = []
    for name, source in inputs.items():
        target = r.RUN / 'tests' / (name + '.xml')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            p.check(target.read_bytes() == source.read_bytes(), 'PHASE17_TEST_REPORT_CONFLICT')
        else:
            target.write_bytes(source.read_bytes())
        refs.append(p.identity(target))
    return put(r.RUN / 'test_evidence.json', p.seal({
        'summaries': summaries, 'inherited': inherited_ref,
        'new_failures': [], 'reports': refs}))


def seal(focused, protected, full):
    result = r.verify()
    test_evidence = certify(focused, protected, full)
    put(r.RUN / 'artifact_identities.json', artifact_identities())
    put(r.RUN / 'pl_source_lineage.json', pl_source_lineage())
    integrity = put(r.RUN / 'protected_comparison.json', p.seal(r.verify_integrity()))
    state = p.read(r.RUN / 'readiness.json')
    lines = [
        '# Phase 17 — all-league v2 readiness audit', '',
        '**Outcome: `PHASE17_PARTIAL_READY_PL_PROTOCOL_BLOCKED`.** Four clean v2 cohorts are ready. '
        'PL has an offline-qualified source adapter, but no clean cohort because no PL Phase 11 '
        'prospective protocol or inherited Phase 11 horizon exists.', '',
        '| League | Clean cohort | Season | Start | Fixed end | Pairs | Settlements | State |',
        '|---|---|---:|---|---|---:|---:|---|',
    ]
    for row in state['leagues']:
        lines.append(f"| {row['league']} | {row['cohort_id'] or 'none'} | {row['season']} | "
                     f"{row['start_at'] or 'none'} | {row['end_at']} | 0 | 0 | {row['state']} |")
    lines += ['',
        'The Phase 16 v2 rule, BL1 clean cohort and all Phase 12–16 sealed files remain byte-identical. '
        'The historical BL1 fixture 565839 stays in its blocked original lineage. SA, FL1 and PD '
        'were initialized through the existing generic Phase 16 path; all four new-cohort journals are empty.', '',
        'PL source qualification reconstructs 380 raw archived 2026/27 provider events (365 timed, '
        '15 finished) from the Phase 7D response and sidecar. The Phase 7E exact aliases and field '
        'authority remain in force. Identical stored input produces identical normalized output. '
        'Focused tests exercise kickoff changes, duplicate retrieval, terminal statuses, malformed '
        'material, final scores and orientation. The single real response has no archived kickoff '
        'revision or terminal example; this report does not claim otherwise.', '',
        'PL Phase 7P has a fixed December 31, 2026 fixture-date window. Substituting it for the '
        'requested Phase 11 inherited horizon would create a new scientific contract. The PL cohort '
        'was not initialized, and no prediction, settlement, scheduler, model, pointer or activation '
        'operation ran.', '',
        f"Protected integrity: {integrity['protected_count']} prior files unchanged; production and PL Phase 7P unchanged.",
        '', '| Tests | Passed | Inherited failures | Skipped |',
        '|---|---:|---:|---:|']
    for name, item in test_evidence['summaries'].items():
        lines.append(f"| {name} | {item['passed']} | {len(item['failures'])} | {item['skipped']} |")
    lines += ['', 'The full-suite failure identities match the six inherited Phase 16 failures. '
              'See `test_evidence.json` and indexed XML. Offline verification reconstructs the '
              'readiness matrix and PL normalization from stored source bytes.', '',
              'Operating guide: `docs/phase17-v2-readiness.md`.']
    report = r.RUN / 'PHASE17_IMPLEMENTATION_REPORT.md'
    content = '\n'.join(lines) + '\n'
    if report.exists():
        p.check(report.read_text(encoding='utf-8') == content, 'PHASE17_REPORT_CONFLICT')
    else:
        report.write_text(content, encoding='utf-8')
    refs = [p.identity(path) for path in r.RUN.rglob('*') if path.is_file() and
            path.name not in ('artifact_index.json', 'offline_verification.json') and
            path.suffix not in ('.lock', '.tmp')]
    put(r.RUN / 'artifact_index.json', p.seal({'artifacts': sorted(refs, key=lambda x:x['path']),
        'implementation': [p.identity(p.ROOT/name) for name in SOURCES]}))
    return verify_package()


def verify_package():
    index = p.read(r.RUN / 'artifact_index.json')
    p.frozen.sealed(index, 'record_sha256')
    for ref in index['artifacts'] + index['implementation']:
        p.frozen.reference(ref)
    phase16_evidence.verify_package(reconstruct=False)
    reconstructed = r.verify()
    identities = p.read(r.RUN / 'artifact_identities.json')
    p.frozen.sealed(identities, 'record_sha256')
    p.check(identities == artifact_identities(), 'PHASE17_ARTIFACT_IDENTITIES_DRIFT')
    lineage = p.read(r.RUN / 'pl_source_lineage.json')
    p.frozen.sealed(lineage, 'record_sha256')
    p.check(lineage == pl_source_lineage(), 'PHASE17_PL_LINEAGE_DRIFT')
    evidence = p.read(r.RUN / 'test_evidence.json')
    p.frozen.sealed(evidence, 'record_sha256')
    inherited = tests(p.frozen.reference(evidence['inherited']))
    observed = {name: tests(p.frozen.reference(ref)) for name, ref in
                zip(('focused','protected','full'), evidence['reports'])}
    p.check(observed == evidence['summaries'] and not observed['focused']['failures'] and
            not observed['protected']['failures'] and
            observed['full']['failures'] == inherited['failures'] and
            evidence['new_failures'] == [], 'PHASE17_TEST_EVIDENCE_MISMATCH')
    return {'verified': True, 'state': reconstructed['scientific_state'],
            'artifact_index': p.identity(r.RUN / 'artifact_index.json'),
            'protected_count': reconstructed['integrity']['protected_count'],
            'inherited_failures': len(inherited['failures']), 'new_failures': 0,
            'pl_cohort_initialized': False, 'activation_authorized': False}

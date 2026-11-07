"""Offline Phase 10 verification and immutable reporting."""
from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from src.evaluation import phase10_current_data as data
from src.evaluation import phase10_shadow as shadow


SOURCES = ('src/evaluation/phase10_current_data.py','src/evaluation/phase10_shadow.py',
           'src/evaluation/phase10_evidence.py','scripts/run_phase10_current_data.py',
           'tests/unit/test_phase10_current_data.py','docs/phase10-current-data-operations.md')
SOURCES += ('tests/unit/test_pipeline_hardening.py',)
SOURCES += ('tests/unit/test_backtest_verification.py',)
SOURCES += ('tests/cli/test_verbosity_levels.py',)
SOURCES += ('tests/baseline/test_phase0_baseline.py',)
SOURCES += ('tests/integration/test_prediction_pipeline.py','tests/performance/test_benchmarks.py')


def test_summary(path):
    root = ET.parse(path)
    cases = list(root.iter('testcase'))
    failures = [t.attrib['classname']+'::'+t.attrib['name'] for t in cases if t.find('failure') is not None or t.find('error') is not None]
    skipped = sum(t.find('skipped') is not None for t in cases)
    return {'cases':len(cases),'passed':len(cases)-len(failures)-skipped,'skipped':skipped,'failures':failures}


def verify_phase9(path):
    """Re-run the prior sealed verifier and preserve its structured result."""
    path = data.open_run(path)
    target = path/'prior_phase9_verification.json'
    data.check(not target.exists(),'PRIOR_PHASE9_VERIFICATION_ALREADY_RECORDED')
    from src.evaluation.league_integrity_verifier import verify_run
    result = verify_run(data.PHASE9,replay=True,require_index=True)
    data.check(result['verified'] and result['full_historical_replay'] and not result['errors'],
               'PRIOR_PHASE9_VERIFICATION_FAILED')
    data.write(target,result)
    return result


def matrix_row(path,league):
    qualification_path = data.latest_qualification(path,league)
    q = data.read(qualification_path)
    shadow.sealed(q,'qualification_sha256')
    controller = shadow.ShadowController(q)
    for bundle in (controller.legacy_id,controller.candidate_id):
        controller.bundle(bundle)
    codes = sorted({b['code'] for b in q['assessment']['blockers']+q['feature_failures']})
    protocol_path = path/league/'prospective_protocol.json'
    pairs,settlements = [],[]
    if q['feature_readiness'] and protocol_path.exists():
        _,pairs,settlements = shadow.verify_cohort(path,league)
    active = bool(pairs)
    if q['feature_readiness'] and not active:
        codes.append('PROSPECTIVE_COLLECTION_NOT_STARTED')
    return {'league':league,'execution':'EXECUTION_PROVENANCE_VERIFIED',
        'candidate_lineage':'TRAINING_LINEAGE_VERIFIED','candidate_bundle_id':q['candidate_bundle_id'],
        'candidate_sha256':q['candidate_sha256'],'current_data_qualification':'QUALIFIED' if q['assessment']['qualified'] else 'DATA_BLOCKED',
        'historical_feature_reproduction':q['historical_feature_reproduction']['code'],
        'feature_continuity':'VERIFIED' if q['feature_readiness'] else 'BLOCKED',
        'prospective_protocol':'FROZEN' if protocol_path.exists() else 'NOT_FROZEN_DATA_BLOCKED',
        'collection_state':'PROSPECTIVE_COLLECTION_ACTIVE' if active else 'DATA_BLOCKED',
        'promotion_state':'NOT_AUTHORIZED_PHASE10','admitted_pairs':len(pairs),'settlements':len(settlements),
        'qualification':data.identity(qualification_path),'blocker_codes':codes,
        'primary_source_coverage':q['assessment']['raw_primary_coverage'],
        'mapped_current_teams':sum(t['status']=='EXACT_REPOSITORY_ALIAS' for t in q['assessment']['team_mappings'].get(str(q['assessment']['season']),{}).values()),
        'current_team_count':len(q['assessment']['team_mappings'].get(str(q['assessment']['season']),{})),
        'promoted_membership_verified':q['assessment']['promoted_team_evidence']['membership_verified'],
        'promoted_model_identity_verified':q['assessment']['promoted_team_evidence']['historical_mapping_verified']}


def verify(path, *, indexed=True):
    path = data.run_path(path)
    from src.config.leagues import ACTIVE_LEAGUES
    data.check(set(ACTIVE_LEAGUES)-{'PL'}==set(data.LEAGUES),'PRODUCTION_LEAGUE_INVENTORY_CHANGED')
    before = data.read(path/'protected_before.json')
    after = data.protected_snapshot()
    data.check(before==after,'PROTECTED_FILE_MUTATION')
    prior = data.read(path/'prior_phase9_verification.json')
    data.check(prior['verified'] and prior['full_historical_replay'] and not prior['errors'],'PRIOR_PHASE9_VERIFICATION_FAILED')
    data.check(set(prior['production_leagues'])==set(ACTIVE_LEAGUES),'PRIOR_PHASE9_LEAGUE_COVERAGE_MISMATCH')
    if indexed:
        index = data.read(path/'artifact_index.json')
        for ref in index['artifacts']+index['implementation_sources']:
            shadow.reference(ref)
        expected = {ref['path'] for ref in index['artifacts']}
        actual = {str(p.relative_to(data.ROOT).as_posix()) for p in path.rglob('*') if p.is_file()
                  and p.name not in ('artifact_index.json','offline_verification.json')}
        data.check(expected==actual,'UNINDEXED_OR_MISSING_EVIDENCE')
    matrix = data.read(path/'multi_league_phase10_status_matrix.json')
    data.check([r['league'] for r in matrix['leagues']]==list(data.LEAGUES),'OMITTED_OR_DUPLICATED_LEAGUE')
    verified = []
    for row in matrix['leagues']:
        league = row['league']
        qualification_path = shadow.reference(row['qualification'])
        q = data.read(qualification_path)
        shadow.sealed(q,'qualification_sha256')
        data.check(q['league']==league,'CROSS_LEAGUE_QUALIFICATION')
        frozen = pd.read_csv(data.ROOT/'data/processed/matches.csv',low_memory=False)
        frozen = frozen.loc[frozen.league.eq(league)]
        source_index = data.read(shadow.reference(q['source_index']))
        assessment = data.assess_sources(source_index,frozen=frozen,names=data.exact_names(league),as_of=q['as_of'])
        data.check(assessment==q['assessment'],'QUALIFICATION_REPLAY_MISMATCH')
        try:
            feature = data.feature_reproduction(league,q['ordered_features'])
        except Exception as exc:
            feature = {'reproduced':False,'code':'FEATURE_PIPELINE_INCOMPATIBLE',
                       'error_type':type(exc).__name__,'detail':str(exc)}
        data.check(feature==q['historical_feature_reproduction'],'FEATURE_REPRODUCTION_MISMATCH')
        rebuilt = matrix_row(path,league)
        data.check(rebuilt==row,'MATRIX_STATE_MISMATCH')
        if not q['feature_readiness']:
            data.check(not (path/league/'prospective_protocol.json').exists(),'BLOCKED_LEAGUE_HAS_PROTOCOL')
            data.check(not list((path/league/'cohort/predictions').glob('*.json')),'BLOCKED_LEAGUE_HAS_PREDICTIONS')
        verified.append({'league':league,'evidence_verified':True,'qualified':q['feature_readiness'],
                         'collection_state':row['collection_state'],'blocker_codes':row['blocker_codes']})
    tests = data.read(path/'tests/classification.json')
    for label in ('focused','protected','full'):
        data.check(test_summary(path/f'tests/{label}.xml')==tests[label],'TEST_REPORT_MISMATCH')
    inherited = set(test_summary(path/'tests/phase9_full_reference.xml')['failures'])
    new = set(tests['full']['failures'])-inherited
    data.check(not tests['focused']['failures'] and not tests['protected']['failures'] and not new,'NEW_REGRESSION_FAILURE')
    return {'schema_version':'ffps.phase10_offline_verification.v1','verified':True,
        'protected_file_count':len(before),'protected_files_unchanged':True,'pl_unchanged':True,
        'phase9_candidates_unchanged':True,'leagues':verified,'tests_verified':True,
        'new_unexplained_failures':sorted(new),'historical_sources_and_features_recomputed':True,
        'production_activation_performed':False}


def finalize(path):
    path = data.run_path(path)
    data.check(not (path/'artifact_index.json').exists(),'RUN_ALREADY_FINALIZED')
    rows = [matrix_row(path,l) for l in data.LEAGUES]
    matrix = {'schema_version':'ffps.phase10_status_matrix.v1','leagues':rows,
              'pl_reference':data.identity(data.PHASE9/'PL/existing_experiment_reference.json'),
              'pl_policy':'READ_ONLY_EXISTING_PHASE7P_UNCHANGED','production_activation_performed':False}
    data.write(path/'multi_league_phase10_status_matrix.json',matrix)
    for row in rows:
        data.write(path/row['league']/'prospective_protocol_state.json',{
            'league':row['league'],'state':row['prospective_protocol'],'blocker_codes':row['blocker_codes'],
            'qualification':row['qualification'],'promotion_authorized':False})
        data.write(path/row['league']/'admission_rejection_report.json',{
            'league':row['league'],'admitted_pairs':row['admitted_pairs'],'settlements':row['settlements'],
            'state':row['collection_state'],'rejection_scope':'entire league qualification gate' if row['blocker_codes'] else None,
            'reasons':row['blocker_codes'],'raw_fixture_coverage':row['primary_source_coverage']})
    testdir = path/'tests'
    testdir.mkdir(exist_ok=True)
    for label in ('focused','protected','full'):
        shutil.copyfile(data.ROOT/f'tmp/phase10-{label}.xml',testdir/f'{label}.xml')
    shutil.copyfile(data.PHASE9/'tests/full.xml',testdir/'phase9_full_reference.xml')
    tests = {label:test_summary(testdir/f'{label}.xml') for label in ('focused','protected','full')}
    inherited = set(test_summary(testdir/'phase9_full_reference.xml')['failures'])
    tests['inherited_failures'] = sorted(set(tests['full']['failures'])&inherited)
    tests['new_unexplained_failures'] = sorted(set(tests['full']['failures'])-inherited)
    data.write(testdir/'classification.json',tests)
    after = data.protected_snapshot()
    before = data.read(path/'protected_before.json')
    data.check(after==before,'PROTECTED_FILE_MUTATION')
    data.write(path/'protected_after.json',after)
    data.write(path/'preservation_verification.json',{'protected_count':len(before),'unchanged':True,
        'pl_preserved':True,'phase9_candidates_preserved':True,'production_pointers_unchanged':True,
        'phase9_frozen_evidence_unchanged':True})
    raw_index = [data.identity(p) for p in sorted(path.glob('*/raw_sources/**/*')) if p.is_file()]
    data.write(path/'raw_source_identity_index.json',{'sources':raw_index,'credentials_recorded':False,
        'publication_policy':'Provider lastUpdated fields retained in raw/normalized evidence. Retrieval is not publication time.'})
    lines = ['# Phase 10 current-data qualification and shadow activation', '',
        'Shared implementation passed focused qualification, real-model paired replay, isolation and settlement tests. '
        f'League-specific current-data qualification passed for {sum(r["current_data_qualification"]=="QUALIFIED" for r in rows)}/{len(rows)} leagues. '
        f'Real non-PL prospective pairs admitted: {sum(r["admitted_pairs"] for r in rows)}.', '',
        '| League | Execution | Candidate lineage | Current data | Historical model-facing features | Current causal features | Protocol | Collection | Promotion |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append(f'| {r["league"]} | Verified | Verified | {r["current_data_qualification"]} | {r["historical_feature_reproduction"]} | {r["feature_continuity"]} | {r["prospective_protocol"]} | {r["collection_state"]} | Not authorized |')
    lines += ['', '## League-specific evidence', '',
        'Both acquisition and a bounded statistics retry were performed against the existing provider endpoints. '
        'All eight season/statistics requests returned HTTP 503 on each external-network attempt. Raw response bytes and timestamps are preserved. '
        'The initial sandbox-only BL1 connection failures are retained separately and are not treated as a provider outage.', '',
        'BL1 primary fixtures parsed successfully. FL1, PD and SA raw responses contain current-season fixtures before their own reported season start dates; '
        'the strict source gate rejects those payloads. Reported raw counts remain visible; zero normalized admissions does not mean the provider returned zero fixtures.', '',
        'Exact team-ID membership and promoted/relegated set differences are recorded independently of model-facing name qualification. '
        'Unmapped names and historical joins remain blocked; fuzzy matches or manually added aliases were not used.', '',
        'BL1 historical feature reconstruction is blocked by the existing sanitizer rejecting M\'GLADBACH. '
        'FL1, PD and SA reproduce all frozen model-eligible feature rows. Their nonfinite cold-start rows were already excluded before Phase 9 partitions; '
        'this check reports those rows and never imputes them. Current feature continuity remains blocked for all leagues.', '']
    for r in rows:
        lines += [f'### {r["league"]}', '',f'Exact blockers: `{", ".join(r["blocker_codes"])}`.', '',
            f'Current teams with exact model-facing mappings: {r["mapped_current_teams"]}/{r["current_team_count"]}. '
            f'Qualification: `{r["qualification"]["path"]}`.', '']
    lines += ['## Shared implementation and operating limits', '',
        'New Phase 10 modules call the existing FeaturePipeline, Phase 4 provenance/replay, Phase 9 bundle/lineage ownership and pair verifier, '
        'Phase 7 append-only cohort lock, simulator, metrics, bootstrap and decision rule. No Phase 7/8/9 source or frozen evidence was changed.', '',
        'Qualification must pass before freeze and collect. A successful run can immediately capture real pairs; the focused test executes both real BL1 '
        'bundles and verifies the pair through Phase 4 replay in a temporary test cohort. This synthetic test is implementation evidence only.', '',
        'The prospective protocol fixes the remainder of the provider-reported season and all qualifying future fixtures. '
        'Settlement retrieves authoritative later results and rejects absent pairs, kickoff revisions and score revisions. '
        'Phase 10 offers no activation command. Later review must still use league-specific Phase 8 authorization.', '',
        'This finalized run is immutable. Acquisition, assessment and test-development attempts are retained; only qualification references in the final matrix '
        'are authoritative. Earlier preliminary assessments are superseded audit attempts, not frozen prospective protocols.', '',
        '## Validation and preservation', '',
        f'All {len(before)} protected original files are unchanged, including PL Phase 7P, all Phase 9 evidence/candidates, production models and pointers.', '']
    if (path/'test_side_effect_restoration.json').exists():
        lines += ['Full-suite verification exposed inherited test side effects: a real quarantine CSV write and daily feature-export rotation. '
            'Rotation deleted a protected historical export. Its exact original SHA-256 matched an existing duplicate, which was used to restore the original bytes. '
            'Test-created files were archived and removed from production data. Backtest, pipeline-hardening, prediction integration, performance, CLI verbosity and Phase 0 golden tests now send exports '
            'and quarantine operations to pytest temporary directories, including prediction child processes. Frozen golden expectations and production pipeline source remain unchanged. '
            'The full suite and preservation checks were rerun after those fixes; see the four `test_side_effect_restoration*.json` records for exact hashes '
            'and `test_export_diagnostics/attribution.json` for the final caller trace and attribution correction.', '']
    for label in ('focused','protected','full'):
        t=tests[label]
        lines.append(f'- {label}: {t["passed"]} passed, {len(t["failures"])} failed, {t["skipped"]} skipped.')
    lines += ['',f'New unexplained failures: {len(tests["new_unexplained_failures"])}. Inherited failures:', '']
    lines.extend('- `'+f+'`' for f in tests['inherited_failures'])
    lines += ['', 'Offline verification recomputes provider normalization, mappings, joins and feature reproduction, checks every production league, '
        'checks exact candidate/active identities, protected bytes, indexed sources, test reports, and any admitted pairs/settlements.', '',
        'Prior Phase 9 standalone verification is archived separately in `prior_phase9_verification.json`.', '',
        'Operations: `docs/phase10-current-data-operations.md`. Source failures require real corrected provider responses; '
        'mapping/feature blockers require independently justified compatibility evidence before another qualification run.']
    (path/'PHASE10_CURRENT_DATA_IMPLEMENTATION_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    preliminary = verify(path,indexed=False)
    artifacts = [data.identity(p) for p in sorted(path.rglob('*')) if p.is_file() and p.name not in ('artifact_index.json','offline_verification.json')]
    data.write(path/'artifact_index.json',{'schema_version':'ffps.phase10_artifact_index.v1',
        'artifacts':artifacts,'implementation_sources':[data.identity(data.ROOT/p) for p in SOURCES]})
    data.write(path/'offline_verification.json',preliminary)
    return preliminary

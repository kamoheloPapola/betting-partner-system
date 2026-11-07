"""Seal the Phase 9 run only after test and protected-artifact checks pass."""
from __future__ import annotations

import shutil
from pathlib import Path

from src.evaluation.multi_league_integrity import ROOT, read, write, identity
from src.predictions.provenance import sha256_file,sha256_json,utc_now
from src.ml.league_integrity import require


def finalize(run_dir, *, focused, protected, full):
    from src.ml.bundle_evidence import junit,verify_infrastructure
    from src.evaluation.league_integrity_verifier import verify_run
    before_path = ROOT/'tmp/phase9-before-full.xml'
    tests = {name:junit(path) for name,path in [('focused',focused),('protected',protected),('full',full),('before',before_path)]}
    tests['new_unexplained_failure_ids'] = sorted(set(tests['full']['failure_ids'])-set(tests['before']['failure_ids']))
    require(tests['focused']['failures'] == tests['protected']['failures'] == 0,'focused/protected tests failed')
    require(not tests['new_unexplained_failure_ids'],'new unexplained full-suite failures')
    before = read(run_dir/'protected_before.json')
    current = {path:sha256_file(ROOT/path) for path in before}
    require(before == current,'protected original bytes changed')
    preliminary = verify_run(run_dir,require_index=False,replay=True)
    require(preliminary['verified'],str(preliminary['errors']))
    phase8 = verify_infrastructure(ROOT/'baselines/phase8/phase8-controlled-pl-bundle-20260905-v1')
    write(run_dir/'prior_phase8_verification.json',phase8)
    for name,path in [('focused',focused),('protected',protected),('full',full),('before',before_path)]:
        target = run_dir/'tests'/f'{name}.xml'
        target.parent.mkdir(exist_ok=True)
        with target.open('xb') as stream:
            stream.write(path.read_bytes())
    write(run_dir/'test_results.json',tests)
    write(run_dir/'protected_after.json',current)
    inventory = read(run_dir/'league_inventory.json')
    matrix = read(run_dir/'multi_league_integrity_matrix.json')
    table = ['| League | Production | Execution | Active Lineage | Candidate | Data Ready | Shadow | Promotion |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for league in matrix['leagues']:
        table.append(f"| {league['league']} | Enabled | Verified | Legacy unproven | {'Existing Phase 6' if league['league']=='PL' else 'Verified fresh bundle'} | {'Existing qualified PL stream' if league['data_ready'] else 'Blocked'} | {'Existing Phase 7P active' if league['league']=='PL' else 'Historical ready; prospective blocked'} | Blocked |")
    partitions = ['| League | Features | Train | Validation | Calibration reserve | Protected diagnostic | Excluded cold starts |',
                  '| --- | --- | --- | --- | --- | --- | --- |']
    metrics = ['| League | Active / candidate log loss | Active / candidate Brier | Active / candidate goal RMSE | Active / candidate Poisson deviance |',
               '| --- | --- | --- | --- | --- |']
    coverage = ['| League | Latest canonical row | Matched unique historical fixtures / canonical total | Exact-equality mismatches across compared fields |',
                '| --- | --- | --- | --- |']
    for row in inventory['leagues']:
        league = row['league']
        data = read(run_dir/league/'data_continuity.json')
        coverage.append(f"| {league} | {row['latest_causal_data_row']} | {data['canonical_unique_fixtures_matched']} / {data['canonical_fixture_count']} | {sum(a['mismatches'] for a in data['field_authorities'])} |")
        if league == 'PL':
            continue
        directory = run_dir/league
        spec = read(directory/'candidate/training_specification.json')
        parts = spec['partitions']
        excluded = read(directory/'candidate/excluded_cold_start_rows.json')['count']
        partitions.append('| '+league+' | '+str(row['feature_schema']['count'])+' | '+ ' | '.join(
            f"{parts[p]['row_count']} ({parts[p]['start']}–{parts[p]['end']})" for p in ('training','validation','calibration_fit','protected_evaluation'))+f' | {excluded} |')
        scores = read(directory/'historical_evaluation.json')['metrics']
        metrics.append('| '+league+' | '+' | '.join(f"{scores['active'][section][name]:.6f} / {scores['candidate'][section][name]:.6f}"
            for section,name in [('1x2','log_loss'),('1x2','multiclass_brier'),('goals','rmse'),('goals','mean_poisson_deviance')])+' |')
    lines = [
        '# FFPS Phase 9 multi-league implementation report','',
        'Phase 9 inventory, execution/lineage assessment, candidate training, historical evaluation and offline verification cover all five production leagues. BL1 passed the first portability gate before the remaining candidate runs. No production model was replaced and no league was promoted.','',
        '## Multi-league integrity matrix','',*table,'',
        'BL1, FL1, PD and SA have the mechanical state `SHADOW_EVALUATION_READY` and the operational outcome `CANDIDATE_READY_DATA_BLOCKED`. PL retains its existing production bundle, trusted Phase 6 shadow candidate and ongoing Phase 7P collection. Its Phase 8 scientific decision remains inconclusive.','',
        '## Shared implementation passed','',
        '- Reused `ffps.prediction_provenance.v1`, `ffps.model_training_lineage.v1`, the Phase 6 fresh-fit primitives and Phase 7 scoring/reliability functions.',
        '- Added immutable LeagueModelBundle ownership and league-bound execution, lineage, schema and field-authority checks. Artifact SHA-256 is the execution/lineage join.',
        '- Generalized Phase 8 around league specifications while retaining its transition journal, atomic pointer replacement, runtime quiescence, generation checks, recovery and rollback. The PL pointer/schema and scientific specification remain unchanged.',
        '- Production registry routing recognizes an explicit league pointer and protects every league’s four-model bootstrap. No non-PL pointer, activation history or pending transition was created by this rollout.',
        '- Shared prospective pair/settlement admission rejects retrospective records and requires a league-frozen data contract/horizon. The final verifier recomputes the existing Phase 7 zero-margin decision from the complete authoritative league cohort. No non-PL prospective protocol has been admitted.',
        '- Offline verification reconstructs exact fit/validation matrices and targets, checks all partitions and repeat-fit artifacts, replays every historical paired model/simulation output, recomputes metrics and verifies the matrix mechanically. Batch failures are explicit; no configured production league is skipped.','',
        '## League-specific evidence passed','',
        'Every league has exact active artifact paths/hashes, model families, embedded and configured tree counts, manifest pointers and independently inspected ordered feature names. All five currently expose the same 30-feature goal contract; that was measured per artifact rather than assumed. All 20 active goal models remain `LEGACY_TRAINING_LINEAGE_UNPROVEN`: exact historical training/validation rows, targets, continuation history, historical source revision and runtime are not mechanically established.','',
        'Each of the four non-PL candidates was fitted twice from zero estimator state using unchanged Phase 6 configurations (500 LightGBM trees, 600 XGBoost trees). All 16 candidate members passed exact repeated prediction comparison within the frozen 1e-12 tolerance and the original Phase 5 lineage verifier. Repeat artifact byte identities and runtime/library versions are recorded per member.','',
        'The four candidates are stored only under their league evidence directories. They have explicit diagnostic validation metadata for the existing prediction guard; no model-level or 1X2 calibrator was fitted or inserted. The match-outcome classifier is not used by the verified goal-ensemble 1X2 path. The broader market inventory describes repository support; Phase 9 does not confer lineage trust on corner/card/classifier artifacts.','',
        *partitions,'',
        'Partitions follow actual match dates and league-local dated season cohorts. Stored season labels are retained but not used to place the mislabeled early BL1 dates. Pairwise fixture overlap is zero. The latest two observed cohorts are protected historical diagnostics, preceded by separate calibration and validation cohorts. Non-finite cold-start feature rows are excluded by a recorded pre-fit feature-only rule; no values were manually filled and no outcome was consulted for eligibility.','',
        '## Historical diagnostics','',*metrics,'',
        'Metrics use identical paired fixtures/features and the same seed-derived simulator state. Reliability diagnostics include top-label and classwise ECE with ten bins. All comparisons remain diagnostic because the active legacy fitting lineage is unknown and historical exposure cannot be ruled out. These metrics authorize neither tuning nor promotion.','',
        '## Data and prospective blockers','',*coverage,'',
        'Overlap qualification is league-specific, uses ordered canonical team/date joins and records field-level source hashes, values and row indices. Only the matched historical snapshots are qualified; unmatched rows and source parse problems remain explicit in each evidence bundle. Exact equality on a partial overlap does not establish whole-provider or current-season coverage.','',
        'For BL1, FL1, PD and SA, no current-season league-qualified fixture/result/corner stream with source retrieval/publication timing, promoted-team mapping and causal feature-continuity evidence was admitted. Consequently no prospective pair was retrospectively reconstructed, no live current-data qualification was inferred from PL, and no non-PL horizon was fabricated. A league-specific current-data qualification and prospective protocol freeze are required before collection can start. The shared prospective final gate is implemented, but no scientifically eligible non-PL activation exists or was exercised on real prospective evidence.','',
        'PL’s later Phase 7ER qualified-source refresh and resumed Phase 7P stream are referenced separately from its stale canonical CSV. Phase 9 neither rewrites those records nor resets the experiment.','',
        '## Verification and preservation','',
        f"- Focused: {tests['focused']['passed']} passed; {tests['focused']['failures']} failed.",
        f"- Protected regressions: {tests['protected']['passed']} passed; {tests['protected']['failures']} failed.",
        f"- Full suite: {tests['full']['passed']} passed, {tests['full']['failures']} failed, {tests['full']['skipped']} skipped.",
        f"- Baseline full suite: {tests['before']['passed']} passed, {tests['before']['failures']} failed, {tests['before']['skipped']} skipped. New unexplained failure IDs: none.",
        f'- Protected original files: {len(before)}; before/after SHA-256 maps are identical. Prior Phase 8 infrastructure verifies offline against its exact archived source versions.',
        '- Cross-league rejection, per-league activation/rollback isolation and post-replacement recovery are tested using temporary state stores. Successful transition tests inject authorization at the unit boundary; they are not scientific promotion evidence.',
        '- The initial baseline test run exposed an existing strategy unit test appending one stub row to the real prediction log. The original bytes were restored only after matching their complete SHA-256 prefix; the removed test row and restoration proof are retained. That test now isolates the logger.',
        '- An early concurrent focused/protected run contended on the existing PL process lock. The protected suite was rerun alone and passed; the lock behavior was preserved.','',
        'Inherited full-suite failures:','',*['- `'+name+'`' for name in tests['full']['failure_ids']],'',
        '## Deliverables and rerun','',
        'The run contains `league_inventory.json`, one directory per league with integrity/feature specifications and evidence, `multi_league_integrity_matrix.json`, indexed test reports, protected hash maps and `offline_verification.json`. The shared runner and verifier are `scripts/run_phase9_multi_league.py` and `src/evaluation/league_integrity_verifier.py`. Operational usage is documented in `docs/phase9-multi-league-integrity.md`.','',
        '```powershell',
        'venv/Scripts/python.exe scripts/run_phase9_multi_league.py verify --run-dir baselines/phase9/phase9-multi-league-20260905-v1',
        '```','',
        'Shared implementation passing and league-specific evidence passing are separate conclusions. The current data and scientific blockers remain enforced; completion of this integrity rollout does not imply promotion readiness.','']
    report = run_dir/'PHASE9_MULTI_LEAGUE_IMPLEMENTATION_REPORT.md'
    with report.open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    sources = ['scripts/run_phase9_multi_league.py','scripts/run_phase8_bundle_activation.py',
        'src/evaluation/multi_league_integrity.py','src/evaluation/league_candidate.py','src/evaluation/league_provider_qualification.py',
        'src/evaluation/league_integrity_verifier.py','src/evaluation/league_prospective.py','src/evaluation/phase9_report.py',
        'src/ml/league_integrity.py','src/ml/league_bundle_activation.py','src/ml/bundle_activation.py','src/ml/bundle_evidence.py','src/ml/registry.py',
        'tests/unit/test_phase9_multi_league.py','tests/strategies/test_forbidden_fruit_value_opportunities.py','docs/phase9-multi-league-integrity.md']
    artifacts = [identity(p) for p in sorted(run_dir.rglob('*')) if p.is_file() and p.name not in ('artifact_index.json','offline_verification.json')]
    write(run_dir/'artifact_index.json',{'schema_version':'ffps.phase9_artifact_index.v1','artifacts':artifacts,
        'aggregate_sha256':sha256_json(artifacts),'sources':[identity(ROOT/p) for p in sources],'sealed_at':utc_now()})
    final = verify_run(run_dir,require_index=True,replay=False)
    require(final['verified'],str(final['errors']))
    final['full_historical_replay'] = preliminary['full_historical_replay']
    final['league_results'] = preliminary['league_results']
    write(run_dir/'offline_verification.json',final)
    return {'verified':True,'report':str(report),'matrix':str(run_dir/'multi_league_integrity_matrix.json'),
            'test_results':tests,'protected_file_count':len(before)}

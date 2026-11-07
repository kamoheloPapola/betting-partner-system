"""League-isolated evidence around the existing FFPS prediction mathematics.

Inventory is a separate, read-only production audit and must precede assessment.
All newly created evidence is exclusive-create beneath the chosen evidence run.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.leagues import ACTIVE_LEAGUES, LEAGUE_METADATA
from src.config.markets import MARKET_DEFINITIONS
from src.predictions.provenance import sha256_file, sha256_json, utc_now

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / 'src/ml/models'
ROLES = dict(home_goals_lgbm='home_goals', away_goals_lgbm='away_goals',
             home_goals_xgb='home_goals_xgb', away_goals_xgb='away_goals_xgb')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve exact executed source versions when later shared work changes the
    # working tree. Existing Phase 4 source verification remains strict.
    if isinstance(value, dict):
        def archive_records(record):
            if record.get('schema_version') == 'ffps.prediction_provenance.v1' and 'code_provenance' in record:
                from src.ml.bundle_evidence import historical_source
                for parent in path.parents:
                    if parent.parent == ROOT/'baselines/phase9':
                        for item in record['code_provenance']['relevant_source_files']:
                            source = historical_source(ROOT/item['path'],item['sha256'])
                            target = parent/'source_versions'/item['sha256']/item['path']
                            if not target.exists():
                                target.parent.mkdir(parents=True,exist_ok=True)
                                target.write_bytes(source.read_bytes())
                        break
            else:
                for child in record.values():
                    if isinstance(child,dict):
                        archive_records(child)
        archive_records(value)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def identity(path):
    path = Path(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha256_file(path),
            'size_bytes': path.stat().st_size}


def protected_snapshot():
    # Include existing untracked frozen evidence, scientific data and model state.
    paths = [p for base in ('baselines', 'data', 'src/ml/models')
             for p in (ROOT / base).rglob('*') if p.is_file()
             and not p.is_relative_to(ROOT / 'baselines/phase9')]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted(paths)}


def inventory(run_dir):
    if run_dir.exists():
        raise ValueError('inventory evidence already exists; choose a new immutable run')
    run_dir.relative_to(ROOT / 'baselines/phase9')
    before = protected_snapshot()
    manifest = read(MODELS / 'manifest.json')
    raw = pd.read_csv(ROOT / 'data/processed/matches.csv', low_memory=False)
    matrix = pd.read_csv(ROOT / 'data/features/feature_matrix.csv', low_memory=False)
    from src.features.pipeline import FeaturePipeline
    from src.data_pipeline.build_matches_dataset import DIV_MAP
    sources = {
        'production_configuration': sorted(ACTIVE_LEAGUES),
        'metadata': sorted(LEAGUE_METADATA),
        'feature_pipeline': sorted(FeaturePipeline.VALID_LEAGUES),
        'canonical_provider_divisions': sorted(set(DIV_MAP.values())),
        'registry': sorted({m['league'] for m in manifest.values()
                            if isinstance(m, dict) and m.get('league')}),
        'canonical_data': sorted(raw.league.unique().tolist()),
        'feature_data': sorted(matrix.league.unique().tolist()),
    }
    universe = sorted(set().union(*map(set, sources.values())))
    records = []
    for league in universe:
        models = {}
        for role, name in ROLES.items():
            slot = f'{name}_{league}'
            key = manifest['active_models'].get(slot)
            meta = manifest.get(key, {})
            path = MODELS / meta['filename'] if meta.get('filename') else None
            member = {'registry_slot': slot, 'manifest_key': key, 'manifest_entry': meta,
                      'available': bool(path and path.is_file())}
            if member['available']:
                member.update(identity(path))
                try:
                    model = pickle.loads(path.read_bytes())
                    names = list(model.feature_name_ if role.endswith('lgbm') else model.get_booster().feature_names)
                    member.update(model_family=type(model).__module__ + '.' + type(model).__name__,
                                  embedded_tree_count=int(model.booster_.num_trees() if role.endswith('lgbm') else model.get_booster().num_boosted_rounds()),
                                  configured_tree_count=int(model.n_estimators), ordered_feature_names=names,
                                  schema_sha256=sha256_json(names), artifact_manifest_features_equal=names == meta.get('features'))
                except Exception as exc:
                    member['inspection_blocker'] = f'{type(exc).__name__}: {exc}'
            models[role] = member
        rows = raw.loc[raw.league.eq(league)]
        features = matrix.loc[matrix.league.eq(league)]
        feature_names = models['home_goals_lgbm'].get('ordered_feature_names', [])
        finite = bool(feature_names and set(feature_names) <= set(features) and np.isfinite(features[feature_names].to_numpy(float)).all())
        names_equal = all(m.get('ordered_feature_names') == feature_names for m in models.values())
        records.append({
            'league': league, 'display_name': LEAGUE_METADATA.get(league, {}).get('full_name'),
            'production_enabled': league in ACTIVE_LEAGUES, 'prediction_markets_supported': list(MARKET_DEFINITIONS),
            'active_goal_models': models, 'model_artifact_availability': all(m['available'] for m in models.values()),
            'feature_schema': {'ordered_feature_names': feature_names, 'count': len(feature_names),
                               'schema_sha256': sha256_json(feature_names), 'all_goal_models_compatible': names_equal,
                               'all_stored_rows_finite': finite,
                               'preprocessing_contract': 'FeaturePipeline point-in-time engineering; _translate_features; pd.to_numeric; PredictionGuard',
                               'required_raw_historical_fields': ['match_hash','match_date','league','home_team','away_team','home_goals','away_goals','home_corners','away_corners']},
            'data_source': {'canonical': 'football-data.co.uk', 'division': next((k for k,v in DIV_MAP.items() if v == league), None),
                            'fixture_results_integration': 'football-data.org; league-specific current coverage requires qualification',
                            'raw_rows': len(rows), 'feature_rows': len(features),
                            'season_coverage': rows.groupby('season').size().to_dict(),
                            'earliest_row': str(rows.match_date.min()) if len(rows) else None,
                            'latest_row': str(rows.match_date.max()) if len(rows) else None,
                            'missing_required_statistics': {c: int(rows[c].isna().sum()) for c in ['home_goals','away_goals','home_corners','away_corners']},
                            'production_files': [identity(p) for p in sorted((ROOT / 'data/processed/matches').glob(f'{league}_*.csv'))]},
            'latest_causal_data_row': str(rows.match_date.max()) if len(rows) else None,
            'calibrator_pointers': {'model_level': {r: m['manifest_entry'].get('calibrator_filename') for r,m in models.items()},
                                    'market_artifacts': [identity(p) for p in sorted((MODELS / 'calibrators').glob(f'*_{league}_*'))],
                                    '1x2': 'none on inspected _calc_goals raw simulator path', 'historical_fitting_provenance': 'UNPROVEN'},
            'active_bundle_pointer': identity(MODELS / f'active_{league.lower()}_bundle.json') if (MODELS / f'active_{league.lower()}_bundle.json').exists() else None,
            'prediction_pathway': ['src.api.main / src.api.routes.cli / src.cli.main show-predictions', 'Predictor -> FeaturePipeline.run',
                                   '_run_predict_loop -> _load_prediction_models -> ModelRegistry', '_calculate_probabilities_v2 -> _calc_goals',
                                   'four _predict_scalar calls -> 60/40 goal ensemble -> MatchSimulator.simulate',
                                   'raw 1X2; binary goals/BTTS calibration separate; match_outcome classifier unused by goal path'],
            'existing_evidence': {'drift_files': [p.relative_to(ROOT).as_posix() for p in (ROOT/'data/drift').rglob('*') if p.is_file() and league in p.name],
                                  'training_metadata': identity(MODELS/f'training_metadata_{league}.json') if (MODELS/f'training_metadata_{league}.json').exists() else None},
            'current_integrity_state': 'UNASSESSED',
        })
    result = {'schema_version': 'ffps.league_inventory.v1', 'created_at': utc_now(),
              'discovery_sources': sources, 'leagues': records, 'production_leagues': sorted(ACTIVE_LEAGUES),
              'source_identities': [identity(ROOT/p) for p in ['src/config/leagues.py','src/features/pipeline.py','src/ml/registry.py','src/ml/models/manifest.json','src/cli/commands/prediction.py','src/predictions/predictor.py','src/data_pipeline/build_matches_dataset.py','data/processed/matches.csv','data/features/feature_matrix.csv']]}
    write(run_dir/'league_inventory.json', result)
    write(run_dir/'protected_before.json', before)
    return {'inventory': str(run_dir/'league_inventory.json'), 'production_leagues': result['production_leagues'], 'league_count': len(records)}


def assess(run_dir, league=None):
    inv = read(run_dir/'league_inventory.json')
    selected = [r for r in inv['leagues'] if r['production_enabled'] and (league is None or r['league'] == league)]
    if not selected:
        raise ValueError('league is not production enabled in the frozen inventory')
    selected.sort(key=lambda r: (r['league'] != 'BL1', r['league']))
    reports = []
    for row in selected:
        directory = run_dir / row['league']
        if (directory/'audit.json').exists():
            reports.append(read(directory/'audit.json'))
            continue
        try:
            reports.append(audit_league(directory, row))
        except Exception as exc:
            report = {'league':row['league'],'execution':'EXECUTION_PROVENANCE_BLOCKED',
                      'execution_blocker':f'{type(exc).__name__}: {exc}',
                      'active_lineage':'TRAINING_LINEAGE_UNPROVEN','candidate_required':True,
                      'integrity_state':'UNASSESSED'}
            if not (directory/'audit.json').exists():
                write(directory/'audit.json',report)
            reports.append(report)
    return {'reports': reports}


def active_bundle(row):
    from src.ml.league_integrity import LeagueModelBundle
    members = {r: {**m, 'league': row['league']} for r,m in row['active_goal_models'].items()}
    value = {'league': row['league'], 'bundle_id': 'phase9-active-' + row['league'] + '-' + sha256_json(members)[:16],
             'members': members, 'feature_contract': row['feature_schema'],
             'activation_status': 'LEGACY_PRODUCTION_ACTIVE', 'execution_compatibility': 'existing_four_goal_ensemble',
             'environment_config_identity': identity(ROOT/'src/cli/commands/prediction.py')}
    value['bundle_sha256'] = sha256_json(value)
    return LeagueModelBundle.from_dict(value, expected_league=row['league'])


def prediction_probe(row, bundle, suite=None):
    from src.cli.commands.prediction import _load_prediction_models, _calc_goals
    from src.predictions.provenance import (PredictionCapture, build_file_snapshot, source_event_index,
        data_provenance_for_fixture, replay_pl_1x2, mark_output_reproduced)
    from src.ml.league_integrity import verify_prediction
    from src.simulation.match_simulator import MatchSimulator, DEFAULT_N_SIMULATIONS
    league = row['league']
    matrix = pd.read_csv(ROOT/'data/features/feature_matrix.csv', low_memory=False)
    matrix = matrix.loc[matrix.league.eq(league)].copy().sort_values(['date','match_hash'])
    raw = pd.read_csv(ROOT/'data/processed/matches.csv', low_memory=False)
    raw = raw.loc[raw.league.eq(league)].copy().rename(columns={'match_date':'date','home_goals':'home_score','away_goals':'away_score'})
    matrix.attrs['ffps_data_snapshot'] = build_file_snapshot(frame=matrix,
        artifacts=[{**identity(ROOT/p), 'load_order': i, 'role': role} for i,(p,role) in enumerate([
            ('data/features/feature_matrix.csv','exact_precomputed_inference_frame'),('data/processed/matches.csv','causal_source_event_audit')])],
        source_type='phase9_league_scoped_frozen_inference_frame')
    matrix.attrs['ffps_source_events'] = source_event_index(raw)
    fixture = matrix.iloc[-1].copy()
    cutoff_columns = [c for c in matrix if 'max_source_date' in c]
    violations = [c for c in cutoff_columns if pd.notna(fixture[c]) and pd.Timestamp(fixture[c]) >= pd.Timestamp(fixture['date'])]
    if violations:
        raise ValueError('stored feature causal cutoff violation: ' + ', '.join(violations))
    timestamp = pd.Timestamp(fixture['date'], tz='UTC').isoformat()
    suite = suite if suite is not None else _load_prediction_models(league)
    simulator = MatchSimulator(league=league, seed=42, n_simulations=DEFAULT_N_SIMULATIONS,
                               seed_scope='phase9_isolated_historical_execution_probe')
    with PredictionCapture(prediction_timestamp=timestamp,
            data_provenance=data_provenance_for_fixture(matrix, fixture, timestamp)) as capture:
        probabilities = _calc_goals(fixture, suite, 'Phase9Probe', simulator, league=league, rl_weights=None, use_simulator=True)
        record = capture.finalize(fixture=fixture, league=league, probabilities=probabilities, simulator=simulator)
    replay = replay_pl_1x2(record, root=ROOT)
    record = mark_output_reproduced(record, replay)
    verified = verify_prediction(record, bundle=bundle, root=ROOT)
    if not verified['verified'] or not replay['reproduced']:
        raise ValueError(str(verified))
    return record.to_dict(), {'verification': verified, 'replay': replay,
                              'feature_source_cutoff_columns_checked': cutoff_columns,
                              'prospective_evidence': False, 'stored_feature_causality': 'timestamp_bound; historical execution diagnostic'}


def audit_league(directory, row):
    from src.ml.training_lineage import build_legacy_unproven_record, seal_record
    from src.ml.league_integrity import verify_training
    bundle = active_bundle(row)
    write(directory/'integrity_specification.json', bundle.to_dict())
    lineage = {}
    for role, item in row['active_goal_models'].items():
        if not item['available']:
            lineage[role] = {'assessment_verified': False, 'errors': ['artifact unavailable']}
            continue
        record = build_legacy_unproven_record(artifact_path=ROOT/item['path'], model_role=role,
            manifest_key=item['manifest_key'], registry_slot=item['registry_slot'], model_family=item['model_family'],
            target_name='home_goals' if role.startswith('home') else 'away_goals',
            ordered_feature_names=item['ordered_feature_names'],
            embedded_config={'configured_tree_count': item['configured_tree_count'], 'embedded_tree_count': item['embedded_tree_count']},
            embedded_randomness={}, audit_findings=['Historical exact fit rows, targets, validation, warm-start chain, source state and runtime are unproven.',
                'Tree counts are inspected from loaded artifact, not inferred from configured n_estimators.'],
            evidence=[identity(ROOT/'src/ml/models/manifest.json')], root=ROOT).to_dict()
        record['model_identity']['league'] = row['league']
        record = seal_record(record)
        write(directory/'lineage'/f'{role}.json', record)
        lineage[role] = verify_training(record, bundle=bundle, role=role, root=ROOT)
    try:
        record, proof = prediction_probe(row, bundle)
        write(directory/'prediction_provenance.json', record)
        write(directory/'execution_verification.json', proof)
        execution, blocker = 'EXECUTION_PROVENANCE_VERIFIED', None
    except Exception as exc:
        execution, blocker = 'EXECUTION_PROVENANCE_BLOCKED', f'{type(exc).__name__}: {exc}'
    audit = {'league': row['league'], 'execution': execution, 'execution_blocker': blocker,
             'active_lineage': 'LEGACY_TRAINING_LINEAGE_UNPROVEN', 'lineage_assessments': lineage,
             'candidate_required': row['league'] != 'PL',
             'data': {'status': 'CURRENT_SEASON_CONTINUITY_UNQUALIFIED', 'latest_canonical_event': row['latest_causal_data_row'],
                      'reason': 'Current-season fixture/result/statistic overlap, publication latency and team mappings not qualified by this historical audit.'},
             'pl_reference_only': row['league'] == 'PL'}
    write(directory/'audit.json', audit)
    return audit


def verify(run_dir):
    from src.evaluation.league_integrity_verifier import verify_run
    return verify_run(run_dir)


def qualify(run_dir, league=None):
    from src.evaluation.league_candidate import train_candidate, evaluate_candidate, load_suite
    inv = read(run_dir/'league_inventory.json')
    rows = [r for r in inv['leagues'] if r['production_enabled'] and (league is None or r['league'] == league)]
    if not rows:
        raise ValueError('unknown production league')
    rows.sort(key=lambda r:(r['league'] != 'BL1',r['league']))
    reports = []
    for row in rows:
        directory = run_dir/row['league']
        if row['league'] == 'PL':
            continue
        attempts = sorted(directory.glob('qualification*.json'))
        if attempts and read(attempts[-1])['candidate'] != 'BLOCKED':
            reports.append(read(attempts[-1]))
            continue
        audit = read(directory/'audit.json')
        if audit['execution'] != 'EXECUTION_PROVENANCE_VERIFIED':
            report = {'league':row['league'],'candidate':'BLOCKED','blocker':audit['execution_blocker']}
        else:
            try:
                from src.ml.league_integrity import LeagueModelBundle
                candidate_path = directory/'candidate/candidate_bundle.json'
                candidate = (LeagueModelBundle.from_dict(read(candidate_path), expected_league=row['league'])
                             if candidate_path.exists() else train_candidate(directory,row))
                record, proof = prediction_probe(row,candidate,suite=load_suite(candidate))
                write(directory/'candidate_prediction_provenance.json',record)
                write(directory/'candidate_execution_verification.json',proof)
                evaluation = evaluate_candidate(directory,row,candidate)
                report = {'league':row['league'],'candidate':'TRAINING_LINEAGE_VERIFIED','bundle_id':candidate.bundle_id,
                          'historical_evaluation':'VERIFIED','fixture_count':evaluation['fixture_count'],
                          'data_ready':False,'shadow':'FROZEN_HISTORICAL_COMPARISON; PROSPECTIVE_DATA_BLOCKED','promotion':False}
            except Exception as exc:
                report = {'league':row['league'],'candidate':'BLOCKED','blocker':f'{type(exc).__name__}: {exc}'}
        write(directory/f'qualification_{len(attempts)+1:03d}.json',report)
        reports.append(report)
    return {'reports':reports}


def prepare_matrix(run_dir):
    from src.evaluation.league_provider_qualification import qualify_provider
    from src.ml.league_integrity import derive_state, LeagueModelBundle
    from src.evaluation.league_candidate import load_suite
    inventory = read(run_dir/'league_inventory.json')
    matrix = []
    for row in inventory['leagues']:
        league = row['league']
        directory = run_dir/league
        if not row['production_enabled']:
            matrix.append({'league':league,'production':False,'state':'UNASSESSED'})
            continue
        audit = read(directory/'audit.json')
        execution = audit['execution'] == 'EXECUTION_PROVENANCE_VERIFIED'
        if not (directory/'data_continuity.json').exists():
            qualify_provider(directory,row)
        data = read(directory/'data_continuity.json')
        if league == 'PL':
            from src.ml.bundle_activation import BundleController,CANDIDATE_ID
            from scripts.run_phase7p_prospective import verify_frozen_contract
            controller = BundleController()
            candidate = controller.bundle(CANDIDATE_ID)
            frozen = verify_frozen_contract()
            resume = ROOT/'baselines/phase7er/phase7er-qualified-source-refresh-79110f9a-20260905-v2/phase7p_resume_result.json'
            resume_result = read(resume)
            reference = {'league':'PL','policy':'existing production + ongoing Phase 7P; reference-only, no reset',
                         'phase7p_contract':frozen,'candidate_bundle_id':candidate['bundle_id'],
                         'candidate_bundle_sha256':candidate['aggregate_sha256'],'latest_resume':identity(resume),
                         'latest_resume_result':resume_result,'promotion_eligible':False,
                         'cohort_predictions':[identity(p) for p in sorted((ROOT/'baselines/phase7p/phase7p-prospective-shadow-79110f9a-20260829-v1/cohort/predictions').glob('*.json'))]}
            if not (directory/'existing_experiment_reference.json').exists():
                write(directory/'existing_experiment_reference.json',reference)
            ready = frozen['verified'] and resume_result['phase7p_collection_resumed']
            matrix.append({'league':league,'production':True,'execution':audit['execution'],'active_lineage':audit['active_lineage'],
                           'candidate':'EXISTING_PHASE6_LINEAGE_VERIFIED','data_ready':ready,'shadow':'EXISTING_PHASE7P_COLLECTION_ACTIVE',
                           'promotion':False,'state':derive_state(execution=execution,active_lineage=False,candidate=True,shadow=True,data_ready=ready),
                           'outcome':'PROSPECTIVE_COLLECTION_ACTIVE','evidence_path':str(directory.relative_to(ROOT)), 'blockers':['Phase 7 scientific decision remains inconclusive.']})
            continue
        attempts = sorted(directory.glob('qualification*.json'))
        qualified = read(attempts[-1]) if attempts else {'candidate':'BLOCKED','blocker':'qualification not executed'}
        candidate_ready = qualified['candidate'] == 'TRAINING_LINEAGE_VERIFIED'
        if candidate_ready:
            candidate_path = directory/'candidate/candidate_bundle.json'
            candidate = LeagueModelBundle.from_dict(read(candidate_path),expected_league=league)
            # Final runtime captures follow all shared source changes; prior audit
            # captures remain immutable and are verified against archived sources.
            for label,bundle,suite in [('active',active_bundle(row),None),('candidate',candidate,load_suite(candidate))]:
                if not (directory/f'final_{label}_execution_verification.json').exists():
                    record,proof = prediction_probe(row,bundle,suite=suite)
                    if not (directory/f'final_{label}_prediction_provenance.json').exists():
                        write(directory/f'final_{label}_prediction_provenance.json',record)
                    write(directory/f'final_{label}_execution_verification.json',proof)
            evaluation = read(directory/'evaluation_specification.json')
            deployment = {'schema_version':'ffps.league_deployment_specification.v1','league':league,
                'active':{**identity(directory/'integrity_specification.json'),'bundle_id':active_bundle(row).bundle_id},
                'candidate':{**identity(candidate_path),'bundle_id':candidate.bundle_id},
                'evaluation_specification':{**identity(directory/'evaluation_specification.json'),'specification_sha256':evaluation['specification_sha256']},
                'smoke_reference':identity(directory/'final_active_prediction_provenance.json'),
                'data_qualification':identity(directory/'data_continuity.json'),'prospective_contract':None,
                'promotion_status':'BLOCKED_NO_LEAGUE_PROSPECTIVE_EVIDENCE'}
            deployment['specification_sha256'] = sha256_json(deployment)
            if not (directory/'deployment_specification.json').exists():
                write(directory/'deployment_specification.json',deployment)
            from src.ml.league_bundle_activation import LeagueBundleController
            controller = LeagueBundleController(directory/'deployment_specification.json',league=league)
            gate = controller.league_promotion_gate(directory)
            smoke = controller.smoke(controller.bundle(controller.candidate_id))
            if not (directory/'phase8_verification.json').exists():
                write(directory/'phase8_verification.json',{'league':league,'promotion_gate':gate,
                    'candidate_bundle_verified':True,'candidate_smoke_provenance':smoke,
                    'production_pointer_created':controller.pointer.exists(),'activation_performed':False,
                    'prospective_adapter_status':'BLOCKED_UNTIL_LEAGUE_DATA_AND_PROTOCOL_QUALIFICATION'})
        matrix.append({'league':league,'production':True,'execution':audit['execution'],'active_lineage':audit['active_lineage'],
            'candidate':'TRAINING_LINEAGE_VERIFIED' if candidate_ready else 'BLOCKED','data_ready':False,
            'shadow':'HISTORICAL_COMPARISON_FROZEN; PROSPECTIVE_DATA_BLOCKED' if candidate_ready else 'BLOCKED',
            'promotion':False,'state':derive_state(execution=execution,active_lineage=False,candidate=candidate_ready,shadow=candidate_ready,data_ready=False),
            'outcome':'CANDIDATE_READY_DATA_BLOCKED' if candidate_ready else 'TRAINING_LINEAGE_UNPROVEN',
            'evidence_path':str(directory.relative_to(ROOT)),'blockers':data['blockers']+([] if candidate_ready else [qualified['blocker']])})
    result = {'schema_version':'ffps.multi_league_integrity_matrix.v1','league_inventory_sha256':sha256_file(run_dir/'league_inventory.json'),
              'leagues':matrix,'production_leagues':inventory['production_leagues'],'production_activation_performed':False}
    if not (run_dir/'multi_league_integrity_matrix.json').exists():
        write(run_dir/'multi_league_integrity_matrix.json',result)
    return result

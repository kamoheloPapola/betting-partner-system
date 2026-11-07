"""One frozen fresh-fit protocol, independently executed for each league."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.multi_league_integrity import ROOT, ROLES, read, write, identity
from src.ml.league_integrity import LeagueModelBundle, require, verify_training
from src.ml.phase6_controlled_training import (explicit_model_configs, fit_fresh_candidate, serialize_candidate,
    compare_predictions, build_reproduced_lineage_record, seal_specification, SPEC_SCHEMA_VERSION, source_runtime_identity)
from src.ml.training_lineage import seal_record
from src.predictions.provenance import sha256_json, utc_now


def partition_frames(frame):
    """Reserve the latest two observed season cohorts; fit/calibration use predecessors.

    Northern European season starts are inferred from July boundaries on the
    actual dated fixtures in these repository leagues. Stored labels are retained
    as evidence but cannot assign a 2017 match to a 2018 fitting cohort.
    """
    dates = pd.to_datetime(frame.date)
    cohorts = dates.dt.year - (dates.dt.month < 7).astype(int)
    years = sorted(cohorts.unique())
    require(len(years) >= 5, 'insufficient dated season cohorts for five partition roles')
    validation_year, calibration_year, test_start = years[-4], years[-3], years[-2]
    parts = {'training': frame.loc[cohorts < validation_year].copy(),
             'validation': frame.loc[cohorts == validation_year].copy(),
             'calibration_fit': frame.loc[cohorts == calibration_year].copy(),
             'protected_evaluation': frame.loc[cohorts >= test_start].copy(),
             'prospective_post_protocol': frame.iloc[:0].copy()}
    require(all(len(parts[k]) for k in parts if k != 'prospective_post_protocol'), 'empty historical partition')
    ids = [set(p.match_hash) for p in parts.values()]
    require(all(not a & b for i,a in enumerate(ids) for b in ids[i+1:]), 'partition identity overlap')
    require(sum(map(len, ids)) == len(frame), 'duplicate or omitted fixture identity')
    return parts


def train_candidate(directory, row):
    require(row['league'] != 'PL', 'PL experiment is reference-only in Phase 9')
    require(not (directory/'candidate').exists(), 'candidate attempt already exists and is immutable')
    output = directory/'candidate'
    names = row['feature_schema']['ordered_feature_names']
    frame = pd.read_csv(ROOT/'data/features/feature_matrix.csv', low_memory=False)
    frame = frame.loc[frame.league.eq(row['league'])].sort_values(['date','match_hash']).copy()
    eligible = np.isfinite(frame[names].to_numpy(float)).all(axis=1)
    excluded = frame.loc[~eligible, ['match_hash','league','date']].to_dict('records')
    frame = frame.loc[eligible].copy()
    require(frame.match_hash.is_unique, 'duplicate league fixture identities')
    for c in (c for c in frame if 'max_source_date' in c):
        source_dates = pd.to_datetime(frame[c], errors='coerce')
        require(not (source_dates >= pd.to_datetime(frame.date)).any(), 'stored feature cutoff violation: '+c)
    parts = partition_frames(frame)
    output.mkdir(parents=True)
    write(output/'excluded_cold_start_rows.json', {'rule':'all ordered model-facing features must be finite before partition allocation',
          'outcomes_considered':False,'rows':excluded,'count':len(excluded)})
    columns = ['match_hash','league','date','season','home_team','away_team','home_score','away_score'] + names
    frame[columns].to_csv(output/'source_features.csv', index=False)
    # Fit the round-tripped frozen bytes so every stored float is the actual fit input.
    frame = pd.read_csv(output/'source_features.csv', float_precision='round_trip')
    parts = partition_frames(frame)
    partition_spec = {}
    for name, part in parts.items():
        part.to_csv(output/f'{name}.csv', index=False)
        partition_spec[name] = {'row_count': len(part), 'fixture_ids': part.match_hash.tolist(),
                                'start': str(part.date.min()) if len(part) else None,
                                'end': str(part.date.max()) if len(part) else None,
                                'artifact': identity(output/f'{name}.csv')}
    sources = []
    for path in [Path(__file__), ROOT/'src/ml/phase6_controlled_training.py', ROOT/'src/ml/training_lineage.py',
                 ROOT/'src/features/build_feature_matrix.py', ROOT/'src/features/engineering.py']:
        target = output/'source_snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources.append(target)
    runtime = source_runtime_identity(root=ROOT, source_files=sources)
    import subprocess
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    spec = seal_specification({'schema_version': SPEC_SCHEMA_VERSION, 'league': row['league'], 'created_at': utc_now(),
        'model_roles': list(ROLES), 'model_configs': explicit_model_configs(), 'feature_contract': row['feature_schema'],
        'partitions': partition_spec, 'partition_policy': 'date-derived July season cohorts; last two observed cohorts protected; preceding one calibration; preceding one validation; earlier training',
        'stored_season_labels_used_for_partitioning': False, 'pairwise_overlap': 0,
        'eligibility_exclusions': identity(output/'excluded_cold_start_rows.json'),
        'source_dataset': identity(output/'source_features.csv'), 'upstream_source': identity(ROOT/'data/features/feature_matrix.csv'),
        'runtime': runtime, 'git_revision': revision, 'source_snapshot_is_authoritative_for_dirty_workspace': True,
        'randomness': {'seed':42,'n_jobs':1,'prediction_atol':1e-12,'prediction_rtol':1e-12},
        'training_mode':'fresh_estimator_from_zero_state','evaluation_only':True,'promotion_authorized':False})
    write(output/'training_specification.json', spec)
    fit, val = parts['training'], parts['validation']
    members = {}
    for role, model_name in ROLES.items():
        target = 'home_score' if role.startswith('home') else 'away_score'
        config = spec['model_configs'][role]
        executions = {}
        for execution in ('primary','repeat'):
            model, predictions, proof = fit_fresh_candidate(role=role, config=config,
                fit_features=fit[names], fit_target=fit[target], validation_features=val[names])
            artifact = serialize_candidate(model, output/execution/f'{role}.joblib', run_dir=output)
            write(output/execution/f'{role}_predictions.json', predictions.tolist())
            executions[execution] = (model, predictions, proof, artifact)
        first, second = executions['primary'], executions['repeat']
        reproduction = compare_predictions(first[1], second[1], atol=1e-12, rtol=1e-12)
        require(reproduction['reproduced'], 'repeat fit prediction mismatch')
        reproduction.update(artifact_bytes_identical=first[3]['sha256'] == second[3]['sha256'],
                            primary_artifact=first[3], repeat_artifact=second[3])
        core = build_reproduced_lineage_record(artifact_path=ROOT/first[3]['path'], role=role, execution='primary',
            fit_features=fit[names], fit_target=fit[target], fit_row_ids=fit.match_hash.tolist(), fit_event_times=fit.date.tolist(),
            validation_features=val[names], validation_target=val[target], validation_row_ids=val.match_hash.tolist(), validation_event_times=val.date.tolist(),
            config=config, randomness=spec['randomness'], warm_start_proof=first[2], source_dataset_path=output/'source_features.csv',
            source_files=sources, training_specification_path=output/'training_specification.json', reproduction=reproduction,
            evidence=[identity(output/'training_specification.json')], root=ROOT).to_dict()
        core['model_identity']['league'] = row['league']
        core = seal_record(core)
        write(output/'lineage'/f'{role}.json', core)
        # Diagnostic validation score satisfies the existing runtime presence guard;
        # it is neither a fitted calibrator nor a promotion gate.
        score = float(abs(np.mean(1-np.exp(-np.maximum(first[1], 0))) - np.mean(val[target].to_numpy()>0)))
        members[role] = {**first[3], 'league': row['league'], 'schema_sha256': sha256_json(names),
                         'lineage': identity(output/'lineage'/f'{role}.json'), 'repeat_fit': reproduction,
                         'meta': {'name': model_name,'league':row['league'],'features':names,'version':'phase9-evaluation-only',
                                  'type':'lgbm_regressor_poisson' if role.endswith('lgbm') else 'xgb_regressor_count_poisson',
                                  'metrics':{'calibration_score':score},'calibration_score_semantics':'validation mean P(Y>0) gap; diagnostic guard metadata only',
                                  'sklearn_version':runtime['runtime']['dependencies']['scikit-learn']}}
    value = {'league':row['league'],'bundle_id':'phase9-candidate-'+row['league']+'-'+sha256_json(members)[:16],
             'members':members,'feature_contract':row['feature_schema'],'training_specification':identity(output/'training_specification.json'),
             'environment_config_identity':runtime,'execution_compatibility':'existing_four_goal_ensemble','activation_status':'EVALUATION_ONLY'}
    value['bundle_sha256'] = sha256_json(value)
    bundle = LeagueModelBundle.from_dict(value, expected_league=row['league'])
    for role, member in members.items():
        result = verify_training(read(ROOT/member['lineage']['path']), bundle=bundle, role=role, root=ROOT)
        require(result['lineage_verified'], str(result))
    write(output/'candidate_bundle.json', value)
    return bundle


def load_suite(bundle):
    from src.ml.bundle_activation import SLOTS
    from src.predictions.provenance import sha256_file
    value = bundle.to_dict()
    suite = {}
    for role, member in value['members'].items():
        path = ROOT/member['path']
        require(sha256_file(path) == member['sha256'], 'loaded candidate artifact mismatch')
        model = joblib.load(path)
        model.meta = member['meta']
        model._ffps_provenance = {'artifact_sha256':member['sha256'],'source_location':member['path'],
            'size_bytes':member['size_bytes'],'manifest_key':value['bundle_id']+':'+role,
            'registry_slot':'EVALUATION_ONLY:'+value['league']+':'+role,
            'training_lineage':{'classification':'proven','lineage_path':member['lineage']['path'],
                                'lineage_fingerprint':read(ROOT/member['lineage']['path'])['training_lineage_fingerprint']}}
        suite[SLOTS[role]] = model
    suite['meta_goals'] = suite['mh_goals'].meta
    suite['meta_goals_xgb'] = suite['mh_goals_xgb'].meta
    return suite


def evaluate_candidate(directory, row, candidate):
    from src.evaluation.phase7_shadow import outcome_indices, summarize_probability_metrics, summarize_goal_metrics
    from src.cli.commands.prediction import _load_prediction_models
    from src.ml.bundle_activation import SLOTS
    from src.simulation.match_simulator import MatchSimulator, DEFAULT_N_SIMULATIONS, clamp_lambda
    from src.evaluation.multi_league_integrity import prediction_probe, active_bundle
    frame = pd.read_csv(directory/'candidate/protected_evaluation.csv', float_precision='round_trip')
    names = row['feature_schema']['ordered_feature_names']
    spec = {'schema_version':'ffps.league_shadow_evaluation.v1','league':row['league'],
            'active_bundle_sha256':active_bundle(row).to_dict()['bundle_sha256'],
            'candidate_bundle_id':candidate.bundle_id,'candidate_bundle_sha256':candidate.to_dict()['bundle_sha256'],
            'feature_schema_sha256':sha256_json(names),'fixture_ids':frame.match_hash.tolist(),
            'cohort':'HISTORICAL_DIAGNOSTIC_NOT_PRISTINE','common_rng':{'protocol':'identical seed per paired fixture','seed_derivation':'first 8 SHA256 hex of league + fixture ID','n_simulations':DEFAULT_N_SIMULATIONS},
            'primary_metrics':['multiclass_log_loss','multiclass_brier','goal_rmse','poisson_deviance'],
            'reliability_bins':10,'promotion_authorized':False,'frozen_at':utc_now()}
    spec['specification_sha256'] = sha256_json(spec)
    write(directory/'evaluation_specification.json', spec)
    suites = {'active':_load_prediction_models(row['league']), 'candidate':load_suite(candidate)}
    predictions = {}
    for label,suite in suites.items():
        outputs = {role:np.asarray(suite[SLOTS[role]].predict(frame[names]),float) for role in ROLES}
        lambdas = np.array([[clamp_lambda(.6*clamp_lambda(outputs[side+'_goals_lgbm'][i]) + .4*clamp_lambda(outputs[side+'_goals_xgb'][i]))
                             for side in ('home','away')] for i in range(len(frame))])
        probabilities = []
        for i, fixture_id in enumerate(frame.match_hash):
            seed = int(sha256_json([row['league'],fixture_id])[:8],16)
            simulation = MatchSimulator(league=row['league'],seed=seed,n_simulations=DEFAULT_N_SIMULATIONS,
                                        seed_scope='phase9_historical_common_rng').simulate(*lambdas[i],rl_weights=None)
            probabilities.append([simulation.home_win_prob,simulation.draw_prob,simulation.away_win_prob])
        predictions[label] = {'probabilities':probabilities,'goal_lambdas':lambdas.tolist()}
    actual = frame[['home_score','away_score']].to_numpy(float)
    labels = outcome_indices(actual[:,0],actual[:,1])
    metrics = {k:{'1x2':summarize_probability_metrics(labels,np.array(v['probabilities'])),
                  'goals':summarize_goal_metrics(actual.reshape(-1),np.array(v['goal_lambdas']).reshape(-1))} for k,v in predictions.items()}
    write(directory/'historical_pairs.json', {'league':row['league'],'specification_sha256':spec['specification_sha256'],
                                             'fixture_ids':frame.match_hash.tolist(),'actual_goals':actual.tolist(),'predictions':predictions})
    result = {'league':row['league'],'fixture_count':len(frame),'metrics':metrics,'classification':'HISTORICAL_DIAGNOSTIC_ONLY',
              'common_rng_verified':True,'same_feature_matrix_sha256':sha256_json(frame[names].values.tolist()),'promotion_authorized':False}
    write(directory/'historical_evaluation.json', result)
    return result

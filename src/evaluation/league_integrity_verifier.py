"""Offline evidence verification; every configured production league is accounted for."""
from __future__ import annotations

from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd

from src.evaluation.multi_league_integrity import ROOT, ROLES, read, identity, active_bundle
from src.ml.league_integrity import LeagueModelBundle, require, verify_prediction, verify_training, derive_state
from src.predictions.provenance import sha256_file, sha256_json


def checked_reference(ref, *, within=ROOT):
    path = (ROOT/ref['path']).resolve()
    require(path.is_relative_to(within.resolve()), 'reference escapes evidence scope')
    require(path.is_file() and sha256_file(path) == ref['sha256'], 'reference hash mismatch: '+str(path))
    if 'size_bytes' in ref:
        require(path.stat().st_size == ref['size_bytes'], 'reference size mismatch')
    return path


def verify_candidate(directory, row):
    from src.ml.phase6_controlled_training import Phase6TrainingSpecification, compare_predictions
    from src.ml.training_lineage import build_partition_record
    output = directory/'candidate'
    bundle = LeagueModelBundle.from_dict(read(output/'candidate_bundle.json'),expected_league=row['league'])
    value = bundle.to_dict()
    spec = Phase6TrainingSpecification.load(checked_reference(value['training_specification'],within=output)).to_dict()
    require(spec['league'] == row['league'], 'training specification league mismatch')
    require(spec['feature_contract']['ordered_feature_names'] == row['feature_schema']['ordered_feature_names'], 'training feature schema mismatch')
    frame = pd.read_csv(checked_reference(spec['source_dataset'],within=output),float_precision='round_trip')
    require(set(frame.league) == {row['league']}, 'training data crosses leagues')
    require(frame.match_hash.is_unique, 'duplicate training fixture identity')
    names = row['feature_schema']['ordered_feature_names']
    require(np.isfinite(frame[names].to_numpy(float)).all(), 'nonfinite frozen training features')
    parts, seen = {}, set()
    for name, partition in spec['partitions'].items():
        part = pd.read_csv(checked_reference(partition['artifact'],within=output),float_precision='round_trip')
        require(part.match_hash.tolist() == partition['fixture_ids'], 'partition row identity mismatch')
        require(len(part) == partition['row_count'], 'partition count mismatch')
        require(set(part.league) <= {row['league']}, 'partition league mismatch')
        require(not seen.intersection(part.match_hash), 'partition overlap')
        seen.update(part.match_hash)
        expected = frame.loc[frame.match_hash.isin(part.match_hash)].reset_index(drop=True)
        pd.testing.assert_frame_equal(part.reset_index(drop=True),expected,check_dtype=False)
        parts[name] = part
    require(seen == set(frame.match_hash), 'partition omissions')
    chronological = [parts[n] for n in ('training','validation','calibration_fit','protected_evaluation')]
    require(all(a.date.max() < b.date.min() for a,b in zip(chronological,chronological[1:])), 'temporal partition boundaries overlap')
    fit,val = parts['training'],parts['validation']
    checks = {}
    for role,member in value['members'].items():
        artifact = checked_reference(member,within=output/'primary')
        core = read(checked_reference(member['lineage'],within=output/'lineage'))
        result = verify_training(core,bundle=bundle,role=role,root=ROOT)
        require(result['lineage_verified'],str(result))
        target = 'home_score' if role.startswith('home') else 'away_score'
        for part, key, partition_role, policy in (
            (fit,'fit_partition','model_fit','reject_all_non_finite_values_before_fit'),
            (val,'validation_partition','frozen_validation_reproduction_corpus','reject_all_non_finite_values_before_prediction')):
            expected = build_partition_record(role=partition_role,features=part[names],targets=part[target],row_ids=part.match_hash.tolist(),
                event_times=part.date.tolist(),target_name='home_goals' if role.startswith('home') else 'away_goals',missing_value_policy=policy)
            require(core['training_data'][key] == expected, 'lineage exact fit matrix/target/row binding mismatch')
        require(core['fit_contract']['effective_config'] == spec['model_configs'][role], 'fit config mismatch')
        require(core['fit_contract']['training_specification_sha256'] == spec['specification_sha256'], 'lineage/specification mismatch')
        repeat = member['repeat_fit']
        require(checked_reference(repeat['primary_artifact'],within=output/'primary') == artifact, 'repeat primary join mismatch')
        second_path = checked_reference(repeat['repeat_artifact'],within=output/'repeat')
        first,second = joblib.load(artifact),joblib.load(second_path)
        for model in (first,second):
            trees = model.booster_.num_trees() if role.endswith('lgbm') else model.get_booster().num_boosted_rounds()
            require(trees == spec['model_configs'][role]['n_estimators'], 'fresh candidate tree count mismatch')
        left,right = first.predict(val[names]),second.predict(val[names])
        require(np.array_equal(left,read(output/'primary'/f'{role}_predictions.json')), 'stored primary predictions differ')
        require(np.array_equal(right,read(output/'repeat'/f'{role}_predictions.json')), 'stored repeat predictions differ')
        check = compare_predictions(left,right,atol=1e-12,rtol=1e-12)
        require(check['reproduced'] and all(repeat[k] == v for k,v in check.items()), 'repeat prediction reproduction mismatch')
        require(repeat['artifact_bytes_identical'] == (member['sha256'] == repeat['repeat_artifact']['sha256']), 'repeat artifact equality mismatch')
        checks[role] = {'lineage_verified':True,'repeat_predictions_reproduced':True,'artifact_bytes_identical':repeat['artifact_bytes_identical']}
    return bundle, checks


def verify_evaluation(directory, row, candidate, *, replay=True):
    from src.evaluation.phase7_shadow import outcome_indices,summarize_probability_metrics,summarize_goal_metrics
    from src.simulation.match_simulator import MatchSimulator,clamp_lambda
    spec = read(directory/'evaluation_specification.json')
    require(spec['specification_sha256'] == sha256_json({k:v for k,v in spec.items() if k != 'specification_sha256'}), 'evaluation fingerprint mismatch')
    require(spec['league'] == row['league'] and spec['candidate_bundle_id'] == candidate.bundle_id, 'evaluation league/candidate mismatch')
    require(spec['active_bundle_sha256'] == active_bundle(row).to_dict()['bundle_sha256'], 'evaluation active identity mismatch')
    require(spec['candidate_bundle_sha256'] == candidate.to_dict()['bundle_sha256'], 'evaluation candidate identity mismatch')
    pairs = read(directory/'historical_pairs.json')
    frame = pd.read_csv(directory/'candidate/protected_evaluation.csv',float_precision='round_trip')
    require(pairs['fixture_ids'] == spec['fixture_ids'] == frame.match_hash.tolist(), 'historical cohort selection mismatch')
    require(pairs['league'] == row['league'] and pairs['specification_sha256'] == spec['specification_sha256'], 'paired evidence scope mismatch')
    actual = frame[['home_score','away_score']].to_numpy(float)
    require(pairs['actual_goals'] == actual.tolist(), 'historical outcome mismatch')
    labels = outcome_indices(actual[:,0],actual[:,1])
    report = read(directory/'historical_evaluation.json')
    names = row['feature_schema']['ordered_feature_names']
    require(report['same_feature_matrix_sha256'] == sha256_json(frame[names].values.tolist()), 'paired feature identity mismatch')
    metrics = {}
    for label,bundle in [('active',active_bundle(row)),('candidate',candidate)]:
        saved = pairs['predictions'][label]
        probabilities,lambdas = np.array(saved['probabilities']),np.array(saved['goal_lambdas'])
        require(np.isfinite(probabilities).all() and (probabilities>=0).all() and np.allclose(probabilities.sum(axis=1),1), 'invalid paired probabilities')
        if replay:
            outputs = {}
            for role,member in bundle.to_dict()['members'].items():
                path = checked_reference(member)
                model = joblib.load(path) if path.suffix == '.joblib' else pickle.loads(path.read_bytes())
                outputs[role] = np.asarray(model.predict(frame[names]),float)
            expected = np.array([[clamp_lambda(.6*clamp_lambda(outputs[side+'_goals_lgbm'][i])+.4*clamp_lambda(outputs[side+'_goals_xgb'][i]))
                                   for side in ('home','away')] for i in range(len(frame))])
            require(np.array_equal(expected,lambdas), 'paired model/lambda replay mismatch')
            for i,fixture in enumerate(frame.match_hash):
                simulation = MatchSimulator(league=row['league'],seed=int(sha256_json([row['league'],fixture])[:8],16),
                    n_simulations=spec['common_rng']['n_simulations'],seed_scope='phase9_historical_common_rng').simulate(*expected[i],rl_weights=None)
                require([simulation.home_win_prob,simulation.draw_prob,simulation.away_win_prob] == saved['probabilities'][i], 'paired simulator replay mismatch')
        metrics[label] = {'1x2':summarize_probability_metrics(labels,probabilities),
                          'goals':summarize_goal_metrics(actual.reshape(-1),lambdas.reshape(-1))}
    require(metrics == report['metrics'] and report['fixture_count'] == len(frame), 'historical metrics mismatch')
    require(report['promotion_authorized'] is False and spec['promotion_authorized'] is False, 'historical evidence grants promotion')
    return {'fixture_count':len(frame),'replayed':replay,'metrics_verified':True}


def verify_provider(directory,row):
    from src.data_pipeline.build_matches_dataset import CanonicalDatasetBuilder
    data = read(directory/'data_continuity.json')
    require(data['league'] == row['league'], 'provider qualification league mismatch')
    overlap = read(checked_reference(data['overlap_evidence'],within=directory))
    require(overlap['league'] == row['league'], 'overlap league mismatch')
    normalized = {}
    builder = CanonicalDatasetBuilder()
    for source in overlap['sources']:
        path = checked_reference(source)
        parsed,_ = builder._parse_and_normalise(path,row['league'])
        require(parsed is not None,'qualified provider source no longer parses')
        parsed['match_date'] = parsed.match_date.dt.strftime('%Y-%m-%d')
        normalized[source['sha256']] = parsed
    canonical = pd.read_csv(ROOT/'data/processed/matches.csv').set_index('match_hash')
    for authority in data['field_authorities']:
        require(authority['league'] == row['league'], 'provider field authority crosses league')
        field = authority['fields'][0]
        observations = overlap['observations']
        require(all(o['league'] == row['league'] and o['fixture'][0] == row['league'] for o in observations), 'overlap rows cross league')
        require(authority['overlap_rows'] == len(observations), 'overlap count mismatch')
        mismatches = 0
        for observation in observations:
            values = observation['fields'][field]
            source_row = normalized[values['raw_source_sha256']].loc[values['normalized_row_index']]
            require([str(source_row[k]) for k in ('league','match_date','home_team','away_team')] == observation['fixture'], 'provider row/orientation join mismatch')
            frozen_row = canonical.loc[observation['match_hash']]
            require([str(frozen_row[k]) for k in ('league','match_date','home_team','away_team')] == observation['fixture'], 'canonical provider join mismatch')
            require((None if pd.isna(source_row[field]) else float(source_row[field])) == values['source_value'],'provider raw statistic binding mismatch')
            require((None if pd.isna(frozen_row[field]) else float(frozen_row[field])) == values['canonical_value'],'provider canonical statistic binding mismatch')
            a,b = values['source_value'],values['canonical_value']
            equal = a is not None and b is not None and a == b and float(a).is_integer() and a >= 0
            require(values['exact_equal'] == equal, 'field equality mismatch')
            mismatches += not equal
        require(mismatches == authority['mismatches'], 'field mismatch count differs')
    require(data['feature_readiness'] is False, 'unqualified current source declared ready')
    return {'historical_overlap_verified':True,'current_data_ready':False}


def verify_run(run_dir, *, require_index=True, replay=True):
    from src.config.leagues import ACTIVE_LEAGUES
    from src.ml.bundle_evidence import verify_historical_record,historical_source
    errors,results = [],{}
    inv = read(run_dir/'league_inventory.json')
    rows = {r['league']:r for r in inv['leagues']}
    matrix = read(run_dir/'multi_league_integrity_matrix.json')
    states = {r['league']:r for r in matrix['leagues']}
    if len(rows) != len(inv['leagues']) or len(states) != len(matrix['leagues']):
        errors.append('duplicate league records')
    expected = set(ACTIVE_LEAGUES)
    if set(inv['production_leagues']) != expected or set(matrix['production_leagues']) != expected:
        errors.append('production league inventory mismatch')
    if {r['league'] for r in inv['leagues'] if r['production_enabled']} != expected:
        errors.append('inventory enabled flags mismatch')
    if {r['league'] for r in matrix['leagues'] if r['production']} != expected:
        errors.append('matrix omits a production league')
    if matrix['league_inventory_sha256'] != sha256_file(run_dir/'league_inventory.json'):
        errors.append('inventory/matrix hash mismatch')
    for league in sorted(expected):
        try:
            row,state = rows[league],states[league]
            directory = run_dir/league
            bundle = LeagueModelBundle.from_dict(read(directory/'integrity_specification.json'),expected_league=league)
            require(bundle.to_dict() == active_bundle(row).to_dict(), 'active specification differs from inventory')
            require(sha256_json(row['feature_schema']['ordered_feature_names']) == row['feature_schema']['schema_sha256'], 'feature contract mismatch')
            feature_evidence = read(directory/'feature_contract_evidence.json')
            require(feature_evidence['league'] == league and feature_evidence['ordered_feature_names'] == row['feature_schema']['ordered_feature_names'],'supplemental feature contract mismatch')
            registry_evidence = read(directory/'all_active_registry_artifacts.json')
            require(registry_evidence['league'] == league,'registry evidence league mismatch')
            for item in registry_evidence['artifacts'].values():
                require(item['available'],'active registry artifact unavailable')
                checked_reference(item['artifact'])
            for role,member in row['active_goal_models'].items():
                checked_reference(member)
                lineage = verify_training(read(directory/'lineage'/f'{role}.json'),bundle=bundle,role=role,root=ROOT)
                require(lineage['assessment_verified'] and not lineage['lineage_verified'], 'legacy lineage classification mismatch')
            for path in directory.glob('*prediction_provenance.json'):
                record = read(path)
                owner = bundle if 'candidate' not in path.name else LeagueModelBundle.from_dict(read(directory/'candidate/candidate_bundle.json'),expected_league=league)
                scope = verify_prediction(record,bundle=owner,root=ROOT,verify_referenced_artifacts=False)
                require(scope['verified'],str(scope))
                verify_historical_record(record,root=ROOT,replay=replay)
            provider = verify_provider(directory,row)
            if league == 'PL':
                from scripts.run_phase7p_prospective import verify_frozen_contract
                from src.ml.bundle_activation import BundleController,CANDIDATE_ID
                require(verify_frozen_contract()['verified'],'PL Phase 7P frozen contract failed')
                candidate = BundleController().bundle(CANDIDATE_ID)
                reference = read(directory/'existing_experiment_reference.json')
                require(reference['candidate_bundle_id'] == candidate['bundle_id'],'PL candidate reference drift')
                for ref in [reference['latest_resume'],*reference['cohort_predictions']]:
                    checked_reference(ref)
                require(read(ROOT/reference['latest_resume']['path']) == reference['latest_resume_result'],'PL resume evidence payload mismatch')
                data_ready = reference['latest_resume_result']['phase7p_collection_resumed']
                candidate_ready = True
                evaluation = {'existing_experiment_preserved':True}
            else:
                candidate,training = verify_candidate(directory,row)
                evaluation = verify_evaluation(directory,row,candidate,replay=replay)
                from src.ml.league_bundle_activation import LeagueBundleController
                controller = LeagueBundleController(directory/'deployment_specification.json',league=league)
                gate = controller.league_promotion_gate(directory)
                require(not gate['promotion_eligible'],'unqualified league promoted')
                require(not controller.pointer.exists() and not controller.pending.exists(),'Phase 9 changed a production pointer')
                data_ready,candidate_ready = False,True
            derived = derive_state(execution=True,active_lineage=False,candidate=candidate_ready,shadow=True,data_ready=data_ready)
            require(state['state'] == derived and state['data_ready'] == data_ready and state['promotion'] is False,'matrix mechanical state mismatch')
            require(state['execution'] == 'EXECUTION_PROVENANCE_VERIFIED' and state['active_lineage'] == 'LEGACY_TRAINING_LINEAGE_UNPROVEN','matrix audit classification mismatch')
            results[league] = {'verified':True,'state':derived,'provider':provider,'evaluation':evaluation}
        except Exception as exc:
            result = f'{type(exc).__name__}: {exc}'
            results[league] = {'verified':False,'error':result}
            errors.append(league+': '+result)
    for path,digest in read(run_dir/'protected_before.json').items():
        if not (ROOT/path).is_file() or sha256_file(ROOT/path) != digest:
            errors.append('protected artifact changed: '+path)
    if require_index:
        try:
            index = read(run_dir/'artifact_index.json')
            require(index['aggregate_sha256'] == sha256_json(index['artifacts']), 'artifact index aggregate mismatch')
            for ref in index['artifacts']:
                checked_reference(ref,within=run_dir)
            for ref in index['sources']:
                checked_reference(ref)
            from src.ml.bundle_evidence import junit
            tests = read(run_dir/'test_results.json')
            for name in ('focused','protected','full','before'):
                require(junit(run_dir/'tests'/f'{name}.xml') == tests[name],'JUnit evidence mismatch')
            require(tests['focused']['failures'] == tests['protected']['failures'] == 0,'required test failures')
            require(not (set(tests['full']['failure_ids'])-set(tests['before']['failure_ids'])),'new unexplained full-suite failures')
        except Exception as exc:
            errors.append('index: '+str(exc))
    return {'schema_version':'ffps.phase9_offline_verification.v1','verified':not errors,'errors':errors,
            'production_leagues':sorted(expected),'league_results':results,'full_historical_replay':replay}

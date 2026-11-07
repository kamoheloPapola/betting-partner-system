"""League-isolated append-only shadow evidence using Phase 7's frozen decision rule.

No protocol is instantiated by Phase 9 for a league with unqualified current data.
When a league qualifies, its immutable deployment specification must bind this
protocol, its source qualification and its independently chosen fixed horizon.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.ml.bundle_activation import require, read_json, persist_new, sha256_file, sha256_json


def timestamp(value):
    result = datetime.fromisoformat(str(value).replace('Z','+00:00'))
    require(result.tzinfo is not None,'prospective timestamps require timezone')
    return result.astimezone(timezone.utc)


def verify_protocol(controller):
    from src.ml.league_integrity import verify_field_authority
    frozen = controller.specification()
    reference = frozen.get('prospective_contract')
    require(reference is not None,'league prospective protocol is not frozen')
    path = (controller.root/reference['path']).resolve()
    require(path.is_relative_to(controller.root/'baselines/phase9'),'prospective protocol escapes evidence root')
    require(sha256_file(path) == reference['sha256'],'prospective protocol reference drift')
    protocol = read_json(path)
    require(protocol['schema_version'] == 'ffps.league_prospective_protocol.v1','unknown prospective protocol schema')
    require(protocol['protocol_sha256'] == sha256_json({k:v for k,v in protocol.items() if k != 'protocol_sha256'}),'prospective protocol fingerprint mismatch')
    require(protocol['league'] == controller.league,'prospective protocol league mismatch')
    require(protocol['candidate_bundle_id'] == controller.candidate_id,'prospective protocol candidate mismatch')
    require(protocol['active_bundle_id'] == controller.legacy_id,'prospective protocol control mismatch')
    require(protocol['evaluation_specification_sha256'] == frozen['evaluation_specification']['specification_sha256'],'prospective evaluation mismatch')
    require(timestamp(protocol['frozen_at']) < timestamp(protocol['start_at']) < timestamp(protocol['end_at']),'prospective horizon must follow freeze')
    require(protocol['decision_implementation_sha256'] == sha256_file(controller.root/'src/evaluation/phase7_shadow.py'),'prospective decision implementation changed')
    qualification_ref = protocol['data_qualification']
    qualification_path = (controller.root/qualification_ref['path']).resolve()
    require(qualification_path.is_relative_to(path.parent),'qualification escapes league protocol directory')
    require(sha256_file(qualification_path) == qualification_ref['sha256'],'prospective source qualification drift')
    qualification = read_json(qualification_path)
    require(qualification['league'] == controller.league and qualification['feature_readiness'] is True,'league feature continuity unqualified')
    for field in ('home_goals','away_goals','home_corners','away_corners'):
        authorities = [a for a in qualification['field_authorities'] if field in a['fields']]
        require(len(authorities) == 1,'ambiguous prospective field authority')
        a = authorities[0]
        verify_field_authority(a,expected_league=controller.league,source=a['source'],field=field)
    require(protocol['bootstrap']['resamples'] >= 1000 and protocol['bootstrap']['confidence_level'] == .95,'invalid frozen paired uncertainty protocol')
    cohort = (controller.root/protocol['cohort_path']).resolve()
    require(cohort.is_relative_to(path.parent) and cohort != path.parent,'prospective cohort escapes protocol directory')
    return protocol,cohort


def validate_pair(pair, *, controller, protocol):
    from src.ml.bundle_evidence import verify_historical_record
    require(pair['league'] == controller.league and pair['protocol_sha256'] == protocol['protocol_sha256'],'prospective pair scope mismatch')
    require(pair['pair_sha256'] == sha256_json({k:v for k,v in pair.items() if k != 'pair_sha256'}),'prospective pair fingerprint mismatch')
    created,kickoff = timestamp(pair['created_at']),timestamp(pair['kickoff_at'])
    require(timestamp(protocol['frozen_at']) < created < kickoff,'retrospective reconstruction forbidden')
    require(timestamp(protocol['start_at']) <= kickoff < timestamp(protocol['end_at']),'fixture outside fixed horizon')
    records = [pair['active'],pair['candidate']]
    for record,bundle_id in zip(records,(controller.legacy_id,controller.candidate_id)):
        require(record['fixture']['league'] == controller.league and record['fixture']['match_hash'] == pair['fixture_id'],'prospective fixture identity mismatch')
        require(timestamp(record['fixture']['kickoff_identity']) == kickoff,'prospective kickoff identity mismatch')
        require(abs((timestamp(record['fixture']['prediction_timestamp'])-created).total_seconds()) <= 60,'prospective prediction timestamp mismatch')
        require(timestamp(record['data_provenance']['causal_cutoff']) <= created,'future feature cutoff')
        bundle = controller.bundle(bundle_id)
        require({m['role']:m['artifact_sha256'] for m in record['model_provenance']['members']} == {r:m['sha256'] for r,m in bundle['members'].items()},'prospective loaded bundle mismatch')
        require(all(c['ordered_feature_names'] == bundle['features'] for c in record['feature_provenance']['model_inputs']),'prospective feature schema mismatch')
        verify_historical_record(record,root=controller.root,replay=True)
    a,b = records
    require(a['data_provenance'] == b['data_provenance'],'prospective paired causal states differ')
    require([(v['role'],v['feature_vector_sha256']) for v in a['feature_provenance']['model_inputs']] == [(v['role'],v['feature_vector_sha256']) for v in b['feature_provenance']['model_inputs']],'prospective paired features differ')
    left,right = (r['method_provenance']['descriptor']['simulation'] for r in records)
    common = ('implementation','league','n_simulations','max_goals','top_scorelines','tempo_sigma',
              'seed','seed_scope','bit_generator','rng_state_before','rl_weights')
    require(all(left.get(k) == right.get(k) for k in common),'prospective common RNG differs')


def append_prediction_pair(*, controller, active, candidate, kickoff_at):
    protocol,cohort = verify_protocol(controller)
    created = datetime.now(timezone.utc).isoformat()
    pair = {'league':controller.league,'protocol_sha256':protocol['protocol_sha256'],
            'fixture_id':active['fixture']['match_hash'],'created_at':created,'kickoff_at':kickoff_at,
            'active':active,'candidate':candidate}
    pair['pair_sha256'] = sha256_json(pair)
    validate_pair(pair,controller=controller,protocol=protocol)
    require(datetime.now(timezone.utc) < timestamp(kickoff_at),'kickoff occurred during prospective verification')
    name = sha256_json([controller.league,pair['fixture_id']])
    persist_new(cohort/'predictions'/f'{name}.json',pair)
    return pair


def append_settlement(*,controller,fixture_id,home_goals,away_goals,result_source):
    protocol,cohort = verify_protocol(controller)
    name = sha256_json([controller.league,fixture_id])
    pair = read_json(cohort/'predictions'/f'{name}.json')
    require(datetime.now(timezone.utc) > timestamp(pair['kickoff_at']),'fixture has not kicked off')
    source = controller.root/result_source['path']
    require(source.resolve().is_relative_to(controller.root.resolve()),'settlement source escapes repository')
    require(sha256_file(source) == result_source['sha256'],'settlement source hash mismatch')
    event = read_json(source)
    require(event['league'] == controller.league and event['fixture_id'] == fixture_id and event['status'] == 'FINISHED','settlement result identity mismatch')
    require(event['home_goals'] == home_goals and event['away_goals'] == away_goals,'settlement scores differ from source')
    require(all(type(x) is int and x >= 0 for x in (home_goals,away_goals)),'invalid settlement goals')
    created = datetime.now(timezone.utc)
    require(timestamp(pair['kickoff_at']) < timestamp(event['observed_at']) <= created,'settlement publication timestamp invalid')
    record = {'league':controller.league,'fixture_id':fixture_id,'protocol_sha256':protocol['protocol_sha256'],
              'prediction_pair_sha256':pair['pair_sha256'],'created_at':created.isoformat(),
              'home_goals':home_goals,'away_goals':away_goals,'source':result_source}
    record['settlement_sha256'] = sha256_json(record)
    persist_new(cohort/'settlements'/f'{name}.json',record)
    return record


def promotion_gate(controller,evidence):
    """Recompute the full authoritative cohort and Phase 7 zero-margin decision."""
    from src.evaluation.phase7_shadow import (outcome_indices,probability_metric_contributions,
        goal_metric_contributions,summarize_probability_metrics,paired_bootstrap,apply_decision_rule)
    from src.simulation.match_simulator import clamp_lambda
    protocol,cohort = verify_protocol(controller)
    require(datetime.now(timezone.utc) >= timestamp(protocol['end_at']),'fixed league horizon incomplete')
    result = read_json(Path(evidence)/'prospective_result.json')
    require(result['league'] == controller.league and result['protocol_sha256'] == protocol['protocol_sha256'],'prospective final scope mismatch')
    require(result['status'] == 'FINALIZED_FIXED_HORIZON','prospective evidence not finalized')
    require(timestamp(protocol['end_at']) <= timestamp(result['finalized_at']) <= datetime.now(timezone.utc),'invalid finalization time')
    pairs = [read_json(p) for p in sorted((cohort/'predictions').glob('*.json'))]
    settlements = [read_json(p) for p in sorted((cohort/'settlements').glob('*.json'))]
    require(pairs and len(pairs) == len(settlements),'incomplete authoritative prospective cohort')
    exported_pairs = read_json(Path(evidence)/'prospective_predictions.json')
    exported_settlements = read_json(Path(evidence)/'prospective_settlements.json')
    for stored,exported in ((pairs,exported_pairs),(settlements,exported_settlements)):
        require(sorted(map(sha256_json,stored)) == sorted(map(sha256_json,exported)),'final export differs from authoritative league cohort')
    by_id = {s['fixture_id']:s for s in settlements}
    require(len(by_id) == len(pairs) == len({p['fixture_id'] for p in pairs}),'duplicate prospective fixture')
    actual,probabilities,lambdas = [],{'active':[],'candidate':[]},{'active':[],'candidate':[]}
    for pair in pairs:
        validate_pair(pair,controller=controller,protocol=protocol)
        settled = by_id[pair['fixture_id']]
        require(settled['league'] == controller.league and settled['protocol_sha256'] == protocol['protocol_sha256'],'settlement league/protocol mismatch')
        require(settled['prediction_pair_sha256'] == pair['pair_sha256'],'settlement pair join mismatch')
        require(settled['settlement_sha256'] == sha256_json({k:v for k,v in settled.items() if k != 'settlement_sha256'}),'settlement fingerprint mismatch')
        source_path = (controller.root/settled['source']['path']).resolve()
        require(source_path.is_relative_to(controller.root.resolve()) and sha256_file(source_path) == settled['source']['sha256'],'settlement source drift')
        source = read_json(source_path)
        require(source['league'] == controller.league and source['fixture_id'] == pair['fixture_id'] and source['status'] == 'FINISHED','settlement source identity mismatch')
        require([source['home_goals'],source['away_goals']] == [settled['home_goals'],settled['away_goals']],'settlement source score mismatch')
        require(timestamp(pair['kickoff_at']) < timestamp(source['observed_at']) <= timestamp(settled['created_at']) <= timestamp(result['finalized_at']),'settlement causal time mismatch')
        actual.append([settled['home_goals'],settled['away_goals']])
        for label in ('active','candidate'):
            record = pair[label]
            probabilities[label].append(record['prediction_output']['final_probabilities'])
            outputs = record['method_provenance']['descriptor']['ensemble']['member_outputs']
            lambdas[label].append([clamp_lambda(.6*clamp_lambda(outputs[side+'_goals_lgbm']['effective'])+.4*clamp_lambda(outputs[side+'_goals_xgb']['effective'])) for side in ('home','away')])
    actual = np.asarray(actual,float)
    labels = outcome_indices(actual[:,0],actual[:,1])
    contributions,reliability = {},{}
    for label in ('active','candidate'):
        p = np.asarray(probabilities[label],float)
        losses = probability_metric_contributions(labels,p)
        goals = goal_metric_contributions(actual.reshape(-1),np.asarray(lambdas[label],float).reshape(-1))
        contributions[label] = {'log_loss':losses['log_loss'],'multiclass_brier':losses['multiclass_brier'],
            'joint_goal_rmse':goals['squared_error'].reshape(-1,2).mean(axis=1),
            'joint_goal_poisson_deviance':goals['poisson_deviance'].reshape(-1,2).mean(axis=1)}
        reliability[label] = summarize_probability_metrics(labels,p)
    intervals = paired_bootstrap(contributions['active'],contributions['candidate'],**protocol['bootstrap'],root_mean_metrics=['joint_goal_rmse'])
    decision = apply_decision_rule(prospective_fixture_count=len(pairs),horizon_complete=True,paired_intervals=intervals['metrics'],
        active_reliability=reliability['active'],candidate_reliability=reliability['candidate'],provenance_verified=True)
    require(result['decision'] == decision and result['paired_intervals'] == intervals and result['reliability'] == reliability and result['fixture_count'] == len(pairs),'prospective final recomputation mismatch')
    return {'promotion_eligible':decision['phase8_promotion_review_authorized'],'classification':decision['classification'],
            'league':controller.league,'candidate_bundle_id':controller.candidate_id,
            'specification_fingerprint':controller.frozen['evaluation_specification']['specification_sha256'],
            'final_decision_sha256':sha256_json(decision),'prospective_protocol_sha256':protocol['protocol_sha256'],
            'final_evidence_sha256':sha256_file(Path(evidence)/'prospective_result.json')}

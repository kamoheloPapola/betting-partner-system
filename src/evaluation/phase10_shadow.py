"""Phase 10 adapter around Phase 9 pairing and the Phase 7 append-only store.

This module exposes no activation or model-fitting operation. Its controller is
composition-only: Phase 8 remains the sole production transition owner.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from src.evaluation import phase10_current_data as data
from src.evaluation.league_prospective import validate_pair
from src.evaluation.phase7p_prospective import ImmutableCohortStore
from src.ml.bundle_activation import persist_new
from src.ml.league_bundle_activation import LeagueBundleController


def sealed(value,key):
    data.check(value[key]==data.sha256_json({k:v for k,v in value.items() if k!=key}),'EVIDENCE_FINGERPRINT_DRIFT')


def reference(ref):
    path = (data.ROOT/ref['path']).resolve()
    data.check(path.is_relative_to(data.ROOT),'REFERENCE_PATH_ESCAPE')
    data.check(data.sha256_file(path)==ref['sha256'],'ARTIFACT_DRIFT')
    return path


def load_qualification(path):
    import pandas as pd
    path = Path(path).resolve()
    data.check(path.is_relative_to(data.ROOT/'baselines/phase10'),'QUALIFICATION_PATH_ESCAPE')
    q = data.read(path)
    sealed(q,'qualification_sha256')
    frozen = pd.read_csv(data.ROOT/'data/processed/matches.csv',low_memory=False)
    frozen = frozen.loc[frozen.league.eq(q['league'])]
    index = data.read(reference(q['source_index']))
    replay = data.assess_sources(index,frozen=frozen,names=data.exact_names(q['league']),as_of=q['as_of'])
    data.check(replay==q['assessment'],'QUALIFICATION_REPLAY_MISMATCH')
    data.check(q['feature_readiness'] and replay['qualified'] and not q['feature_failures'] and q['feature_proofs'],
               'CURRENT_DATA_UNQUALIFIED')
    feature = data.feature_reproduction(q['league'],q['ordered_features'])
    data.check(feature['reproduced'] and feature==q['historical_feature_reproduction'],'FEATURE_REPRODUCTION_MISMATCH')
    return q,index


class ShadowController:
    """Read-only model ownership, deliberately no inherited activation methods."""
    def __init__(self,qualification):
        self.root = data.ROOT
        self.league = qualification['league']
        data.check(self.league in data.LEAGUES,'WRONG_LEAGUE')
        self.owner = LeagueBundleController(reference(qualification['phase9_deployment']),league=self.league)
        data.check(not self.owner.pointer.exists() and not self.owner.pending.exists(),'ACTIVE_PRODUCTION_BUNDLE_CHANGED')
        self.legacy_id,self.candidate_id = self.owner.legacy_id,self.owner.candidate_id
        data.check([self.legacy_id,self.candidate_id]==[qualification['active_bundle_id'],qualification['candidate_bundle_id']],
                   'BUNDLE_SCOPE_MISMATCH')
        for bundle_id,field in ((self.legacy_id,'active_sha256'),(self.candidate_id,'candidate_sha256')):
            bundle = self.owner.bundle(bundle_id)
            data.check(bundle['aggregate_sha256']==qualification[field],'BUNDLE_DRIFT')
            data.check(bundle['features']==qualification['ordered_features'] and
                       data.sha256_json(bundle['features'])==qualification['feature_schema_sha256'],'FEATURE_SCHEMA_MISMATCH')

    def bundle(self,bundle_id):
        return self.owner.bundle(bundle_id)


def freeze(path,league,qualification_path):
    path = data.open_run(path)
    q,index = load_qualification(qualification_path)
    data.check(q['league']==league,'CROSS_LEAGUE_QUALIFICATION')
    controller = ShadowController(q)
    contract = Path(qualification_path).parent/'current_data_contract.json'
    value = data.read(contract)
    sealed(value,'contract_sha256')
    data.check(reference(value['qualification'])==Path(qualification_path).resolve(),'CONTRACT_QUALIFICATION_MISMATCH')
    directory = data.run_path(path)/league
    frozen_at = data.instant(data.now())
    future = [e for e in q['assessment']['upcoming'] if data.instant(e['kickoff_at'])>frozen_at+timedelta(seconds=1)]
    data.check(future,'NO_UPCOMING_CAUSAL_FIXTURES')
    current_payload = data.payload(index['sources'][3])
    season_ends = {m['season']['endDate'] for m in current_payload['matches']}
    data.check(len(season_ends)==1,'AMBIGUOUS_SEASON_END')
    end = data.instant(next(iter(season_ends))+'T00:00:00Z')+timedelta(days=1)
    protocol = {'schema_version':'ffps.league_prospective_protocol.v1','league':league,
        'frozen_at':frozen_at.isoformat(),'start_at':(frozen_at+timedelta(seconds=1)).isoformat(),'end_at':end.isoformat(),
        'season':q['assessment']['season'],'qualification':data.identity(Path(qualification_path)),
        'current_data_contract':data.identity(contract),'candidate_bundle_id':controller.candidate_id,
        'active_bundle_id':controller.legacy_id,'feature_schema_sha256':q['feature_schema_sha256'],
        'evaluation_specification_sha256':controller.owner.frozen['evaluation_specification']['specification_sha256'],
        'decision_implementation_sha256':data.sha256_file(data.ROOT/'src/evaluation/phase7_shadow.py'),
        'cohort_path':str((directory/'cohort').relative_to(data.ROOT).as_posix()),
        'bootstrap':{'resamples':10000,'confidence_level':.95,'seed':42},
        'eligible_fixture_policy':'all current-season league fixtures with TIMED or SCHEDULED status and complete causal features',
        'common_rng':'SHA256([league, fixture_id]) first 8 hex; identical RNG state for paired models',
        'promotion_authorized':False}
    protocol['protocol_sha256'] = data.sha256_json(protocol)
    persist_new(directory/'prospective_protocol.json',protocol)
    return protocol


def load_protocol(path,league):
    path = data.run_path(path)/league/'prospective_protocol.json'
    p = data.read(path)
    sealed(p,'protocol_sha256')
    data.check(p['league']==league and league in data.LEAGUES,'CROSS_LEAGUE_PROTOCOL')
    data.check(data.instant(p['frozen_at'])<data.instant(p['start_at'])<data.instant(p['end_at']),'INVALID_HORIZON')
    data.check(p['decision_implementation_sha256']==data.sha256_file(data.ROOT/'src/evaluation/phase7_shadow.py'),'DECISION_IMPLEMENTATION_DRIFT')
    q,index = load_qualification(reference(p['qualification']))
    controller = ShadowController(q)
    data.check(p['candidate_bundle_id']==controller.candidate_id and p['active_bundle_id']==controller.legacy_id,'BUNDLE_SCOPE_MISMATCH')
    data.check(p['feature_schema_sha256']==q['feature_schema_sha256'] and p['season']==q['assessment']['season'],'FEATURE_SCHEMA_MISMATCH')
    data.check(p['evaluation_specification_sha256']==controller.owner.frozen['evaluation_specification']['specification_sha256'],
               'CROSS_LEAGUE_EVALUATION_SPECIFICATION')
    data.check(p['bootstrap']['resamples']>=1000 and p['bootstrap']['confidence_level']==.95 and p['bootstrap']['seed']==42,
               'INVALID_FROZEN_UNCERTAINTY_PROTOCOL')
    reference(p['current_data_contract'])
    cohort = (data.ROOT/p['cohort_path']).resolve()
    data.check(cohort==path.parent/'cohort','COHORT_SCOPE_MISMATCH')
    return p,cohort,controller,q,index


def capture(matrix,fixture,suite,league,at):
    from src.cli.commands.prediction import _calc_goals
    from src.predictions.provenance import PredictionCapture,data_provenance_for_fixture,mark_output_reproduced,replay_pl_1x2
    from src.simulation.match_simulator import MatchSimulator,DEFAULT_N_SIMULATIONS
    simulator = MatchSimulator(league=league,seed=int(data.sha256_json([league,fixture.match_id])[:8],16),
        n_simulations=DEFAULT_N_SIMULATIONS,seed_scope='phase10_prospective_common_rng')
    with PredictionCapture(prediction_timestamp=at,data_provenance=data_provenance_for_fixture(matrix,fixture,at)) as context:
        probabilities = _calc_goals(fixture,suite,'Phase10Shadow',simulator,league=league,rl_weights=None,use_simulator=True)
        record = context.finalize(fixture=fixture,league=league,probabilities=probabilities,simulator=simulator)
    replay = replay_pl_1x2(record,root=data.ROOT)
    data.check(replay['reproduced'],'PREDICTION_REPLAY_FAILED')
    return mark_output_reproduced(record,replay).to_dict()


def pair_path(cohort,league,fixture):
    return cohort/'predictions'/(data.sha256_json([league,fixture])+'.json')


def persist_report(path,value):
    if path.exists():
        data.check(data.read(path)==value,'CONFLICTING_IMMUTABLE_REPORT')
    else:
        persist_new(path,value)


def collect(path,league):
    path = data.open_run(path)
    p,cohort,controller,q,index = load_protocol(path,league)
    active,candidate = [controller.owner.load(controller.bundle(b)) for b in (controller.legacy_id,controller.candidate_id)]
    report = {'league':league,'started_at':data.now(),'admitted':[],'rejected':[],'already_present':[]}
    with ImmutableCohortStore(cohort).batch_lock('phase10_collection'):
        for target in q['assessment']['upcoming']:
            destination = pair_path(cohort,league,target['fixture_id'])
            if destination.exists():
                old = data.read(destination)
                validate_pair(old,controller=controller,protocol=p)
                report['already_present'].append(target['fixture_id'])
                continue
            try:
                at = data.now()
                data.check(data.instant(p['start_at'])<=data.instant(target['kickoff_at'])<data.instant(p['end_at']),'OUTSIDE_FIXED_HORIZON')
                matrix,fixture = data.target_matrix(q['assessment'],target,as_of=at,source_index=index)
                data.validate_features(matrix,fixture,q['ordered_features'],at)
                fixture = fixture.copy()
                fixture['kickoff_utc'] = target['kickoff_at']
                at = data.now()
                records = [capture(matrix,fixture,s,league,at) for s in (active,candidate)]
                pair = {'league':league,'protocol_sha256':p['protocol_sha256'],'fixture_id':target['fixture_id'],
                    'created_at':data.now(),'kickoff_at':target['kickoff_at'],'qualification':p['qualification'],
                    'active':records[0],'candidate':records[1]}
                pair['pair_sha256'] = data.sha256_json(pair)
                validate_pair(pair,controller=controller,protocol=p)
                data.check(data.instant(data.now())<data.instant(target['kickoff_at']),'LATE_PREDICTION')
                persist_new(destination,pair)
                report['admitted'].append(target['fixture_id'])
            except (ValueError,KeyError,TypeError,RuntimeError) as exc:
                report['rejected'].append({'fixture_id':target['fixture_id'],'code':str(exc),'error_type':type(exc).__name__})
    report['finished_at'] = data.now()
    persist_report(cohort/'admission_reports'/(data.sha256_json(report)+'.json'),report)
    return report


def settle(path,league,source_index_path):
    """Only freshly archived authoritative FINISHED rows can settle existing pairs."""
    path = data.open_run(path)
    p,cohort,controller,q,_ = load_protocol(path,league)
    index = data.read(source_index_path)
    primary_source = result_source(index,league,p['season'])
    teams = q['assessment']['team_mappings'][str(p['season'])]
    events = data.primary_events(data.payload(primary_source),league,p['season'],teams,primary_source['retrieved_at'])
    report = {'league':league,'created_at':data.now(),'settled':[],'already_present':[],'rejected':[]}
    with ImmutableCohortStore(cohort).batch_lock('phase10_collection'):
        for event in events:
            if event['status']!='FINISHED':
                continue
            prediction = pair_path(cohort,league,event['fixture_id'])
            if not prediction.exists():
                previous = [t for t in q['assessment'].get('upcoming',[]) if
                            (t['season'],t['home_id'],t['away_id'])==(event['season'],event['home_id'],event['away_id'])]
                code = 'FIXTURE_ID_REVISED' if previous and all(t['fixture_id']!=event['fixture_id'] for t in previous) else 'NO_PREKICKOFF_PAIR'
                report['rejected'].append({'fixture_id':event['fixture_id'],'code':code})
                continue
            pair = data.read(prediction)
            validate_pair(pair,controller=controller,protocol=p)
            data.check(data.instant(pair['kickoff_at'])==data.instant(event['kickoff_at']),'KICKOFF_REVISED')
            data.check(data.instant(pair['created_at'])<data.instant(event['retrieved_at'])<=data.instant(data.now()),'RETROSPECTIVE_SETTLEMENT')
            destination = cohort/'settlements'/prediction.name
            scores = [event['home_goals'],event['away_goals']]
            if destination.exists():
                stored = data.read(destination)
                sealed(stored,'settlement_sha256')
                data.check([stored['home_goals'],stored['away_goals']]==scores,'CONFLICTING_RESULT_REVISION')
                report['already_present'].append(event['fixture_id'])
                continue
            record = {'league':league,'fixture_id':event['fixture_id'],'protocol_sha256':p['protocol_sha256'],
                'prediction_pair_sha256':pair['pair_sha256'],'created_at':data.now(),
                'home_goals':scores[0],'away_goals':scores[1],'source_index':data.identity(Path(source_index_path)),
                'normalized_event':event}
            record['settlement_sha256'] = data.sha256_json(record)
            persist_new(destination,record)
            report['settled'].append(event['fixture_id'])
    persist_report(cohort/'settlement_reports'/(data.sha256_json(report)+'.json'),report)
    return report


def result_source(index,league,season):
    """A later-season acquisition may settle the prior frozen season."""
    data.check(index['league']==league,'CROSS_LEAGUE_SETTLEMENT')
    expected = {'url':f'https://api.football-data.org/v4/competitions/{league}/matches','params':{'season':season}}
    sources = [source for source in index['sources'] if source.get('request')==expected]
    data.check(len(sources)==1,'SETTLEMENT_SOURCE_SCOPE_MISMATCH')
    return sources[0]


def verify_cohort(path,league):
    """Replay pairs and provider results, including feature reconstruction."""
    import numpy as np
    p,cohort,controller,q,index = load_protocol(path,league)
    pairs = [data.read(f) for f in sorted((cohort/'predictions').glob('*.json'))]
    settlements = [data.read(f) for f in sorted((cohort/'settlements').glob('*.json'))]
    by_id = {v['fixture_id']:v for v in pairs}
    data.check(len(by_id)==len(pairs),'DUPLICATE_OBSERVATION')
    data.check(len({v['fixture_id'] for v in settlements})==len(settlements),'DUPLICATE_SETTLEMENT')
    targets = {e['fixture_id']:e for e in q['assessment']['upcoming']}
    for pair in pairs:
        validate_pair(pair,controller=controller,protocol=p)
        data.check(pair['qualification']==p['qualification'],'PAIR_QUALIFICATION_MISMATCH')
        data.check(pair['fixture_id'] in targets,'UNKNOWN_TARGET_FIXTURE')
        matrix,fixture = data.target_matrix(q['assessment'],targets[pair['fixture_id']],as_of=pair['created_at'],source_index=index)
        data.validate_features(matrix,fixture,q['ordered_features'],pair['created_at'])
        for record in (pair['active'],pair['candidate']):
            for features in record['feature_provenance']['model_inputs']:
                data.check(np.array_equal(np.asarray(features['feature_values']),fixture[q['ordered_features']].to_numpy(float)),
                           'CAUSAL_FEATURE_REPLAY_MISMATCH')
    for settled in settlements:
        sealed(settled,'settlement_sha256')
        data.check(settled['league']==league and settled['protocol_sha256']==p['protocol_sha256'],'CROSS_LEAGUE_SETTLEMENT')
        data.check(settled['fixture_id'] in by_id,'NO_PREKICKOFF_PAIR')
        pair = by_id[settled['fixture_id']]
        data.check(settled['prediction_pair_sha256']==pair['pair_sha256'],'SETTLEMENT_PAIR_MISMATCH')
        source_index = data.read(reference(settled['source_index']))
        source = result_source(source_index,league,p['season'])
        events = data.primary_events(data.payload(source),league,p['season'],q['assessment']['team_mappings'][str(p['season'])],source['retrieved_at'])
        event = next((e for e in events if e['fixture_id']==settled['fixture_id']),None)
        data.check(event is not None and event==settled['normalized_event'] and event['status']=='FINISHED','SETTLEMENT_RAW_REPLAY_MISMATCH')
        data.check([event['home_goals'],event['away_goals']]==[settled['home_goals'],settled['away_goals']],'CONFLICTING_RESULT_REVISION')
        data.check(data.instant(pair['kickoff_at'])==data.instant(event['kickoff_at']),'KICKOFF_REVISED')
        data.check(data.instant(pair['kickoff_at'])<data.instant(event['retrieved_at'])<=data.instant(settled['created_at']), 'INVALID_SETTLEMENT_ORDER')
    return p,pairs,settlements


def evaluate(path,league):
    path = data.open_run(path)
    import numpy as np
    from src.evaluation.phase7_shadow import (outcome_indices,probability_metric_contributions,
        goal_metric_contributions,summarize_probability_metrics,paired_bootstrap,apply_decision_rule)
    from src.simulation.match_simulator import clamp_lambda
    p,pairs,settlements = verify_cohort(path,league)
    complete = data.instant(data.now())>=data.instant(p['end_at']) and len(pairs)==len(settlements)
    reliability,intervals = None,None
    by_id = {v['fixture_id']:v for v in pairs}
    if settlements:
        actual = np.array([[s['home_goals'],s['away_goals']] for s in settlements],float)
        labels = outcome_indices(actual[:,0],actual[:,1])
        contributions,reliability = {},{}
        for label in ('active','candidate'):
            records = [by_id[s['fixture_id']][label] for s in settlements]
            probabilities = np.array([r['prediction_output']['final_probabilities'] for r in records],float)
            lambdas = []
            for record in records:
                outputs = record['method_provenance']['descriptor']['ensemble']['member_outputs']
                lambdas.append([clamp_lambda(.6*clamp_lambda(outputs[side+'_goals_lgbm']['effective'])+
                    .4*clamp_lambda(outputs[side+'_goals_xgb']['effective'])) for side in ('home','away')])
            loss = probability_metric_contributions(labels,probabilities)
            goals = goal_metric_contributions(actual.reshape(-1),np.array(lambdas).reshape(-1))
            contributions[label] = {'log_loss':loss['log_loss'],'multiclass_brier':loss['multiclass_brier'],
                'joint_goal_rmse':goals['squared_error'].reshape(-1,2).mean(axis=1),
                'joint_goal_poisson_deviance':goals['poisson_deviance'].reshape(-1,2).mean(axis=1)}
            reliability[label] = summarize_probability_metrics(labels,probabilities)
        intervals = paired_bootstrap(contributions['active'],contributions['candidate'],**p['bootstrap'],root_mean_metrics=['joint_goal_rmse'])
    decision = apply_decision_rule(prospective_fixture_count=len(settlements),horizon_complete=complete,
        paired_intervals=intervals['metrics'] if intervals else None,
        active_reliability=reliability['active'] if reliability else None,
        candidate_reliability=reliability['candidate'] if reliability else None,provenance_verified=True)
    result = {'league':league,'protocol_sha256':p['protocol_sha256'],'created_at':data.now(),
        'pairs':len(pairs),'settlements':len(settlements),'fixed_horizon_complete':complete,
        'reliability':reliability,'paired_intervals':intervals,'decision':decision,'production_activation_authorized':False}
    persist_report(data.run_path(path)/league/'evaluations'/(data.sha256_json(result)+'.json'),result)
    return result

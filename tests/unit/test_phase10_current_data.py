"""Phase 10 source gates and isolated, genuinely timed shadow admission."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.evaluation import phase10_current_data as data
from src.evaluation import phase10_shadow as shadow


AT = '2026-09-06T10:00:00+00:00'


def team_payload(league,year):
    return {'competition':{'code':league},'season':{'startDate':f'{year}-08-01'},
            'teams':[{'id':1,'name':'Alpha','shortName':'Alpha'},{'id':2,'name':'Beta','shortName':'Beta'}]}


def match(league,year,fid,date,status='FINISHED'):
    return {'id':fid,'utcDate':date,'status':status,'competition':{'code':league},
            'season':{'startDate':f'{year}-08-01','endDate':f'{year+1}-05-31'},
            'homeTeam':{'id':1},'awayTeam':{'id':2},'lastUpdated':AT,
            'score':{'fullTime':{'home':1 if status=='FINISHED' else None,'away':0 if status=='FINISHED' else None}}}


def matches(league,year):
    rows = [match(league,year,year,f'{year}-08-10T12:00:00Z')]
    if year==2026:
        upcoming = match(league,year,year+10,'2026-09-10T12:00:00Z','TIMED')
        upcoming['homeTeam'],upcoming['awayTeam'] = {'id':2},{'id':1}
        rows.append(upcoming)
    return {'competition':{'code':league},'filters':{'season':year},'matches':rows,'resultSet':{'count':len(rows)}}


def sources(tmp_path,monkeypatch,league='BL1'):
    monkeypatch.setattr(data,'ROOT',tmp_path)
    division = {'BL1':'D1','FL1':'F1','PD':'SP1','SA':'I1'}[league]
    directory = tmp_path/'baselines/phase10/test'/league
    directory.mkdir(parents=True)
    items = []
    for year in (2025,2026):
        for endpoint in ('matches','teams','statistics'):
            if endpoint=='statistics':
                body = f'Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC\n{division},10/08/{year},Alpha,Beta,1,0,4,2\n'.encode()
                url = f'https://www.football-data.co.uk/mmz4281/{year%100:02d}{(year+1)%100:02d}/{division}.csv'
                params = {}
            else:
                body = json.dumps(matches(league,year) if endpoint=='matches' else team_payload(league,year)).encode()
                url = f'https://api.football-data.org/v4/competitions/{league}/{endpoint}'
                params = {'season':year}
            p = directory/f'{year}-{endpoint}.raw'
            p.write_bytes(body)
            items.append({'request':{'url':url,'params':params},'status':'RETRIEVED','http_status':200,
                'started_at':AT,'retrieved_at':AT,'raw':{'path':str(p.relative_to(tmp_path)),'sha256':data.sha256_file(p)}})
    return {'league':league,'requested_season':2026,'division':division,'sources':items}


def frozen(league='BL1'):
    return pd.DataFrame([{'league':league,'match_date':'2025-08-10','home_team':'ALPHA','away_team':'BETA',
                         'home_goals':1,'away_goals':0,'home_corners':4,'away_corners':2}])


def assess(index):
    return data.assess_sources(index,frozen=frozen(index['league']),names={'ALPHA':{'ALPHA'},'BETA':{'BETA'}},as_of=AT)


@pytest.mark.parametrize('league',data.LEAGUES)
def test_successful_qualification_is_per_league(tmp_path,monkeypatch,league):
    result = assess(sources(tmp_path,monkeypatch,league))
    assert result['qualified'],result['blockers']
    assert len(result['composed_observations'])==1
    assert set(result['composed_observations'][0]['field_provenance'])=={'home_goals','away_goals','home_corners','away_corners'}
    assert result['historical_overlap'][0]['statistics_present']


@pytest.mark.parametrize('mutation,code',[
    ('http','STATISTICS_HTTP_ERROR'),('empty','EMPTY_STATISTICS'),('missing','MISSING_REQUIRED_STATISTIC_FIELDS'),
    ('nonfinite','MISSING_OR_INVALID_STATISTICS'),('wrong_div','WRONG_LEAGUE'),('duplicate','DUPLICATE_STATISTICS_FIXTURE'),
    ('conflict','SOURCE_SEMANTICS_INCOMPATIBLE'),('unmapped','UNMAPPED_OR_AMBIGUOUS_STATISTICS_TEAM'),
    ('stale','STALE_SEASON')])
def test_statistics_fail_closed(tmp_path,monkeypatch,mutation,code):
    index = sources(tmp_path,monkeypatch)
    source = index['sources'][2]
    p = tmp_path/source['raw']['path']
    content = p.read_text()
    if mutation=='http':
        source.update(status='HTTP_ERROR',http_status=503)
    else:
        if mutation=='empty': content = content.splitlines()[0]+'\n'
        elif mutation=='missing': content = content.replace('HC','NoCorners')
        elif mutation=='nonfinite': content = content.replace(',4,2',',,2')
        elif mutation=='wrong_div': content = content.replace('D1,','E0,')
        elif mutation=='duplicate': content += content.splitlines()[1]+'\n'
        elif mutation=='conflict': content = content.replace(',4,2',',8,2')
        elif mutation=='unmapped': content = content.replace('Alpha','Unknown')
        elif mutation=='stale': content = content.replace('10/08/2025','10/08/2024')
        p.write_text(content)
        source['raw']['sha256'] = data.sha256_file(p)
    assert code in {b['code'] for b in assess(index)['blockers']}


@pytest.mark.parametrize('mutation,code',[
    ('duplicate','DUPLICATE_FIXTURE_ID'),('changed_id','CHANGED_OR_DUPLICATE_FIXTURE_ID'),
    ('wrong_league','WRONG_LEAGUE'),('stale','STALE_SEASON'),('empty','EMPTY_MATCHES'),
    ('timezone','INVALID_TIMEZONE_TIMESTAMP'),('future_update','FUTURE_PROVIDER_UPDATE'),('score','INVALID_SCORE')])
def test_primary_failure_classes(mutation,code):
    payload = matches('BL1',2026)
    if mutation in ('duplicate','changed_id'):
        row = copy.deepcopy(payload['matches'][0])
        if mutation=='changed_id': row['id']=99999
        payload['matches'].append(row)
        payload['resultSet']['count']+=1
    elif mutation=='wrong_league': payload['competition']['code']='PL'
    elif mutation=='stale': payload['filters']['season']=2025
    elif mutation=='empty': payload['matches']=[]
    elif mutation=='timezone': payload['matches'][0]['utcDate']='2026-08-10T12:00:00'
    elif mutation=='future_update': payload['matches'][0]['lastUpdated']='2026-09-07T12:00:00Z'
    elif mutation=='score': payload['matches'][0]['score']['fullTime']['home']=True
    teams,_ = data.map_teams(team_payload('BL1',2026),'BL1',2026,{'ALPHA':{'ALPHA'},'BETA':{'BETA'}})
    with pytest.raises(data.QualificationError,match=code):
        data.primary_events(payload,'BL1',2026,teams,AT)


def test_ambiguous_and_many_to_one_mapping():
    _,errors = data.map_teams(team_payload('BL1',2026),'BL1',2026,{'ALPHA':{'ONE','TWO'},'BETA':{'ONE'}})
    assert errors[0]['code']=='AMBIGUOUS_TEAM_MAPPING'
    _,errors = data.map_teams(team_payload('BL1',2026),'BL1',2026,{'ALPHA':{'ONE'},'BETA':{'ONE'}})
    assert errors[0]['code']=='MANY_TO_ONE_TEAM_MAPPING'


def test_promoted_id_membership_is_distinct_from_name_mapping(tmp_path,monkeypatch):
    index = sources(tmp_path,monkeypatch)
    source = index['sources'][4]
    p = tmp_path/source['raw']['path']
    payload = json.loads(p.read_bytes())
    payload['teams'].append({'id':3,'name':'Promoted New Club','shortName':'New Club'})
    p.write_text(json.dumps(payload))
    source['raw']['sha256']=data.sha256_file(p)
    result = assess(index)
    assert result['promoted_team_evidence']['membership_verified']
    assert result['promoted_team_evidence']['promoted'][0]['id']==3
    assert not result['promoted_team_evidence']['historical_mapping_verified']
    assert not result['qualified']


@pytest.mark.parametrize('change', ['late','cross_league','payload_drift'])
def test_source_binding(tmp_path,monkeypatch,change):
    index = sources(tmp_path,monkeypatch)
    if change=='late':
        index['sources'][0]['retrieved_at']='2026-09-07T10:00:00Z'
        with pytest.raises(data.QualificationError,match='LATE_RETRIEVAL'): assess(index)
    elif change=='cross_league':
        index['sources'][0]['request']['url']=index['sources'][0]['request']['url'].replace('BL1','PL')
        with pytest.raises(data.QualificationError,match='SOURCE_REQUEST_SCOPE_MISMATCH'): assess(index)
    else:
        (tmp_path/index['sources'][0]['raw']['path']).write_bytes(b'changed')
        assert 'RAW_PAYLOAD_DRIFT' in {b['code'] for b in assess(index)['blockers']}


def test_malformed_json_and_unavailable_are_reported(tmp_path,monkeypatch):
    index = sources(tmp_path,monkeypatch)
    index['sources'][0].update(status='SOURCE_UNAVAILABLE',http_status=None)
    source = index['sources'][3]
    (tmp_path/source['raw']['path']).write_bytes(b'<html>not json</html>')
    source['raw']['sha256']=data.sha256_file(tmp_path/source['raw']['path'])
    reasons = {b['code'] for b in assess(index)['blockers']}
    assert {'PRIMARY_HTTP_ERROR','MALFORMED_PRIMARY_JSON'}<=reasons


def test_batch_failure_does_not_skip_leagues(monkeypatch,tmp_path):
    observed = []
    def qualify(path,league,attempt):
        observed.append(league)
        if league=='BL1': raise RuntimeError('blocked')
        return {'league':league,'ready':False}
    monkeypatch.setattr(data,'ROOT',tmp_path)
    monkeypatch.setattr(data,'qualify',qualify)
    monkeypatch.setattr(data,'open_run',lambda p:p)
    result = data.qualify_all(tmp_path/'baselines/phase10/test','initial')
    assert observed==list(data.LEAGUES) and len(result)==4
    assert result[0]['code']=='QUALIFICATION_EXECUTION_FAILED'


def test_no_pl_acquisition_or_shadow_controller(tmp_path):
    with pytest.raises(Exception,match='PL'): data.acquire(tmp_path,'PL')
    with pytest.raises(data.QualificationError,match='WRONG_LEAGUE'): shadow.ShadowController({'league':'PL'})
    assert not hasattr(shadow.ShadowController,'activate')
    assert not hasattr(shadow.ShadowController,'rollback')


def test_contract_is_immutable(tmp_path):
    path = tmp_path/'contract.json'
    shadow.persist_new(path,{'league':'BL1'})
    with pytest.raises(Exception): shadow.persist_new(path,{'league':'PL'})
    assert data.read(path)=={'league':'BL1'}
    value = {'league':'BL1'}
    value['sha']=data.sha256_json(value)
    shadow.sealed(value,'sha')
    value['league']='PL'
    with pytest.raises(data.QualificationError): shadow.sealed(value,'sha')


@pytest.mark.parametrize('mode,code',[('nonfinite','NONFINITE'),('schema','FEATURE_SCHEMA'),('future','FUTURE_FEATURE'),('cutoff','CUTOFF_UNPROVEN')])
def test_feature_failure_classes(mode,code):
    row = {'match_id':'x','date':pd.Timestamp('2026-09-10T12:00:00Z'),'x':1.,'max_source_date':pd.Timestamp('2026-09-01T00:00:00Z')}
    names = ['x']
    if mode=='nonfinite': row['x']=float('nan')
    elif mode=='schema': names=['missing']
    elif mode=='future': row['max_source_date']=pd.Timestamp('2026-09-07T00:00:00Z')
    elif mode=='cutoff': row.pop('max_source_date')
    frame = pd.DataFrame([row])
    with pytest.raises(data.QualificationError,match=code): data.validate_features(frame,frame.iloc[0],names,AT)


def test_retrospective_feature_rebuild_rejected():
    target = {'kickoff_at':'2026-09-05T00:00:00Z'}
    with pytest.raises(data.QualificationError,match='RETROSPECTIVE'):
        data.target_matrix({'qualified':True,'upcoming':[target]},target,as_of=AT,source_index={})


def test_real_model_pair_collection_is_isolated(tmp_path,monkeypatch):
    """Real Phase 4 replay and Phase 9 model bytes; synthetic test-only fixture."""
    from datetime import timedelta
    from src.ml.league_bundle_activation import LeagueBundleController
    from src.predictions.provenance import build_file_snapshot,source_event_index
    from src.ml.bundle_activation import ActivationError
    league='BL1'
    monkeypatch.setattr(data,'open_run',lambda p:p)
    owner = LeagueBundleController(data.PHASE9/league/'deployment_specification.json',league=league)
    q = data.read(data.PHASE9/league/'integrity_specification.json')
    names = q['feature_contract']['ordered_feature_names']
    before = {str(p):data.sha256_file(p) for p in (data.PHASE9/league/'candidate').rglob('*.joblib')}
    controller = SimpleNamespace(root=data.ROOT,league=league,legacy_id=owner.legacy_id,candidate_id=owner.candidate_id,
                                 owner=owner,bundle=owner.bundle)
    frame = pd.read_csv(data.ROOT/'data/features/feature_matrix.csv',low_memory=False)
    frame = frame.loc[frame.league.eq(league)].tail(1).copy()
    future = data.instant(data.now())+timedelta(days=7)
    frame['date']=future.isoformat()
    frame['match_id']=frame['match_hash']='PHASE10_TEST_ONLY_BL1'
    frame['home_score']=frame['away_score']=float('nan')
    frame.attrs['ffps_data_snapshot'] = build_file_snapshot(frame=frame,
        artifacts=[dict(data.identity(data.ROOT/'data/features/feature_matrix.csv'),role='test_frozen_feature_source',load_order=0)],
        source_type='phase10_test_only')
    frame.attrs['ffps_source_events'] = source_event_index(data.base_frame(league))
    target={'fixture_id':'PHASE10_TEST_ONLY_BL1','kickoff_at':future.isoformat(),'league':league}
    protocol={'league':league,'frozen_at':(data.instant(data.now())-timedelta(days=1)).isoformat(),
        'start_at':data.now(),'end_at':(future+timedelta(days=1)).isoformat(),'qualification':{},'season':2026,
        'bootstrap':{'resamples':1000,'confidence_level':.95,'seed':42}}
    protocol['protocol_sha256']=data.sha256_json(protocol)
    qualification={'assessment':{'upcoming':[target]},'ordered_features':names}
    cohort=tmp_path/'BL1/cohort'
    monkeypatch.setattr(shadow,'load_protocol',lambda *args:(protocol,cohort,controller,qualification,{}))
    monkeypatch.setattr(data,'target_matrix',lambda *args,**kwargs:(frame,frame.iloc[0]))
    result=shadow.collect(tmp_path,league)
    assert result['admitted']==[target['fixture_id']],result
    again=shadow.collect(tmp_path,league)
    assert again['already_present']==result['admitted'] and not again['admitted']
    assert not (tmp_path/'PL').exists()
    assert {str(p):data.sha256_file(p) for p in (data.PHASE9/league/'candidate').rglob('*.joblib')}==before
    pair=data.read(shadow.pair_path(cohort,league,target['fixture_id']))
    assert pair['active']['data_provenance']==pair['candidate']['data_provenance']
    _,verified_pairs,settlements=shadow.verify_cohort(tmp_path,league)
    assert len(verified_pairs)==1 and not settlements
    monkeypatch.setattr(data,'run_path',lambda p:p)
    interim=shadow.evaluate(tmp_path,league)
    assert not interim['decision']['phase8_promotion_review_authorized']
    # Exercise Phase 7 metrics with one synthetic test settlement; no real cohort
    # is populated with this outcome and its horizon remains incomplete.
    monkeypatch.setattr(shadow,'verify_cohort',lambda *args:(protocol,[pair],[{
        'fixture_id':pair['fixture_id'],'home_goals':1,'away_goals':0}]))
    interim=shadow.evaluate(tmp_path,league)
    assert interim['paired_intervals'] and interim['reliability']
    assert not interim['decision']['phase8_promotion_review_authorized']
    for mutation in ('league','features','rng'):
        bad=copy.deepcopy(pair)
        if mutation=='league': bad['league']='PL'
        elif mutation=='features': bad['candidate']['feature_provenance']['model_inputs'][0]['ordered_feature_names'][0]='wrong'
        else: bad['candidate']['method_provenance']['descriptor']['simulation']['seed']=1
        bad['pair_sha256']=data.sha256_json({k:v for k,v in bad.items() if k!='pair_sha256'})
        with pytest.raises(ActivationError): shadow.validate_pair(bad,controller=controller,protocol=protocol)


def test_settlement_requires_pair_and_rejects_revision(tmp_path,monkeypatch):
    league='BL1'
    monkeypatch.setattr(data,'open_run',lambda p:p)
    cohort=tmp_path/'cohort'
    protocol={'protocol_sha256':'p','season':2026}
    teams,_=data.map_teams(team_payload(league,2026),league,2026,{'ALPHA':{'ALPHA'},'BETA':{'BETA'}})
    qualification={'assessment':{'team_mappings':{'2026':teams}}}
    monkeypatch.setattr(shadow,'load_protocol',lambda *args:(protocol,cohort,None,qualification,None))
    monkeypatch.setattr(shadow,'validate_pair',lambda *args,**kwargs:None)
    monkeypatch.setattr(data,'now',lambda:AT)
    monkeypatch.setattr(data,'identity',lambda p:{'path':str(p),'sha256':data.sha256_file(p)})
    raw=matches(league,2026)
    monkeypatch.setattr(data,'payload',lambda _:raw)
    index={'league':league,'requested_season':2026,'sources':[{}, {}, {},
        {'request':{'url':f'https://api.football-data.org/v4/competitions/{league}/matches','params':{'season':2026}},'retrieved_at':AT}]}
    path=tmp_path/'source-index.json'
    path.write_text(json.dumps(index))
    result=shadow.settle(tmp_path,league,path)
    assert result['rejected'][0]['code']=='NO_PREKICKOFF_PAIR'
    fid='BL1_football_data_org_2026'
    pair={'pair_sha256':'pair','kickoff_at':'2026-08-10T12:00:00Z','created_at':'2026-08-09T12:00:00Z'}
    shadow.persist_new(shadow.pair_path(cohort,league,fid),pair)
    result=shadow.settle(tmp_path,league,path)
    assert result['settled']==[fid]
    # Identical repeated settlement returns the existing result; no second row.
    monkeypatch.setattr(data,'now',lambda:'2026-09-06T10:00:01+00:00')
    again=shadow.settle(tmp_path,league,path)
    assert again['already_present']==[fid]
    assert len(list((cohort/'settlements').glob('*.json')))==1
    raw['matches'][0]['score']['fullTime']['home']=2
    with pytest.raises(data.QualificationError,match='CONFLICTING_RESULT_REVISION'):
        shadow.settle(tmp_path,league,path)
    raw['matches'][0]['utcDate']='2026-08-10T13:00:00Z'
    with pytest.raises(data.QualificationError,match='KICKOFF_REVISED'):
        shadow.settle(tmp_path,league,path)


def test_finalized_run_cannot_be_extended(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'ROOT',tmp_path)
    run=tmp_path/'baselines/phase10/test'
    run.mkdir(parents=True)
    (run/'run_specification.json').write_text('{}')
    (run/'artifact_index.json').write_text('{}')
    with pytest.raises(data.QualificationError,match='RUN_ALREADY_FINALIZED'):
        data.open_run(run)


def test_statistic_bookmaker_columns_are_not_prediction_inputs():
    raw=b'Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,HC,AC,B365H\nD1,10/08/2026,Alpha,Beta,1,0,4,2,1.1\n'
    rows=data.statistics_events(raw,'BL1',2026,'D1',{'ALPHA':{'ALPHA'},'BETA':{'BETA'}})
    assert 'B365H' not in rows[0]


def test_entire_stale_season_index_rejected(tmp_path,monkeypatch):
    index=sources(tmp_path,monkeypatch)
    index['requested_season']=2025
    with pytest.raises(data.QualificationError,match='STALE_SEASON'): assess(index)


def test_credentials_are_not_archived_if_provider_reflects_them(tmp_path,monkeypatch):
    from src.ml.bundle_activation import ActivationError
    monkeypatch.setattr(data.requests,'get',lambda *args,**kwargs:SimpleNamespace(
        content=b'reflected secret-token',status_code=200,headers={}))
    with pytest.raises(ActivationError,match='credential reflected'):
        data.fetch(tmp_path,'test','https://api.football-data.org/v4/competitions/BL1/matches',headers={'X-Auth-Token':'secret-token'})
    assert not list(tmp_path.glob('*.raw')) and not list(tmp_path.glob('*.json'))


def test_later_season_acquisition_can_settle_frozen_prior_season():
    source={'request':{'url':'https://api.football-data.org/v4/competitions/BL1/matches','params':{'season':2026}}}
    index={'league':'BL1','requested_season':2027,'sources':[source]}
    assert shadow.result_source(index,'BL1',2026)==source
    with pytest.raises(data.QualificationError,match='CROSS_LEAGUE'):
        shadow.result_source(index,'PL',2026)
    with pytest.raises(data.QualificationError,match='SCOPE_MISMATCH'):
        shadow.result_source(index,'BL1',2025)

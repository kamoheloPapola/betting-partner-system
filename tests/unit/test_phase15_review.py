"""Isolated incident reviews; no writes to real model or cohort evidence."""
import copy
import json
from pathlib import Path
import pytest
from src.evaluation import phase15_review_contract as c
from src.evaluation import phase15_review as r
from src.evaluation import phase15_review_verifier as v
from src.evaluation import phase15_phase14 as integration


@pytest.fixture
def incident(tmp_path, monkeypatch):
    p = c.a.p
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(c.a, 'COHORT', tmp_path / 'baselines/phase12/test')
    monkeypatch.setattr(c, 'RUN', tmp_path / 'baselines/phase15/test')
    def ref(path):
        return {'path': Path(path).relative_to(tmp_path).as_posix(), 'sha256': p.source.sha256_file(path),
                'size_bytes': Path(path).stat().st_size}
    def reference(value):
        path = tmp_path / value['path']
        p.check(path.is_file() and ref(path) == {k: value[k] for k in ('path','sha256','size_bytes')}, 'SOURCE_HASH_MISMATCH')
        return path
    monkeypatch.setattr(p, 'identity', ref)
    monkeypatch.setattr(p.frozen, 'reference', reference)
    for name in (*c.RULE_SOURCES, 'src/evaluation/phase15_review_contract.py'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('isolated frozen rule source')
    fixture = 'BL1_football_data_org_565839'
    original = {'league':'BL1','season':2026,'fixture_id':fixture,'provider_fixture_id':565839,
        'home_id':18,'away_id':29,'home_team':"M'GLADBACH",'away_team':'PADERBORN',
        'kickoff_at':'2026-10-31T00:00:00+00:00','retrieved_at':'2026-09-09T18:43:12+00:00'}
    revised = {**original,'kickoff_at':'2026-10-31T14:30:00+00:00','retrieved_at':'2026-09-14T18:08:19+00:00'}
    def metadata(name, target):
        path = tmp_path / (name + '.json')
        p.persist_new(path, {'matches':[target]})
        return {'provider':'football-data.org:v4','status':'RETRIEVED','retrieved_at':target['retrieved_at'],
            'request':{'url':'https://api.football-data.org/v4/competitions/BL1/matches','params':{'season':2026}},'raw':ref(path)}
    old, new = metadata('original',original), metadata('revised',revised)
    indexpath = tmp_path/'source_index.json'
    p.persist_new(indexpath,{'sources':[old]})
    qpath = tmp_path/'qualification.json'
    p.persist_new(qpath,{'source_index':ref(indexpath)})
    cutoff='2026-09-09T18:47:11+00:00'
    record = {'fixture':{'prediction_timestamp':cutoff},'recorded_at':'2026-09-09T18:47:25+00:00',
        'data_provenance':{'causal_cutoff':cutoff,'maximum_source_event_timestamp':'2026-09-05T13:30:00+00:00',
                           'source_events':[{'event_timestamp':'2026-09-05T13:30:00+00:00'}]},
        'feature_provenance':{'model_inputs':[{'causal_cutoff':cutoff}],'feature_bundle_sha256':'same-feature-hash'}}
    pair = p.seal({'fixture_id':fixture,'kickoff_at':original['kickoff_at'], 'created_at':'2026-09-09T18:47:40+00:00',
        'qualification':ref(qpath),'active':record,'candidate':copy.deepcopy(record)},'pair_sha256')
    event = p.seal({'sequence':1,'kind':'pair','created_at':'2026-09-09T18:47:41+00:00','payload':pair})
    issue = p.seal({'sequence':2,'kind':'incident','created_at':'2026-09-14T18:11:46+00:00',
        'payload':{'fixture_id':fixture,'code':'ADMITTED_KICKOFF_REVISION','protocol_violation':True,'source':new}})
    protocol={'season':2026,'end_at':'2027-05-23T00:00:00+00:00','protocol_sha256':'BL1-protocol',
              'qualification':ref(qpath),'current_data_contract':ref(qpath)}
    path=tmp_path/'protocol.json'
    p.persist_new(path,protocol)
    contract={'protocol':ref(path)}
    context=[protocol,None,{},[pair],[],{fixture:original},[event,issue]]
    for e in context[6]:
        p.persist_new(c.a.COHORT/'BL1/journal'/f"{e['sequence']:08d}.json",e)
    p.persist_new(c.a.COHORT/'run_specification.json',{'leagues':{'BL1':{'prefix':[]}}})
    monkeypatch.setattr(p,'specification',lambda _:p.read(c.a.COHORT/'run_specification.json'))
    monkeypatch.setattr(p,'load_cohort',lambda *args:context)
    monkeypatch.setattr(c.d14,'binding',lambda _: (None,contract,{'league':'BL1','test':True}))
    monkeypatch.setattr(p,'current_events',lambda source,*args:p.read(reference(source['raw']))['matches'])
    monkeypatch.setattr(p,'journal',lambda *args:context[6])
    return {'context':context,'original':original,'revised':revised,'pair':pair,'root':tmp_path,'new':new}


def test_exact_incident_and_no_review(incident):
    value=r.status('BL1',565839)
    assert value['review'] is None and value['state']=='UNRESOLVED_PROTOCOL_CONFLICT'
    facts=value['basis']['revisions'][0]['facts']
    assert facts['original']['kickoff_at']=='2026-10-31T00:00:00+00:00'
    assert facts['revised']['kickoff_at']=='2026-10-31T14:30:00+00:00'
    assert facts['feature_times_compatible'] and facts['prediction_times_compatible']
    assert not facts['cohort_order_changed'] and not facts['kickoff_instant_equal']


def test_immutable_idempotent_review_and_independent_verifier(incident):
    first=r.evaluate('BL1',565839)
    path=c.a.p.ROOT/first['review']['path']
    before=path.read_bytes()
    assert first['outcome']=='UNRESOLVED_PROTOCOL_CONFLICT'
    assert v.verify(path)['verified']
    assert r.evaluate('BL1',565839)['idempotent']
    assert path.read_bytes()==before


@pytest.mark.parametrize('field', ['identity_matches','feature_times_compatible','prediction_times_compatible','source_retrievals_compatible'])
def test_scientific_violation_is_blocking(incident,field):
    facts=c.reconstruct('BL1',565839)['revisions'][0]['facts']
    facts[field]=False
    result=c.adjudicate(facts)
    assert result['outcome']=='BLOCKED_SCIENTIFIC_VIOLATION'
    assert not result['clears_protocol_blocker']


def test_resolvable_factual_identity_has_no_discretionary_clearance(incident):
    facts=c.reconstruct('BL1',565839)['revisions'][0]['facts']
    facts['kickoff_instant_equal']=True
    result=c.adjudicate(facts)
    assert result['outcome']=='RESOLVED_FACTUAL_IDENTITY'
    assert not result['clears_protocol_blocker']
    assert 'NO_FROZEN_REVIEW_CLEARANCE_AUTHORITY' in result['reason_codes']
    base={'state':'AWAITING_HORIZON','blockers':[{'code':'AWAITING_HORIZON'}]}
    report=integration.attach_verified(base,[{'verified':True,**result}])
    assert report['state']=='AWAITING_HORIZON' and report['blockers']==base['blockers']


@pytest.mark.parametrize('filename', ['original.json','revised.json'])
def test_altered_kickoff_or_source_hash_rejected(incident,filename):
    path=incident['root']/filename
    row=c.a.p.read(path)
    row['matches'][0]['kickoff_at']='2026-10-01T00:00:00+00:00'
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='SOURCE_HASH_MISMATCH'):
        r.evaluate('BL1',565839)


@pytest.mark.parametrize('kind', ['feature','prediction'])
def test_actual_timestamp_incompatibility(incident,kind):
    record=incident['pair']['candidate']
    if kind=='feature':
        record['data_provenance']['source_events'][0]['event_timestamp']='2026-11-01T00:00:00+00:00'
    else:
        record['fixture']['prediction_timestamp']='2026-11-01T00:00:00+00:00'
    result=r.evaluate('BL1',565839)
    assert result['outcome']=='BLOCKED_SCIENTIFIC_VIOLATION'


def test_cohort_order_effect(incident):
    incident['context'][3].append({'fixture_id':'BL1-other','kickoff_at':'2026-10-31T10:00:00+00:00','pair_sha256':'other'})
    result=c.reconstruct('BL1',565839)
    assert result['revisions'][0]['facts']['cohort_order_changed']
    assert result['outcome']=='BLOCKED_SCIENTIFIC_VIOLATION'


def test_lineage_successor_and_historical_verification(incident):
    first=r.evaluate('BL1',565839)
    old=c.a.p.ROOT/first['review']['path']
    original=old.read_bytes()
    event=c.a.p.seal({'sequence':3,'kind':'incident','created_at':'2026-09-15T00:00:00+00:00',
        'payload':{'fixture_id':'other','code':'SOURCE_UNAVAILABLE','source':None,'protocol_violation':False}})
    incident['context'][6].append(event)
    c.a.p.persist_new(c.a.COHORT/'BL1/journal/00000003.json',event)
    with pytest.raises(ValueError,match='RECONSTRUCTION_OR_LINEAGE'):
        v.verify_locked(old)
    assert r.status('BL1',565839)['review']['state']=='BLOCKED_STALE_OR_INVALID_REVIEW'
    second=r.evaluate('BL1',565839)
    assert not second['idempotent'] and old.read_bytes()==original
    successor=c.a.p.read(c.a.p.ROOT/second['review']['path'])
    assert successor['predecessor']==first['review']
    historical=v.verify(old)
    assert historical['verified'] and not historical['applies_to_current_lineage']


@pytest.mark.parametrize('reseal', [False,True])
def test_review_mutation_rejected(incident,reseal):
    result=r.evaluate('BL1',565839)
    path=c.a.p.ROOT/result['review']['path']
    row=c.a.p.read(path)
    row['clears_protocol_blocker']=True
    if reseal:
        row.pop('review_sha256')
        row=c.a.p.seal(row,'review_sha256')
    path.write_bytes(v.canonical_bytes(row))
    with pytest.raises(ValueError):
        v.verify(path)


def test_phase14_consumes_verified_review_without_waiver(incident,monkeypatch):
    r.evaluate('BL1',565839)
    base={'state':'BLOCKED_PROTOCOL_VIOLATION','blockers':[{'code':'AWAITING_HORIZON'},{'code':'BLOCKED_PROTOCOL_VIOLATION'}]}
    monkeypatch.setattr(integration.d,'inspect_locked',lambda _:copy.deepcopy(base))
    result=integration.status('BL1',565839)
    assert result['state']==base['state'] and result['blockers']==base['blockers']
    assert result['incident_reviews'][0]['verified']
    path=next((c.RUN/'reviews/BL1/565839').glob('review-*.json'))
    path.write_text('{}')
    assert integration.status('BL1',565839)['state']=='BLOCKED_REVIEW_INVALID'


def test_manual_clearance_never_accepted():
    with pytest.raises(ValueError,match='UNAUTHORIZED_REVIEW_CLEARANCE'):
        integration.attach_verified({'state':'AWAITING_HORIZON'},[{'verified':True,'clears_protocol_blocker':True}])


def test_os_lock_excludes_competing_review(incident):
    import subprocess
    import sys
    lock=c.a.COHORT/'BL1/operation.lock'
    code='from pathlib import Path; from src.ml.bundle_activation import state_lock;\nwith state_lock(Path('+repr(str(lock))+')): print("unexpected")'
    with c.a.p.state_lock(lock):
        child=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=30)
    assert child.returncode and 'busy' in child.stderr


def test_unbound_review_path_refused(incident):
    with pytest.raises(ValueError):
        v.verify(incident['root']/'outside/review.json')


@pytest.mark.parametrize('league',['SA','FL1','PD'])
def test_unaffected_league_status_compatibility(incident,monkeypatch,league):
    base={'league':league,'state':'AWAITING_HORIZON','blockers':[{'code':'AWAITING_HORIZON'}]}
    monkeypatch.setattr(integration.d,'inspect_locked',lambda _:copy.deepcopy(base))
    result=integration.status(league,1)
    assert result['state']==base['state'] and result['blockers']==base['blockers']
    assert result['incident_reviews']==[]


def test_verifier_does_not_import_publisher():
    import ast
    tree=ast.parse(Path(v.__file__).read_text())
    assert not any(isinstance(node,ast.ImportFrom) and any(alias.name=='phase15_review' for alias in node.names)
                   for node in ast.walk(tree))

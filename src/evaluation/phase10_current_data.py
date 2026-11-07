"""Immutable current-source qualification; no production writes or model fitting."""
from __future__ import annotations

import io
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import dotenv_values

from src.evaluation.multi_league_integrity import ROOT, identity, read, write
from src.predictions.provenance import sha256_file, sha256_json
from src.ml.bundle_activation import require

LEAGUES = ('BL1', 'FL1', 'PD', 'SA')
PHASE9 = ROOT / 'baselines/phase9/phase9-multi-league-20260905-v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def protected_snapshot():
    return {p.relative_to(ROOT).as_posix(): sha256_file(p)
            for base in ('baselines', 'data', 'src/ml/models')
            for p in sorted((ROOT/base).rglob('*'))
            if p.is_file() and not p.is_relative_to(ROOT/'baselines/phase10')}


def run_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT/'baselines/phase10') and path != ROOT/'baselines/phase10',
            'Phase 10 evidence must be under its own run directory')
    return path


def open_run(path):
    path = run_path(path)
    check((path/'run_specification.json').exists(),'RUN_NOT_PREPARED')
    check(not (path/'artifact_index.json').exists(),'RUN_ALREADY_FINALIZED')
    return path


def prepare(path):
    path = run_path(path)
    require(not path.exists(), 'immutable run already exists')
    before = protected_snapshot()
    write(path/'protected_before.json', before)
    write(path/'run_specification.json', {'schema_version':'ffps.phase10_run.v1',
        'created_at':now(), 'leagues':list(LEAGUES), 'phase9_run':str(PHASE9.relative_to(ROOT)),
        'phase9_index':identity(PHASE9/'artifact_index.json'), 'promotion_authorized':False,
        'model_training_authorized':False})


def fetch(directory, name, url, *, params=None, headers=None):
    """Archive every response, including HTTP errors. Never persist credentials."""
    require(name.replace('-', '').replace('_', '').isalnum(), 'unsafe source name')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/(name+'.raw')
    require(not target.exists() and not (directory/(name+'.json')).exists(), 'source already archived')
    started = now()
    record = {'schema_version':'ffps.phase10_raw_source.v1','request':{'url':url,'params':params or {}},
              'started_at':started, 'provider_publication_at':None}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30, allow_redirects=False)
        body = response.content
        record.update(http_status=response.status_code, retrieved_at=now(),
                      last_modified=response.headers.get('Last-Modified'),
                      content_type=response.headers.get('Content-Type'),
                      status='RETRIEVED' if response.status_code == 200 else 'HTTP_ERROR')
        # Provider error text can reflect credentials. Archive successful provider
        # bytes verbatim; reject any credential reflection rather than persist it.
        for header,secret in (headers or {}).items():
            if header.lower() not in ('authorization','x-auth-token'):
                continue
            require(not secret or str(secret).encode() not in body, 'credential reflected in response')
        with target.open('xb') as stream:
            stream.write(body)
        record['raw'] = identity(target)
    except requests.RequestException as exc:
        record.update(status='SOURCE_UNAVAILABLE', error_type=type(exc).__name__, retrieved_at=now())
    write(directory/(name+'.json'),record)
    return record


def acquire(path, league, attempt='initial'):
    from src.data_pipeline.build_matches_dataset import DIV_MAP
    from src.utils.naming import calculate_season
    require(league in LEAGUES, 'Phase 10 cannot acquire PL or unknown leagues')
    path = open_run(path)
    season = calculate_season(datetime.now(timezone.utc), league)
    div = next(d for d,l in DIV_MAP.items() if l == league)
    key = dotenv_values(ROOT/'.env').get('FOOTBALL_DATA_API_KEY') or os.getenv('FOOTBALL_DATA_API_KEY')
    headers = {'X-Auth-Token':key} if key else {}
    require(attempt.replace('-', '').isalnum(), 'unsafe acquisition attempt')
    directory = path/league/'raw_sources'/attempt
    sources = []
    for year in (season-1,season):
        for endpoint in ('matches','teams'):
            sources.append(fetch(directory, f'primary-{year}-{endpoint}',
                f'https://api.football-data.org/v4/competitions/{league}/{endpoint}',
                params={'season':year}, headers=headers))
        token = f'{year % 100:02d}{(year+1) % 100:02d}'
        sources.append(fetch(directory,f'statistics-{year}',
            f'https://www.football-data.co.uk/mmz4281/{token}/{div}.csv'))
    result = {'league':league,'requested_season':season,'division':div,'sources':sources}
    write(path/league/f'raw_source_index_{attempt}.json',result)
    return {'league':league,'sources':[{k:s.get(k) for k in ('status','http_status','error_type')} for s in sources]}


def retry_statistics(path,league,attempt):
    check(league in LEAGUES,'WRONG_LEAGUE')
    path = open_run(path)
    index = read(path/league/f'raw_source_index_{attempt}.json')
    for offset in (2,5):
        old = index['sources'][offset]
        index['sources'][offset] = fetch(path/league/'raw_sources/statistics-retry',f'statistics-{index["requested_season"]-1+offset//3}',
            old['request']['url'],headers={'User-Agent':'FFPS-source-qualification/1.0','Accept':'text/csv'})
    write(path/league/'raw_source_index_statistics-retry.json',index)
    return {'league':league,'statistics_http_status':[index['sources'][i].get('http_status') for i in (2,5)]}


class QualificationError(ValueError):
    pass


def check(condition, code):
    if not condition:
        raise QualificationError(code)


def instant(value):
    from src.evaluation.league_prospective import timestamp
    from src.ml.bundle_activation import ActivationError
    try:
        return timestamp(value)
    except (ValueError, TypeError, ActivationError):
        raise QualificationError('INVALID_TIMEZONE_TIMESTAMP') from None


def token(name):
    check(isinstance(name,str) and bool(name.strip()), 'INVALID_TEAM_NAME')
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKD',name).encode('ascii','ignore').decode().upper()).strip()


def exact_names(league):
    """Existing explicit aliases and canonical spellings only; never fuzzy match."""
    from src.utils.naming import LEAGUE_TEAMS, LEAGUE_ALIASES
    names = {}
    for season in LEAGUE_TEAMS[league].values():
        for name in season:
            names.setdefault(token(name),set()).add(name.upper())
    for aliases in LEAGUE_ALIASES.get(league,{}).values():
        for name, canonical in aliases.items():
            names.setdefault(token(name),set()).add(canonical.upper())
    return names


def map_teams(payload, league, season, names):
    check(payload['competition']['code'] == league, 'WRONG_LEAGUE')
    check(int(payload['season']['startDate'][:4]) == season, 'STALE_SEASON')
    teams = payload['teams']
    check(isinstance(teams,list) and teams, 'EMPTY_TEAMS')
    result, failures, used = {}, [], {}
    for team in teams:
        tid = team['id']
        check(type(tid) is int and tid > 0 and str(tid) not in result, 'DUPLICATE_TEAM_ID')
        possibilities = set()
        from src.utils.naming import _normalization_candidates
        for field in ('name','shortName'):
            if team.get(field):
                for variant in _normalization_candidates(token(team[field])):
                    possibilities.update(names.get(variant,set()))
        canonical = next(iter(possibilities)) if len(possibilities) == 1 else None
        reason = 'UNMAPPED_TEAM' if not possibilities else 'AMBIGUOUS_TEAM_MAPPING' if len(possibilities)>1 else None
        if canonical in used:
            reason = 'MANY_TO_ONE_TEAM_MAPPING'
        if reason:
            failures.append({'team_id':tid,'name':team['name'],'code':reason,'candidates':sorted(possibilities)})
        elif canonical:
            used[canonical] = tid
        result[str(tid)] = {'id':tid,'name':team['name'],'short_name':team.get('shortName'),
                            'canonical':canonical,'status':reason or 'EXACT_REPOSITORY_ALIAS'}
    return result, failures


def primary_events(payload,league,season,teams,retrieved_at):
    check(payload['competition']['code'] == league, 'WRONG_LEAGUE')
    check(payload.get('filters',{}).get('season') == season, 'STALE_SEASON')
    rows = payload['matches']
    check(isinstance(rows,list) and rows, 'EMPTY_MATCHES')
    check(payload['resultSet']['count'] == len(rows), 'FIXTURE_COUNT_MISMATCH')
    events, ids, joins = [], set(), set()
    for raw in rows:
        check(raw['competition']['code'] == league, 'WRONG_LEAGUE')
        check(int(raw['season']['startDate'][:4]) == season, 'STALE_SEASON')
        kickoff = instant(raw['utcDate'])
        check(instant(raw['season']['startDate']+'T00:00:00Z') <= kickoff <
              instant(raw['season']['endDate']+'T00:00:00Z')+pd.Timedelta(days=1), 'KICKOFF_OUTSIDE_SEASON')
        fid = raw['id']
        check(type(fid) is int and fid > 0 and fid not in ids, 'DUPLICATE_FIXTURE_ID')
        ids.add(fid)
        h,a = str(raw['homeTeam']['id']),str(raw['awayTeam']['id'])
        check(h in teams and a in teams and h != a, 'INVALID_FIXTURE_TEAM')
        join = (season,h,a)
        check(join not in joins, 'CHANGED_OR_DUPLICATE_FIXTURE_ID')
        joins.add(join)
        updated = raw.get('lastUpdated')
        if updated:
            check(instant(updated) <= instant(retrieved_at), 'FUTURE_PROVIDER_UPDATE')
        scores = raw['score']['fullTime']
        if raw['status'] == 'FINISHED':
            check(kickoff < instant(retrieved_at), 'FUTURE_RESULT')
            check(all(type(scores[s]) is int and scores[s]>=0 for s in ('home','away')), 'INVALID_SCORE')
        events.append({'league':league,'season':season,'fixture_id':f'{league}_football_data_org_{fid}',
            'provider_fixture_id':fid,'kickoff_at':kickoff.isoformat(),'fixture_date':kickoff.date().isoformat(),
            'home_id':int(h),'away_id':int(a),'home_team':teams[h]['canonical'],'away_team':teams[a]['canonical'],
            'status':raw['status'],'home_goals':scores['home'],'away_goals':scores['away'],
            'provider_updated_at':updated,'retrieved_at':retrieved_at})
    return events


def statistics_events(body,league,season,division,names):
    try:
        frame = pd.read_csv(io.BytesIO(body))
    except (ValueError,UnicodeError):
        raise QualificationError('MALFORMED_STATISTICS') from None
    required = ['Div','Date','HomeTeam','AwayTeam','FTHG','FTAG','HC','AC']
    check(set(required) <= set(frame), 'MISSING_REQUIRED_STATISTIC_FIELDS')
    check(len(frame)>0, 'EMPTY_STATISTICS')
    check(frame.Div.eq(division).all(), 'WRONG_LEAGUE')
    events, joins = [],set()
    for ordinal,row in frame.iterrows():
        date = pd.to_datetime(row.Date,dayfirst=True,errors='raise').date()
        from src.utils.naming import calculate_season
        check(calculate_season(date,league)==season,'STALE_SEASON')
        mapped = []
        for field in ('HomeTeam','AwayTeam'):
            candidates = names.get(token(row[field]),set())
            check(len(candidates)==1,'UNMAPPED_OR_AMBIGUOUS_STATISTICS_TEAM')
            mapped.append(next(iter(candidates)))
        values = [row[k] for k in ('FTHG','FTAG','HC','AC')]
        check(all(pd.notna(x) and float(x)>=0 and float(x).is_integer() for x in values), 'MISSING_OR_INVALID_STATISTICS')
        key = (date.isoformat(),*mapped)
        check(key not in joins,'DUPLICATE_STATISTICS_FIXTURE')
        joins.add(key)
        events.append({'fixture_date':key[0],'home_team':mapped[0],'away_team':mapped[1],
            'home_goals':int(values[0]),'away_goals':int(values[1]),'home_corners':int(values[2]),
            'away_corners':int(values[3]),'row_ordinal':int(ordinal),'league':league,'season':season})
    return events


def source_bytes(source):
    check(source['status']=='RETRIEVED' and source['http_status']==200,
          'STATISTICS_HTTP_ERROR' if source['request']['url'].endswith('.csv') else 'PRIMARY_HTTP_ERROR')
    path = (ROOT/source['raw']['path']).resolve()
    check(path.is_relative_to(ROOT/'baselines/phase10'),'SOURCE_PATH_ESCAPE')
    check(sha256_file(path)==source['raw']['sha256'],'RAW_PAYLOAD_DRIFT')
    check(instant(source['started_at']) <= instant(source['retrieved_at']), 'INVALID_RETRIEVAL_ORDER')
    return path.read_bytes()


def payload(source):
    try:
        return json.loads(source_bytes(source))
    except (ValueError,UnicodeError) as exc:
        if isinstance(exc,QualificationError):
            raise
        raise QualificationError('MALFORMED_PRIMARY_JSON') from None


def join_key(row):
    return (row['fixture_date'],row['home_team'],row['away_team'])


def assess_sources(index, *, frozen, names, as_of):
    """Pure qualification, replayed offline against the archived provider bytes."""
    league, season = index['league'],index['requested_season']
    sources = index['sources']
    check(league in LEAGUES and len(sources)==6,'SOURCE_INDEX_SCOPE_MISMATCH')
    from src.data_pipeline.build_matches_dataset import DIV_MAP
    from src.utils.naming import calculate_season
    check(DIV_MAP.get(index['division'])==league, 'WRONG_LEAGUE')
    check(season==calculate_season(instant(as_of),league),'STALE_SEASON')
    for i,source in enumerate(sources):
        year = season-1+i//3
        request = source['request']
        if i%3 != 2:
            endpoint = 'matches' if i%3 == 0 else 'teams'
            check(request == {'url':f'https://api.football-data.org/v4/competitions/{league}/{endpoint}',
                              'params':{'season':year}}, 'SOURCE_REQUEST_SCOPE_MISMATCH')
        else:
            expected = f'https://www.football-data.co.uk/mmz4281/{year%100:02d}{(year+1)%100:02d}/{index["division"]}.csv'
            check(request == {'url':expected,'params':{}}, 'SOURCE_REQUEST_SCOPE_MISMATCH')
        check(instant(source['retrieved_at'])<=instant(as_of),'LATE_RETRIEVAL')
    blockers, mappings, events, statistics, source_summaries = [],{},[],[],[]
    for offset,year in ((0,season-1),(3,season)):
        teams = {}
        try:
            teams, failures = map_teams(payload(sources[offset+1]),league,year,names)
            mappings[str(year)] = teams
            blockers.extend(failures)
        except (KeyError,TypeError,ValueError) as exc:
            blockers.append({'code':str(exc) if isinstance(exc,QualificationError) else 'MALFORMED_TEAM_PAYLOAD','season':year})
        try:
            raw = payload(sources[offset])
            source_summaries.append({'season':year,'reported_fixture_count':raw.get('resultSet',{}).get('count'),
                'reported_completed_count':raw.get('resultSet',{}).get('played'),
                'outside_reported_season_bounds':[{'provider_fixture_id':e['id'],'kickoff_at':e['utcDate'],
                    'season_start':e['season']['startDate'],'season_end':e['season']['endDate']} for e in raw.get('matches',[])
                    if not e['season']['startDate']<=e['utcDate'][:10]<=e['season']['endDate']]})
            events.extend(primary_events(raw,league,year,teams,sources[offset]['retrieved_at']))
        except (KeyError,TypeError,ValueError) as exc:
            blockers.append({'code':str(exc) if isinstance(exc,QualificationError) else 'MALFORMED_MATCH_PAYLOAD','season':year})
        try:
            statistics.extend(statistics_events(source_bytes(sources[offset+2]),league,year,index['division'],names))
        except (KeyError,TypeError,ValueError) as exc:
            blockers.append({'code':str(exc) if isinstance(exc,QualificationError) else 'MALFORMED_STATISTICS','season':year,
                             'source':sources[offset+2].get('raw'), 'http_status':sources[offset+2].get('http_status')})
    prior,current = mappings.get(str(season-1),{}),mappings.get(str(season),{})
    promoted = [current[k] for k in sorted(set(current)-set(prior))]
    relegated = [prior[k] for k in sorted(set(prior)-set(current))]
    continuity = {'promoted':promoted,'relegated':relegated,'identity_basis':'exact provider team ID set difference',
                  'membership_verified':bool(prior and current),'historical_mapping_verified':all(t['status']=='EXACT_REPOSITORY_ALIAS' for t in promoted)}
    if not prior or not current:
        blockers.append({'code':'PROMOTED_MEMBERSHIP_UNPROVEN'})
    stat_index = {join_key(e):e for e in statistics}
    frozen_index = {(str(r.match_date)[:10],r.home_team,r.away_team):r for r in frozen.itertuples()}
    overlap, composed, missing, conflicts, unresolved = [],[],[],[],[]
    for e in events:
        if e['status']!='FINISHED':
            continue
        key = join_key(e)
        old,stat = frozen_index.get(key),stat_index.get(key)
        if old is not None:
            fields = {'home_goals':e['home_goals']==old.home_goals,'away_goals':e['away_goals']==old.away_goals}
            if stat:
                fields.update({f:stat[f]==getattr(old,f) for f in ('home_goals','away_goals','home_corners','away_corners')})
            overlap.append({'fixture_id':e['fixture_id'],'key':list(key),'fields':fields,'statistics_present':stat is not None})
            if not all(fields.values()):
                conflicts.append(e['fixture_id'])
        elif e['fixture_date'] <= str(frozen.match_date.max())[:10]:
            unresolved.append(e['fixture_id'])
        if stat and [e['home_goals'],e['away_goals']] != [stat['home_goals'],stat['away_goals']]:
            conflicts.append(e['fixture_id'])
        if e['fixture_date'] > str(frozen.match_date.max())[:10]:
            if stat is None:
                missing.append(e['fixture_id'])
            else:
                source_offset = 0 if e['season']==season-1 else 3
                authority = {field:{'source':sources[source_offset if 'goals' in field else source_offset+2]['raw'],
                    'provider_fixture_id':e['provider_fixture_id'] if 'goals' in field else None,
                    'row_ordinal':stat['row_ordinal'] if 'corners' in field else None,
                    'retrieved_at':sources[source_offset if 'goals' in field else source_offset+2]['retrieved_at']}
                    for field in ('home_goals','away_goals','home_corners','away_corners')}
                composed.append({**e, 'home_corners':stat['home_corners'],'away_corners':stat['away_corners'],
                                 'statistics_row':stat['row_ordinal'],'field_provenance':authority})
    if conflicts:
        blockers.append({'code':'SOURCE_SEMANTICS_INCOMPATIBLE','fixtures':sorted(set(conflicts))})
    if unresolved:
        blockers.append({'code':'HISTORICAL_PRIMARY_JOIN_UNPROVEN','fixtures':unresolved})
    if missing:
        blockers.append({'code':'MISSING_REQUIRED_CURRENT_STATISTICS','fixtures':missing})
    if not overlap or not any(o['statistics_present'] for o in overlap):
        blockers.append({'code':'HISTORICAL_STATISTIC_OVERLAP_UNPROVEN'})
    upcoming = [e for e in events if e['season']==season and e['status'] in ('TIMED','SCHEDULED') and instant(e['kickoff_at'])>instant(as_of)]
    if not upcoming:
        blockers.append({'code':'UPCOMING_CAUSAL_FIXTURES_UNPROVEN'})
    if not any(e['season']==season and e['status']=='FINISHED' for e in events):
        blockers.append({'code':'CURRENT_COMPLETED_COVERAGE_UNPROVEN'})
    return {'league':league,'season':season,'as_of':as_of,'blockers':blockers,'team_mappings':mappings,
            'promoted_team_evidence':continuity,'events':events,'statistics':statistics,'historical_overlap':overlap,
            'raw_primary_coverage':source_summaries,
            'composed_observations':composed,'upcoming':sorted(upcoming,key=lambda e:(e['kickoff_at'],e['fixture_id'])),
            'qualified':not blockers,'source_index_sha256':sha256_json(index)}


def base_frame(league):
    frame = pd.read_csv(ROOT/'data/processed/matches.csv',low_memory=False)
    frame = frame.loc[frame.league.eq(league)].copy().rename(columns={
        'match_date':'date','home_goals':'home_score','away_goals':'away_score','match_hash':'match_id'})
    frame['date'] = pd.to_datetime(frame.date,utc=True)
    frame['status'] = 'FINISHED'
    return frame


def pipeline(frame,league):
    """Use the Phase 7E temporary-output pattern with an explicit league."""
    import tempfile
    import src.features.pipeline as module
    from src.features.pipeline import FeaturePipeline
    with tempfile.TemporaryDirectory(prefix='ffps-phase10-') as temp:
        original = module.DATA_DIR
        try:
            module.DATA_DIR = Path(temp)
            return FeaturePipeline().run(match_data=frame,league=league)
        finally:
            module.DATA_DIR = original


def feature_reproduction(league,names):
    import numpy as np
    generated = pipeline(base_frame(league),league)
    frozen = pd.read_csv(ROOT/'data/features/feature_matrix.csv',low_memory=False)
    frozen = frozen.loc[frozen.league.eq(league)]
    missing = sorted(set(names)-set(generated))
    if missing:
        return {'reproduced':False,'code':'FEATURE_SCHEMA_MISMATCH','missing_features':missing}
    joined = generated.merge(frozen[['match_hash',*names]],left_on='match_id',right_on='match_hash',suffixes=('_new','_old'))
    differences, eligible_differences = {}, {}
    eligible = np.isfinite(joined[[feature+'_old' for feature in names]].to_numpy(float)).all(axis=1)
    for feature in names:
        left,right = (joined[feature+s].to_numpy(float) for s in ('_new','_old'))
        same = np.isclose(left,right,rtol=0,atol=1e-12,equal_nan=True)
        if not same.all():
            differences[feature] = int((~same).sum())
        if not same[eligible].all():
            eligible_differences[feature] = int((~same[eligible]).sum())
    reproduced = len(generated)==len(joined)==len(frozen) and bool(eligible.any()) and not eligible_differences
    return {'reproduced':reproduced,'code':'VERIFIED' if reproduced else 'FEATURE_SEMANTICS_INCOMPATIBLE',
        'generated_rows':len(generated),'frozen_rows':len(frozen),'joined_rows':len(joined),
        'all_row_differing_features':differences,'model_eligible_differing_features':eligible_differences,
        'model_eligible_rows':int(eligible.sum()),'excluded_frozen_nonfinite_rows':int((~eligible).sum()),
        'eligibility_rule':'frozen ordered features all finite, identical to Phase 9 pre-partition cold-start exclusion; no imputation',
        'schema_sha256':sha256_json(names),
        'source':identity(ROOT/'data/features/feature_matrix.csv')}


def target_matrix(assessment,target,*,as_of,source_index):
    """Rebuild from pre-existing historical bytes plus observed authoritative rows."""
    import numpy as np
    from src.predictions.provenance import build_file_snapshot,source_event_index
    check(assessment['qualified'],'CURRENT_DATA_UNQUALIFIED')
    check(target in assessment['upcoming'],'UNKNOWN_TARGET_FIXTURE')
    check(instant(as_of)<instant(target['kickoff_at']),'RETROSPECTIVE_RECONSTRUCTION')
    rows = []
    for e in assessment['composed_observations']:
        check(instant(e['retrieved_at'])<=instant(as_of),'LATE_RETRIEVAL')
        if instant(e['kickoff_at'])>=instant(as_of):
            continue
        rows.append({'match_id':e['fixture_id'],'date':e['kickoff_at'],'league':e['league'],'season':e['season'],
            'home_team':e['home_team'],'away_team':e['away_team'],'home_score':e['home_goals'],'away_score':e['away_goals'],
            'home_corners':e['home_corners'],'away_corners':e['away_corners'],'status':'FINISHED'})
    rows.append({'match_id':target['fixture_id'],'date':target['kickoff_at'],'league':target['league'],'season':target['season'],
        'home_team':target['home_team'],'away_team':target['away_team'],'home_score':np.nan,'away_score':np.nan,
        'home_corners':np.nan,'away_corners':np.nan,'status':target['status']})
    composite = pd.concat([base_frame(target['league']),pd.DataFrame(rows)],ignore_index=True)
    composite['date'] = pd.to_datetime(composite.date,utc=True)
    matrix = pipeline(composite,target['league'])
    matrix['match_hash'] = matrix.match_id
    artifacts = [dict(identity(ROOT/'data/processed/matches.csv'),role='frozen_causal_history',load_order=0)]
    for source in source_index['sources']:
        artifacts.append(dict(source['raw'],role='archived_current_provider_payload',load_order=len(artifacts)))
    matrix.attrs['ffps_data_snapshot'] = build_file_snapshot(frame=matrix,artifacts=artifacts,source_type='phase10_causal_multi_source')
    matrix.attrs['ffps_source_events'] = source_event_index(composite)
    target_rows = matrix.loc[matrix.match_id.eq(target['fixture_id'])]
    check(len(target_rows)==1,'NONUNIQUE_FEATURE_TARGET')
    fixture = target_rows.iloc[0]
    return matrix,fixture


def validate_features(matrix,fixture,names,as_of):
    import numpy as np
    check(set(names)<=set(matrix) and len(set(names))==len(names),'FEATURE_SCHEMA_MISMATCH')
    check(np.isfinite(fixture[names].to_numpy(float)).all(),'NONFINITE_OR_INSUFFICIENT_HISTORY')
    columns = [c for c in matrix if 'max_source_date' in c]
    check(bool(columns),'FEATURE_CAUSAL_CUTOFF_UNPROVEN')
    for column in columns:
        if pd.notna(fixture[column]):
            source_at = pd.to_datetime(fixture[column],utc=True)
            check(source_at < pd.to_datetime(fixture.date,utc=True) and
                  source_at <= pd.to_datetime(as_of,utc=True),'FUTURE_FEATURE_SOURCE')
    return {'fixture_id':fixture.match_id,'ordered_features':names,'feature_schema_sha256':sha256_json(names),
            'feature_vector_sha256':sha256_json(fixture[names].to_numpy(float).tolist()),'cutoff_columns':columns,'finite':True}


def qualify(path,league,attempt='network-retry'):
    from src.ml.league_bundle_activation import LeagueBundleController
    check(league in LEAGUES,'WRONG_LEAGUE')
    path = open_run(path)
    league_dir = path/league
    directory = league_dir/'assessments'/f'assessment-{len(list((league_dir/"assessments").glob("assessment-*")))+1:03d}'
    index_path = league_dir/f'raw_source_index_{attempt}.json'
    index = read(index_path)
    frozen = pd.read_csv(ROOT/'data/processed/matches.csv',low_memory=False)
    frozen = frozen.loc[frozen.league.eq(league)]
    as_of = now()
    assessment = assess_sources(index,frozen=frozen,names=exact_names(league),as_of=as_of)
    controller = LeagueBundleController(PHASE9/league/'deployment_specification.json',league=league)
    active,candidate = (controller.bundle(b) for b in (controller.legacy_id,controller.candidate_id))
    check(active['features']==candidate['features'],'ACTIVE_CANDIDATE_SCHEMA_MISMATCH')
    try:
        reproduction = feature_reproduction(league,active['features'])
    except Exception as exc:
        reproduction = {'reproduced':False,'code':'FEATURE_PIPELINE_INCOMPATIBLE',
                        'error_type':type(exc).__name__,'detail':str(exc)}
    proofs,feature_failures = [],[]
    if not reproduction['reproduced']:
        feature_failures.append({'code':reproduction['code']})
    if assessment['qualified'] and reproduction['reproduced']:
        # Every presently eligible fixture is assessed; no outcome-based selection.
        for target in assessment['upcoming']:
            try:
                matrix,fixture = target_matrix(assessment,target,as_of=as_of,source_index=index)
                proofs.append(validate_features(matrix,fixture,active['features'],as_of))
            except (ValueError,KeyError,TypeError) as exc:
                feature_failures.append({'fixture_id':target['fixture_id'],'code':str(exc) if isinstance(exc,QualificationError) else 'FEATURE_PIPELINE_INCOMPATIBLE'})
    else:
        feature_failures.append({'code':'CURRENT_FEATURE_CONTINUITY_BLOCKED_BY_SOURCE_QUALIFICATION'})
    result = {'schema_version':'ffps.phase10_qualification.v1','league':league,'as_of':as_of,
        'assessment':assessment,'historical_feature_reproduction':reproduction,'feature_proofs':proofs,
        'feature_failures':feature_failures,'feature_readiness':assessment['qualified'] and not feature_failures and bool(proofs),
        'active_bundle_id':controller.legacy_id,'candidate_bundle_id':controller.candidate_id,
        'active_sha256':active['aggregate_sha256'],'candidate_sha256':candidate['aggregate_sha256'],
        'ordered_features':active['features'],'feature_schema_sha256':sha256_json(active['features']),
        'source_index':identity(index_path),
        'phase9_deployment':identity(PHASE9/league/'deployment_specification.json'),
        'promotion_authorized':False}
    result['qualification_sha256'] = sha256_json(result)
    write(directory/'qualification.json',result)
    write(directory/'team_mapping_evidence.json',{'league':league,'mappings':assessment['team_mappings'],
        'promoted_relegated':assessment['promoted_team_evidence'],'normalization_source':identity(ROOT/'src/utils/naming.py')})
    write(directory/'feature_continuity_evidence.json',{'league':league,'historical':reproduction,
        'current':proofs,'failures':feature_failures,'ready':result['feature_readiness']})
    if result['feature_readiness']:
        contract = {'schema_version':'ffps.phase10_current_data_contract.v1','league':league,'season':assessment['season'],
            'qualification':identity(directory/'qualification.json'),'frozen_at':now(),
            'feature_schema_sha256':result['feature_schema_sha256'],'source_index':result['source_index'],
            'authority':{'fixture_identity':'football-data.org','kickoff':'football-data.org','goals':'football-data.org',
                         'corners':'football-data.co.uk'},'provider_publication_policy':'reported lastUpdated retained; retrieval is availability upper bound only',
            'cohort_policy':'all future timed or scheduled current-season league fixtures; reject incomplete features; no retrospective pairs',
            'promotion_authorized':False}
        contract['contract_sha256'] = sha256_json(contract)
        write(directory/'current_data_contract.json',contract)
    return {'league':league,'ready':result['feature_readiness'],
            'blockers':sorted({b['code'] for b in assessment['blockers']+feature_failures})}


def latest_qualification(path,league):
    files = sorted((run_path(path)/league/'assessments').glob('assessment-*/qualification.json'))
    check(bool(files),'LEAGUE_NOT_ASSESSED')
    return files[-1]


def qualify_all(path,attempt):
    from src.config.leagues import ACTIVE_LEAGUES
    check(set(ACTIVE_LEAGUES)-{'PL'}==set(LEAGUES),'PRODUCTION_LEAGUE_INVENTORY_CHANGED')
    path = open_run(path)
    reports = []
    for league in LEAGUES:
        try:
            reports.append(qualify(path,league,attempt))
        except Exception as exc:
            failure = {'league':league,'ready':False,'code':'QUALIFICATION_EXECUTION_FAILED','error_type':type(exc).__name__}
            write(run_path(path)/league/'execution_failures'/(sha256_json([now(),failure])+'.json'),failure)
            reports.append(failure)
    return reports

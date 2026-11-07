"""Read-only, league-scoped field overlap qualification on repository sources."""
from __future__ import annotations

import pandas as pd

from src.evaluation.multi_league_integrity import ROOT, identity, write
from src.predictions.provenance import sha256_json


def qualify_provider(directory, row):
    from src.data_pipeline.build_matches_dataset import CanonicalDatasetBuilder
    builder = CanonicalDatasetBuilder()
    league = row['league']
    raw = pd.read_csv(ROOT/'data/processed/matches.csv', low_memory=False)
    raw = raw.loc[raw.league.eq(league)].copy()
    keys = ['league','match_date','home_team','away_team']
    fields = ['home_goals','away_goals','home_corners','away_corners']
    observations, sources, problems = [], [], []
    canonical = raw.set_index(keys)
    for path in sorted((ROOT/'data/historical'/league).glob('*.csv')):
        try:
            parsed, coverage = builder._parse_and_normalise(path, league)
            if parsed is None:
                problems.append({'path':str(path.relative_to(ROOT)), 'classification':'UNPARSEABLE_SOURCE'})
                continue
            parsed['match_date'] = parsed.match_date.dt.strftime('%Y-%m-%d')
            src = identity(path)
            sources.append(src)
            for ordinal, event in parsed.iterrows():
                key = tuple(event[k] for k in keys)
                if key not in canonical.index:
                    continue
                frozen = canonical.loc[key]
                if isinstance(frozen,pd.DataFrame):
                    problems.append({'fixture':list(key),'classification':'AMBIGUOUS_CANONICAL_JOIN'})
                    continue
                provenance = {}
                for field in fields:
                    a,b = event[field],frozen[field]
                    equal = bool(pd.notna(a) and pd.notna(b) and float(a) == float(b) and float(a).is_integer() and float(a) >= 0)
                    provenance[field] = {'raw_source_sha256':src['sha256'],'normalized_row_index':int(ordinal),
                        'source_value':float(a) if pd.notna(a) else None,'canonical_value':float(b) if pd.notna(b) else None,'exact_equal':equal}
                observations.append({'league':league,'fixture':list(key),'match_hash':frozen['match_hash'],
                                      'home_away_orientation':'exact ordered canonical team join','fields':provenance})
        except Exception as exc:
            problems.append({'path':str(path.relative_to(ROOT)), 'classification':'SOURCE_INSPECTION_FAILED','error':f'{type(exc).__name__}: {exc}'})
    counts = {field:{'overlap_rows':len(observations),'mismatches':sum(not o['fields'][field]['exact_equal'] for o in observations)} for field in fields}
    authorities = [{'league':league,'source':'football-data.co.uk:csv','fields':[field],**count,
                    'qualification':'EXACT_HISTORICAL_OVERLAP' if count['overlap_rows']>0 and count['mismatches']==0 else 'SOURCE_SEMANTICS_UNQUALIFIED',
                    'scope':'frozen historical snapshots only; current publication timing not proven'} for field,count in counts.items()]
    write(directory/'provider_overlap_rows.json', {'league':league,'sources':sources,'observations':observations,'problems':problems})
    result = {'league':league,'primary_fixture_result_source':'football-data.org integration; current league response not captured here',
              'required_statistics_source':'football-data.co.uk:csv','field_authorities':authorities,
              'overlap_evidence':identity(directory/'provider_overlap_rows.json'),
              'canonical_unique_fixtures_matched':len({o['match_hash'] for o in observations}), 'canonical_fixture_count':len(raw),
              'latest_causal_canonical_event':row['latest_causal_data_row'],
              'current_season_fixture_coverage':'UNQUALIFIED','current_season_completed_result_coverage':'UNQUALIFIED',
              'current_season_corner_coverage':'UNQUALIFIED','provider_publication_latency':'UNPROVEN',
              'team_mapping':'exact existing central normalization; current-season promoted-team mapping not independently qualified',
              'normalization_source':identity(ROOT/'src/utils/naming.py'),
              'feature_readiness':False,'status':'CURRENT_SEASON_DATA_BLOCKED',
              'blockers':['No league-scoped current-season raw response and retrieval timestamp in Phase 9 evidence.',
                          'Historical overlap does not prove current coverage, publication latency, or pre-kickoff feature availability.'],
              'production_data_modified':False,'problems':problems}
    result['qualification_sha256'] = sha256_json(result)
    write(directory/'data_continuity.json',result)
    return result

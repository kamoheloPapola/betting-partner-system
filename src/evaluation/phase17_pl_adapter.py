"""Offline PL fixture adapter for the already-authorized Phase 7P source lineage.

This module does not fetch data or publish a prospective observation. Every
normalization consumes an immutable football-data.org response and its sidecar.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation import phase12_prospective as p
from src.evaluation import phase7d_data_continuity as pl

PHASE7D = p.ROOT / 'baselines/phase7d/phase7d-approved-provider-pl-continuity-79110f9a-20260829-v1'
PHASE7E = p.ROOT / 'baselines/phase7e/phase7e-football-data-co-uk-corners-79110f9a-20260830-v1'
SOURCE_2026 = PHASE7D / 'raw_provider_snapshot/football-data-org-pl-2026.source.json'
RAW_2026 = PHASE7D / 'raw_provider_snapshot/football-data-org-pl-2026.response.json'
ALIAS = PHASE7E / 'team_alias_extension.json'
AUTHORITY = PHASE7E / 'field_authority_matrix.json'
STATUSES = frozenset(('TIMED', 'SCHEDULED', 'FINISHED', 'POSTPONED', 'CANCELLED',
                      'CANCELED', 'ABANDONED', 'SUSPENDED'))


def _instant(value):
    p.check(isinstance(value, str) and (value.endswith('Z') or value.endswith('+00:00')),
            'PL_TIMESTAMP_NOT_EXPLICIT_UTC')
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    p.check(result.tzinfo is not None, 'PL_TIMESTAMP_NO_ZONE')
    return result.astimezone(timezone.utc).isoformat()


def load_raw(source_path=SOURCE_2026, raw_path=RAW_2026):
    source_path, raw_path = Path(source_path), Path(raw_path)
    source = json.loads(source_path.read_bytes())
    payload = raw_path.read_bytes()
    p.check(source['schema_version'] == pl.RAW_SOURCE_SCHEMA_VERSION and
            source['provider'] == pl.PROVIDER_ID and source['competition'] == 'PL' and
            source['request_url'] == f"https://api.football-data.org/v4/competitions/PL/matches?season={source['season']}" and
            source['response_sha256'] == p.source.sha256_file(raw_path) and
            source['response_size_bytes'] == len(payload) and
            source['path'] == raw_path.resolve().relative_to(p.ROOT.resolve()).as_posix(),
            'PL_RAW_SOURCE_IDENTITY_MISMATCH')
    _instant(source['retrieved_at'])
    document = json.loads(payload)
    p.check(document['competition']['code'] == 'PL' and isinstance(document['matches'], list),
            'PL_COMPETITION_OR_MATCHES_INVALID')
    return source, document


def _team(value, season, aliases):
    p.check(isinstance(value, dict) and isinstance(value.get('id'), int) and
            value['id'] > 0 and isinstance(value.get('name'), str) and value['name'],
            'PL_TEAM_IDENTITY_INVALID')
    matches = [row for row in aliases if row['primary_provider_team_id'] == str(value['id']) and
               row['primary_provider_name'] == value['name']]
    p.check(len(matches) <= 1, 'PL_ALIAS_COLLISION')
    if matches:
        p.check(season == 2026, 'PL_ALIAS_OUTSIDE_SCOPE')
        return matches[0]['canonical_team']
    return pl.strict_normalize_team(value['name'], season=season)


def normalize_event(raw_event, source, aliases):
    p.check(raw_event['competition']['code'] == 'PL' and
            str(raw_event['season']['startDate']).startswith(str(source['season']) + '-') and
            isinstance(raw_event.get('id'), int) and raw_event['id'] > 0,
            'PL_FIXTURE_COMPETITION_OR_IDENTITY_INVALID')
    season = source['season']
    kickoff = _instant(raw_event.get('utcDate'))
    home = _team(raw_event.get('homeTeam'), season, aliases)
    away = _team(raw_event.get('awayTeam'), season, aliases)
    home_id, away_id = str(raw_event['homeTeam']['id']), str(raw_event['awayTeam']['id'])
    p.check(home_id != away_id and home != away, 'PL_HOME_AWAY_COLLISION')
    status = raw_event.get('status')
    p.check(status in STATUSES, 'PL_UNSUPPORTED_STATUS')
    score = raw_event.get('score')
    p.check(isinstance(score, dict) and isinstance(score.get('fullTime'), dict),
            'PL_SCORE_STRUCTURE_INVALID')
    goals = (score['fullTime'].get('home'), score['fullTime'].get('away'))
    if status == 'FINISHED':
        p.check(all(type(x) is int and x >= 0 for x in goals), 'PL_FINAL_SCORE_INVALID')
    else:
        p.check(goals == (None, None), 'PL_PREMATCH_OR_TERMINAL_SCORE_CONFLICT')
    date = kickoff[:10]
    return {'league': 'PL', 'season': season,
            'fixture_id': f"PL_football_data_org_{raw_event['id']}",
            'provider_fixture_id': str(raw_event['id']),
            'home_id': home_id, 'away_id': away_id,
            'home_team': home, 'away_team': away,
            'kickoff_at': kickoff, 'status': status,
            'home_goals': goals[0], 'away_goals': goals[1],
            'phase7p_canonical_fixture_id': pl.canonical_fixture_identity(
                league='PL', season=pl.season_label(season), fixture_date=date,
                home_team=home, away_team=away)}


def normalize_snapshot(source_path=SOURCE_2026, raw_path=RAW_2026):
    source, document = load_raw(source_path, raw_path)
    alias_doc = p.read(ALIAS)
    p.check(alias_doc['schema_version'] == 'ffps.phase7e_team_alias_extension.v1' and
            alias_doc['scope'] == 'evaluation_only_post_protocol_PL_2026' and
            alias_doc['production_mapping_changed'] is False, 'PL_ALIAS_AUTHORITY_INVALID')
    matrix = p.read(AUTHORITY)
    for name in ('competition', 'home_team', 'away_team', 'kickoff_at', 'status',
                 'home_goals', 'away_goals'):
        p.check(matrix['fields'][name]['primary'] == pl.PROVIDER_ID,
                'PL_FIELD_AUTHORITY_MISMATCH')
    rows = [normalize_event(event, source, alias_doc['aliases']) for event in document['matches']]
    ids = [row['provider_fixture_id'] for row in rows]
    p.check(len(ids) == len(set(ids)), 'PL_DUPLICATE_PROVIDER_FIXTURE')
    return source, rows


def normalize_versions(items):
    """Replay archived versions; reject equal-time disagreement and identity drift."""
    rows = []
    for source_path, raw_path, provider_id in items:
        source, events = normalize_snapshot(source_path, raw_path)
        matches = [row for row in events if row['provider_fixture_id'] == str(provider_id)]
        p.check(len(matches) == 1, 'PL_FIXTURE_MISSING_OR_AMBIGUOUS')
        rows.append({'retrieved_at': _instant(source['retrieved_at']), 'fixture': matches[0],
                     'source': p.identity(Path(raw_path)), 'source_metadata': p.identity(Path(source_path))})
    rows.sort(key=lambda row: (row['retrieved_at'], row['source']['sha256']))
    for first, second in zip(rows, rows[1:]):
        p.check(first['fixture']['fixture_id'] == second['fixture']['fixture_id'] and
                all(first['fixture'][key] == second['fixture'][key] for key in
                    ('league','season','provider_fixture_id','home_id','away_id','home_team','away_team')),
                'PL_REVISION_IDENTITY_DRIFT')
        p.check(first['retrieved_at'] != second['retrieved_at'] or
                first['fixture'] == second['fixture'], 'PL_EQUAL_TIME_REVISION_CONFLICT')
    return rows

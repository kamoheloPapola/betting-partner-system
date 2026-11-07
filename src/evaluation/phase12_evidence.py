"""Complete-cohort Phase 12 reporting and offline verification."""
from pathlib import Path
import shutil

from src.evaluation import phase12_prospective as p

SOURCES = ('src/evaluation/phase12_prospective.py', 'src/evaluation/phase12_evidence.py',
           'scripts/run_phase12_prospective.py', 'tests/unit/test_phase12_prospective.py',
           'docs/phase12-prospective-operations.md')


def latest(path, league):
    items = [p.read(item) for item in (p.run_path(path) / league / 'checkpoints').glob('*.json')]
    p.check(items, 'NO_OPERATIONAL_CHECKPOINT:' + league)
    return max(items, key=lambda row: row['as_of'])


def status(path):
    return {'leagues': [latest(path, league) for league in p.source.LEAGUES], 'promotion_authorized': False}


def checkpoint(path):
    """Replay and refresh observability without a source request or prediction."""
    rows = []
    for league in p.source.LEAGUES:
        with p.state_lock(p.run_path(path) / league / 'operation.lock'):
            previous = latest(path, league)
            rows.append(p.write_checkpoint(path, league, discovery=previous['discovery'],
                                           temporarily_blocked=previous['temporary_source_block']))
    return {'leagues': rows, 'promotion_authorized': False}


def integrity(path):
    spec = p.specification(path)
    before = p.read(p.frozen.reference(spec['protected_before']))
    changed = [key for key, value in before.items() if not (p.ROOT / key).is_file() or
               p.source.sha256_file(p.ROOT / key) != value]
    p.check(not changed, 'PROTECTED_FILE_MUTATION:' + ','.join(changed))
    for base, inventory in spec['production_inventory'].items():
        p.check(p.source._tree_snapshot(p.ROOT / base) == inventory, 'PRODUCTION_OR_PL_INVENTORY_CHANGED:' + base)
    return {'protected_count': len(before), 'unchanged': True, 'changed': changed,
            'phase11_original_protected_count': len(p.read(p.PHASE11 / 'protected_before.json')),
            'phase11_evidence_unchanged': True, 'production_models_and_pointers_unchanged': True,
            'phase9_candidates_unchanged': True, 'pl_phase7p_unchanged': True,
            'training': False, 'calibration': False, 'promotion': False}


def verify_checkpoint(row, context, prefix_count):
    p.frozen.sealed(row, 'record_sha256')
    protocol, _, _, all_pairs, all_settlements, _, events = context
    count = row['journal_count']
    p.check(count <= len(events), 'CHECKPOINT_TAIL_MISSING')
    members = events[:count]
    pairs = all_pairs[:prefix_count] + [e['payload'] for e in members if e['kind'] == 'pair']
    settlements = [e['payload'] for e in members if e['kind'] == 'settlement']
    p.check(row['cohort_pair_hashes'] == [v['pair_sha256'] for v in pairs] and
            row['cohort_settlement_hashes'] == [v['settlement_sha256'] for v in settlements], 'CHECKPOINT_COHORT_CHANGED')
    p.check(row['league'] == protocol['league'] and row['protocol_sha256'] == protocol['protocol_sha256'],
            'CHECKPOINT_LEAGUE_OR_PROTOCOL_MISMATCH')
    controller = context[1]
    for label, bundle_id in (('active', controller.legacy_id), ('candidate', controller.candidate_id)):
        bundle = controller.bundle(bundle_id)
        p.check(row[label] == {key: bundle[key] for key in ('bundle_id', 'aggregate_sha256')},
                'CHECKPOINT_BUNDLE_IDENTITY_MISMATCH')
    p.check(row['pairs'] == len(pairs) and row['settled'] == len(settlements) and
            row['pending'] == len(pairs)-len(settlements), 'CHECKPOINT_COUNT_MISMATCH')
    p.check(all(p.instant(event['created_at']) <= p.instant(row['as_of']) for event in members) and
            all(p.instant(event['created_at']) > p.instant(row['as_of']) for event in events[count:]),
            'CHECKPOINT_OMITS_ADMITTED_EVIDENCE')
    incidents = [e['payload'] for e in members if e['kind'] == 'incident']
    p.check(row['incidents'] == incidents, 'INCIDENT_LEDGER_MISMATCH')
    evaluation = p.evaluate(protocol, pairs, settlements, row['as_of'])
    p.check(row['evaluation'] == evaluation, 'COHORT_METRICS_RECOMPUTATION_MISMATCH')
    state = p.state_for(evaluation, pairs, incidents, row['as_of'], row['temporary_source_block'])
    p.check(row['scientific_state'] == state and row['promotion'] == 'NOT_AUTHORIZED', 'SCIENTIFIC_STATE_DRIFT')
    discovery = row['discovery']
    if discovery:
        p.current_events(discovery['source'], protocol, context[2])


def verify(path, *, report=None):
    path = p.run_path(path)
    preserved = integrity(path)
    spec = p.specification(path)
    results, contexts = [], {}
    for league in p.source.LEAGUES:
        context = p.load_cohort(path, league)
        contexts[league] = context
        for item in (path / league / 'checkpoints').glob('*.json'):
            verify_checkpoint(p.read(item), context, len(spec['leagues'][league]['prefix']))
        p.check(not list((path / league / 'failures').glob('*.json')), 'UNRESOLVED_OPERATIONAL_FAILURE:' + league)
        results.append({'league': league, 'pairs': len(context[3]), 'settled': len(context[4]), 'replay': 'VERIFIED'})
    reports = [report] if report else sorted((path / 'reports').glob('report-*'))
    for directory in reports:
        index_path = directory / 'artifact_index.json'
        if not index_path.exists():
            continue
        index = p.read(index_path)
        p.frozen.sealed(index, 'record_sha256')
        for ref in index['artifacts'] + index['implementation_sources']:
            p.frozen.reference(ref)
        report_rows = p.read(directory / 'multi_league_prospective_status_matrix.json')['leagues']
        p.check([row['league'] for row in report_rows] == list(p.source.LEAGUES), 'REPORT_LEAGUE_INVENTORY_MISMATCH')
        for row in report_rows:
            league = row['league']
            context = contexts[league]
            prefix = spec['leagues'][league]['prefix']
            verify_checkpoint(row, context, len(prefix))
            members = context[-1][:row['journal_count']]
            pairs = context[3][:len(prefix)] + [event['payload'] for event in members if event['kind'] == 'pair']
            settlements = [event['payload'] for event in members if event['kind'] == 'settlement']
            p.check(p.read(directory / league / 'prospective_predictions.json') == {'league': league, 'pairs': pairs},
                    'REPORT_PAIR_SUBSET_OR_REORDER')
            p.check(p.read(directory / league / 'prospective_settlements.json') == {'league': league, 'settlements': settlements},
                    'REPORT_SETTLEMENT_SUBSET_OR_REORDER')
            p.check(p.read(directory / league / 'metrics_reliability.json') == row['evaluation'], 'REPORT_METRICS_MISMATCH')
            cohort = p.read(directory / league / 'cohort_index.json')
            p.check(cohort['prefix'] == prefix and cohort['ordered_pair_hashes'] == row['cohort_pair_hashes'] and
                    cohort['ordered_settlement_hashes'] == row['cohort_settlement_hashes'], 'REPORT_COHORT_INDEX_MISMATCH')
            files = sorted((path / league / 'journal').glob('*.json'))[:row['journal_count']]
            p.check(cohort['ordered_journal'] == [p.identity(item) for item in files], 'REPORT_JOURNAL_SUBSET_OR_REORDER')
        expected = {ref['path'] for ref in index['artifacts'] if (p.ROOT / ref['path']).is_relative_to(directory)}
        actual = {item.relative_to(p.ROOT).as_posix() for item in directory.rglob('*')
                  if item.is_file() and item.name not in ('artifact_index.json', 'offline_verification.json')}
        p.check(expected == actual, 'UNINDEXED_OR_MISSING_REPORT_EVIDENCE')
        # Additional later operations are allowed; each sealed package is immutable.
        tests = p.read(directory / 'tests/classification.json')
        for label in ('focused', 'protected', 'full'):
            p.check(p.evidence11.test_summary(directory / f'tests/{label}.xml') == tests[label], 'TEST_REPORT_MISMATCH')
        inherited = set(p.evidence11.test_summary(directory / 'tests/phase11_full_reference.xml')['failures'])
        new = sorted(set(tests['full']['failures']) - inherited)
        p.check(new == tests['new_unexplained_failures'] and not new and not tests['focused']['failures'] and
                not tests['protected']['failures'], 'NEW_TEST_FAILURE')
    return {'schema_version': 'ffps.phase12_offline_verification.v1', 'verified': True,
            'protected_integrity': preserved, 'leagues': results, 'promotion_performed': False}


def finalize(path):
    path = p.run_path(path)
    verified = verify(path)
    rows = status(path)['leagues']
    directory = path / 'reports' / f"report-{len(list((path / 'reports').glob('report-*')))+1:03d}"
    directory.mkdir(parents=True)
    p.persist_new(directory / 'multi_league_prospective_status_matrix.json', {'leagues': rows})
    p.persist_new(directory / 'protected_hash_comparison.json', verified['protected_integrity'])
    horizon, readiness, rejection_ledger, all_pairs, all_settlements = [], [], [], [], []
    acquisitions, qualified_sources = set(), []
    for league in p.source.LEAGUES:
        context = p.load_cohort(path, league)
        protocol, controller, _, pairs, settlements, _, events = context
        row = next(r for r in rows if r['league'] == league)
        for qualification in {p.digest(pair['qualification']): pair['qualification'] for pair in pairs}.values():
            q = p.read(p.frozen.reference(qualification))
            source_index = p.read(p.frozen.reference(q['source_index']))
            qualified_sources.extend(source_index['sources'])
        by_reason = {}
        for incident in row['incidents']:
            if incident['code'] == 'SOURCE_REFRESH_STARTED':
                acquisitions.add(p.frozen.reference(incident['source']).parent)
            else:
                by_reason.setdefault(incident['code'], set()).add(incident['fixture_id'])
                rejection_ledger.append({'league': league, **incident})
        progress = {'league': league, **row['horizon'], 'admitted_pairs': len(pairs),
                    'settled_pairs': len(settlements), 'unsettled_pairs': len(pairs)-len(settlements),
                    'rejected_fixtures_by_reason': {key: len(value - {None}) for key, value in by_reason.items()},
                    'cohort_completeness': row['evaluation']['cohort_completeness'],
                    'remaining_eligible_horizon': row['discovery']['remaining_eligible_horizon'] if row['discovery'] else None,
                    'remaining_eligible_horizon_basis': 'latest successful provider snapshot; unknown if retrieval blocked'}
        horizon.append(progress)
        readiness.append({'league': league, 'scientific_state': row['scientific_state'],
                          'decision': row['evaluation']['decision'],
                          'phase8_contract_bound': controller.owner.frozen.get('prospective_contract') is not None,
                          'phase8_contract_note': 'Phase8 authorization remains separate; Phase12 changes no deployment specification',
                          'promotion': 'NOT_AUTHORIZED'})
        p.persist_new(directory / league / 'cohort_index.json', {
            'league': league, 'protocol': row['protocol'], 'protocol_sha256': protocol['protocol_sha256'],
            'prefix': p.specification(path)['leagues'][league]['prefix'],
            'ordered_journal': [p.identity(item) for item in sorted((path / league / 'journal').glob('*.json'))],
            'ordered_pair_hashes': [v['pair_sha256'] for v in pairs],
            'ordered_settlement_hashes': [v['settlement_sha256'] for v in settlements]})
        p.persist_new(directory / league / 'metrics_reliability.json', row['evaluation'])
        p.persist_new(directory / league / 'prospective_predictions.json', {'league': league, 'pairs': pairs})
        p.persist_new(directory / league / 'prospective_settlements.json', {'league': league, 'settlements': settlements})
        all_pairs.extend({'league': league, 'fixture_id': v['fixture_id'], 'pair_sha256': v['pair_sha256'],
                          'created_at': v['created_at'], 'kickoff_at': v['kickoff_at'], 'qualification': v['qualification']}
                         for v in pairs)
        all_settlements.extend(settlements)
    for name, value in (('pair_index', all_pairs), ('settlement_index', all_settlements),
                        ('rejection_ledger', rejection_ledger), ('horizon_progress', horizon), ('decision_readiness', readiness)):
        p.persist_new(directory / (name + '.json'), {'records': value})
    source_files = [item for base in [path, *sorted(acquisitions)] for item in base.rglob('*')
                    if item.is_file() and ('raw_sources' in item.parts or 'retrievals' in item.parts)]
    source_refs = {p.digest(source['raw']): source['raw'] for source in qualified_sources}
    source_refs.update({p.digest(p.identity(item)): p.identity(item) for item in source_files})
    p.persist_new(directory / 'source_identity_index.json', {
        'sources': sorted(source_refs.values(), key=lambda ref: ref['path']),
        'qualified_source_records': list({p.digest(item): item for item in qualified_sources}.values()),
        'credentials_recorded': False})
    tests_dir = directory / 'tests'
    tests_dir.mkdir()
    for label in ('focused', 'protected', 'full'):
        shutil.copyfile(p.ROOT / f'tmp/phase12-{label}.xml', tests_dir / f'{label}.xml')
    shutil.copyfile(p.PHASE11 / 'tests/full.xml', tests_dir / 'phase11_full_reference.xml')
    tests = {label: p.evidence11.test_summary(tests_dir / f'{label}.xml') for label in ('focused', 'protected', 'full')}
    inherited = set(p.evidence11.test_summary(tests_dir / 'phase11_full_reference.xml')['failures'])
    tests['inherited_failures'] = sorted(set(tests['full']['failures']) & inherited)
    tests['new_unexplained_failures'] = sorted(set(tests['full']['failures']) - inherited)
    p.check(not tests['new_unexplained_failures'] and not tests['focused']['failures'] and not tests['protected']['failures'],
            'NEW_TEST_FAILURE_BLOCKS_SEAL')
    p.persist_new(tests_dir / 'classification.json', tests)
    text = ['# Phase 12 — Multi-League Prospective Collection, Settlement & Decision Readiness', '',
            'The frozen Phase 11 prefix is unchanged. The authoritative continuation binds every original pair and every ordered league journal entry. No models, calibration, production pointers, Phase 9 candidates, or PL Phase 7P evidence changed.', '',
            '| League | Protocol | Eligible fixtures | Pairs | Settled | Pending | Replay | Active log loss | Candidate log loss | Horizon end | Scientific state | Promotion |',
            '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in rows:
        metrics = row['evaluation']['metrics']
        active = metrics.get('active', {}).get('log_loss', 'Unavailable')
        candidate = metrics.get('candidate', {}).get('log_loss', 'Unavailable')
        eligible = len(row['discovery']['eligible_fixture_ids']) if row['discovery'] else 'Unknown'
        text.append(f"| {row['league']} | {row['protocol_sha256'][:12]} | {eligible} | {row['pairs']} | {row['settled']} | {row['pending']} | {row['replay']} | {active} | {candidate} | {row['horizon']['end_at'][:10]} | {row['scientific_state']} | None |")
    text += ['', '## Scientific contract', '',
             'The fixed calendar horizons and 10,000-resample paired bootstrap come from the frozen league protocols. Phase 7 scoring, reliability, and zero-margin rules are reused unchanged. All settled outcomes are included; pending outcomes are explicit. Interim metrics cannot authorize promotion. The Phase 9 deployment specifications currently lack prospective_contract bindings; Phase 12 records this separate Phase 8 authorization gap and does not modify those specifications.', '',
             'Admitted kickoff revisions and corrected settled results block the league pending scientific review. Pending fixtures may use fresh provider kickoff metadata before admission. No mixed active baseline or score overwrite is allowed.', '',
             '## Source incidents and rejections', '']
    for league in p.source.LEAGUES:
        codes = sorted({item['code'] for item in rejection_ledger if item['league'] == league})
        text.append(f"- {league}: " + ('; '.join(codes) if codes else 'None'))
    text += ['', 'Source retrieval failures are preserved as failures and do not establish a provider result. BL1 cannot settle before its September 11 fixture completes. SA, FL1 and PD can settle only when a newly retrieved authoritative FINISHED result passes identity, kickoff, and score checks.', '',
             '## Artifact identities', '']
    for row in rows:
        text += [f"- {row['league']} active `{row['active']['bundle_id']}` / `{row['active']['aggregate_sha256']}`; candidate `{row['candidate']['bundle_id']}` / `{row['candidate']['aggregate_sha256']}`."]
    text += ['', '## Integrity and validation', '',
             f"All {verified['protected_integrity']['protected_count']} pre-Phase 12 protected files are unchanged, including all 3,231 Phase 11 protected files and the complete sealed Phase 11 evidence."]
    for label in ('focused', 'protected', 'full'):
        result = tests[label]
        text.append(f"- {label}: {result['passed']} passed, {len(result['failures'])} failed, {result['skipped']} skipped.")
    text += [f"- New unexplained failures: {len(tests['new_unexplained_failures'])}.", '', 'Inherited failures:', '']
    text.extend('- ' + name for name in tests['inherited_failures'])
    text += ['', 'Offline verification validates the sealed source bytes, source parsers, immutable protocol references, complete cohort ordering, features, model identities, common RNG and Phase 4 replay, settlements, metrics, checkpoints, scientific states and protected hashes.', '',
             'The operating guide is docs/phase12-prospective-operations.md. Operational runs remain appendable; each numbered report is a sealed snapshot. A sealed report is never updated in place.', '']
    (directory / 'PHASE12_IMPLEMENTATION_RUN_REPORT.md').write_text('\n'.join(text), encoding='utf-8')
    artifacts = [p.identity(item) for base in [path, *sorted(acquisitions)] for item in sorted(base.rglob('*'))
                 if item.is_file() and item.suffix not in ('.lock', '.tmp') and item.name != 'offline_verification.json']
    p.persist_new(directory / 'artifact_index.json', p.seal({'artifacts': artifacts,
        'implementation_sources': [p.identity(p.ROOT / item) for item in SOURCES]}))
    final = verify(path, report=directory)
    p.persist_new(directory / 'offline_verification.json', final)
    return {'report': str(directory.relative_to(p.ROOT)), **final}

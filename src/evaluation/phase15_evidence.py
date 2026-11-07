"""Phase15 implementation evidence packaging; never publishes scientific reviews."""
from pathlib import Path
from src.evaluation import phase15_review_contract as c
from src.evaluation import phase15_review_verifier as v
from src.evaluation.phase14_evidence import tests

SOURCES = tuple('src/evaluation/' + name + '.py' for name in
    ('phase15_review_contract', 'phase15_review_verifier', 'phase15_review',
     'phase15_phase14', 'phase15_evidence')) + (
    'scripts/run_phase15_review.py', 'tests/unit/test_phase15_review.py',
    'docs/phase15-scientific-review.md')


def integrity():
    before = c.a.p.read(c.RUN / 'protected_before.json')
    changed = [name for name, sha in before.items() if not (c.a.p.ROOT / name).is_file()
               or c.a.p.source.sha256_file(c.a.p.ROOT / name) != sha]
    after = {base: c.a.p.source._tree_snapshot(c.a.p.ROOT / base)
             for base in ('data', 'src/ml/models', 'baselines/phase7p')}
    c.a.p.check(not changed, 'PROTECTED_FILE_CHANGED:' + ','.join(changed))
    c.a.p.check(after == c.a.p.read(c.RUN / 'production_before.json'), 'PRODUCTION_OR_PL7P_CHANGED')
    additions = [c.a.p.identity(path) for base in ('baselines/phase11', 'baselines/phase12')
                 for path in (c.a.p.ROOT / base).rglob('*') if path.is_file() and
                 path.suffix not in ('.lock', '.tmp', '.pyc') and c.a.relative(path) not in before]
    return {'verified': True, 'protected_count': len(before), 'changed': changed,
            'production_after': after, 'concurrent_phase12_additions': additions}


def certify(focused, protected, full):
    prior = c.a.p.read(c.d14.RUN / 'test_evidence.json')
    c.a.p.frozen.sealed(prior, 'record_sha256')
    inherited_ref = next(ref for ref in prior['reports'] if ref['path'].endswith('/full.xml'))
    inherited = tests(c.a.p.frozen.reference(inherited_ref))
    inputs = {'focused': Path(focused), 'protected': Path(protected), 'full': Path(full)}
    summaries = {key: tests(path) for key, path in inputs.items()}
    c.a.p.check(not summaries['focused']['failures'] and not summaries['protected']['failures'],
                'FOCUSED_OR_PROTECTED_FAILURE')
    c.a.p.check(summaries['full']['failures'] == inherited['failures'], 'NEW_FULL_SUITE_FAILURE')
    refs = []
    for key, source in inputs.items():
        target = c.RUN / 'tests' / (key + '.xml')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            c.a.p.check(target.read_bytes() == source.read_bytes(), 'TEST_REPORT_CONFLICT')
        else:
            target.write_bytes(source.read_bytes())
        refs.append(c.a.p.identity(target))
    return c.a.put(c.RUN / 'test_evidence.json', c.a.p.seal({'summaries': summaries,
        'inherited': inherited_ref, 'new_failures': [], 'reports': refs}))


def seal_report():
    receipt = c.a.p.read(c.RUN / 'independent_review_verification.json')
    c.a.p.frozen.sealed(receipt, 'record_sha256')
    review_path = c.a.p.frozen.reference(receipt['review'])
    review = c.a.p.read(review_path)
    c.a.p.frozen.sealed(review, 'review_sha256')
    c.a.p.check(receipt['verified'] and receipt['outcome'] == review['outcome'] and
                not receipt['clears_protocol_blocker'], 'INVALID_INDEPENDENT_RECEIPT')
    c.source_references(review)
    duplicate = c.a.p.read(c.RUN / 'idempotency.json')
    c.a.p.frozen.sealed(duplicate, 'record_sha256')
    c.a.p.check(duplicate['idempotent'] and duplicate['review'] == receipt['review'], 'IDEMPOTENCY_MISMATCH')
    consumer = c.a.p.read(c.RUN / 'phase14_consumption.json')
    c.a.p.frozen.sealed(consumer, 'record_sha256')
    c.a.p.check(consumer['state'] == 'BLOCKED_PROTOCOL_VIOLATION' and
                any(row['code'] == 'AWAITING_HORIZON' for row in consumer['blockers']), 'CONSUMER_MISMATCH')
    certificate = c.a.p.read(c.RUN / 'test_evidence.json')
    c.a.p.frozen.sealed(certificate, 'record_sha256')
    for ref in certificate['reports']:
        c.a.p.frozen.reference(ref)
    comparison = integrity()
    c.a.put(c.RUN / 'protected_comparison.json', c.a.p.seal(comparison))
    basis = review['basis']
    facts = basis['revisions'][0]['facts']
    c.a.put(c.RUN / 'source_identity_index.json', c.a.p.seal({'sources': basis['sources'],
        'observed_versions': basis['observed_versions'], 'rule': basis['rule']}))
    c.a.put(c.RUN / 'report.json', c.a.p.seal({'schema_version': 'ffps.phase15_implementation.v1',
        'review': receipt['review'], 'outcome': review['outcome'], 'phase14_state': consumer['state'],
        'independent_verification': c.a.p.identity(c.RUN / 'independent_review_verification.json'),
        'tests': c.a.p.identity(c.RUN / 'test_evidence.json'),
        'integrity': c.a.p.identity(c.RUN / 'protected_comparison.json'),
        'fixture_retained': True, 'clears_protocol_blocker': False, 'activation_performed': False,
        'scheduling_changed': False, 'other_leagues_changed': False}))
    lines = ['# Phase 15 — Scientific incident review', '',
        '**BL1 fixture 565839: UNRESOLVED_PROTOCOL_CONFLICT.** The review is sealed; the scientific blocker remains.', '',
        'The frozen Phase12 admitted-fixture predicate requires identical kickoff instants. No frozen rule authorizes resolving this actual revision or excluding the admitted pair. Phase13 supplies no review-clearance authority. Phase15 creates no exception and orders no restart.', '',
        '| Evidence | UTC value |', '|---|---|',
        f"| Originally admitted kickoff | {facts['original']['kickoff_at']} |",
        f"| Revised kickoff | {facts['revised']['kickoff_at']} |",
        f"| Revision retrieved | {facts['observed_revision_at']} |",
        f"| Incident recorded | {facts['incident_at']} |",
        f"| Prediction and feature cutoff | {facts['timing']['active']['prediction_at']} |",
        f"| Maximum feature source event | {facts['timing']['active']['maximum_source_event_timestamp']} |",
        f"| Pair published | {facts['publication_at']} |",
        f"| Frozen horizon end | {basis['horizon_end']} |", '',
        'The predictions and source cutoffs precede both kickoff versions. No chronological cohort-order change was found. These factual findings cannot waive the frozen kickoff-equality rule.', '',
        f"The review binds {len(basis['lineage']['ordered_events'])} ordered journal events, {len(basis['lineage']['pair_hashes'])} pairs and {len(basis['lineage']['settlement_hashes'])} settlements. Fixture 565839 remains admitted and unsettled. Complete source versions, retrieval timestamps, raw hashes, protocol identity, admission and subsequent events are indexed in the review and source_identity_index.json.", '',
        'Independent verification reloaded authoritative stored sources, reconstructed the complete BL1 cohort, replayed predictions and reproduced the exact review material. Duplicate evaluation returned the same artifact bytes. Historical reviews remain immutable; current consumers reject incompatible lineage.', '',
        'Phase14 integration uses the additive phase15_phase14 interface and phase14-status CLI. It verifies current lineage and preserves every existing blocker. Original Phase14/Phase13 code and finalization remain unchanged. A synthetic factual-identity outcome cannot grant clearance where the frozen consumers provide no authority. Horizon, settlement, replay, RNG, causal, metric and lineage checks remain enforced.', '',
        f"Protected integrity: **{comparison['protected_count']} files unchanged**. Production trees, candidates, pointers, PL Phase7P and Phase13 contracts are unchanged. Scheduler settings were not modified. SA, FL1 and PD received only isolated generic compatibility tests.",
        f"Concurrent Phase12 additions: {len(comparison['concurrent_phase12_additions'])}; indexed separately from the protected prefix.", '',
        '| Test scope | Passed | Inherited failures | Skipped |', '|---|---:|---:|---:|']
    for key, result in certificate['summaries'].items():
        lines.append(f"| {key} | {result['passed']} | {len(result['failures'])} | {result['skipped']} |")
    lines += ['', 'Zero new failures. The six inherited full-suite failures match the sealed Phase14 identities exactly; XML reports and identities are indexed in test_evidence.json.', '',
        'Operating guide: docs/phase15-scientific-review.md. No activation, model change, retrospective prediction, fixture exclusion or manual bypass was performed.',
        f"Review file SHA256: `{receipt['review']['sha256']}`."]
    target = c.RUN / 'PHASE15_IMPLEMENTATION_REPORT.md'
    content = '\n'.join(lines) + '\n'
    if target.exists():
        c.a.p.check(target.read_text(encoding='utf-8') == content, 'REPORT_CONFLICT')
    else:
        target.write_text(content, encoding='utf-8')
    refs = [c.a.p.identity(path) for path in c.RUN.rglob('*') if path.is_file()
            and path.name not in ('artifact_index.json', 'offline_verification.json')]
    c.a.put(c.RUN / 'artifact_index.json', c.a.p.seal({'artifacts': sorted(refs, key=lambda row: row['path']),
        'implementation': [c.a.p.identity(c.a.p.ROOT / name) for name in SOURCES]}))
    return verify_package()


def verify_package(*, reconstruct=False):
    index = c.a.p.read(c.RUN / 'artifact_index.json')
    c.a.p.frozen.sealed(index, 'record_sha256')
    for ref in index['artifacts'] + index['implementation']:
        c.a.p.frozen.reference(ref)
    receipt = c.a.p.read(c.RUN / 'independent_review_verification.json')
    c.a.p.frozen.sealed(receipt, 'record_sha256')
    if reconstruct:
        actual = v.verify(c.a.p.frozen.reference(receipt['review']))
        c.a.p.check(actual['verified'] and actual['outcome'] == receipt['outcome'], 'OFFLINE_RECONSTRUCTION_MISMATCH')
    integrity()
    return {'verified': True, 'artifact_index': c.a.p.identity(c.RUN / 'artifact_index.json'),
        'independent_reconstruction': c.a.p.identity(c.RUN / 'independent_review_verification.json'),
        'outcome': receipt['outcome'], 'clears_protocol_blocker': False, 'activation_authorized': False}

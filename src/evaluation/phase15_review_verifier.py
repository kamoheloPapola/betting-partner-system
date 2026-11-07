"""Independent review consumer; never imports the review publisher."""
from pathlib import Path
from src.evaluation import phase15_review_contract as c
from src.ml.bundle_activation import canonical_bytes


def chain(directory):
    previous = None
    rows = []
    for number, path in enumerate(sorted(directory.glob('review-*.json')), 1):
        c.a.p.check(path.name == f'review-{number:06d}.json', 'REVIEW_CHAIN_MEMBER_MISSING')
        row = c.a.p.read(path)
        c.a.p.frozen.sealed(row, 'review_sha256')
        c.a.p.check(row['schema_version'] == 'ffps.scientific_incident_review.v1' and
                    row['revision'] == number and row['predecessor'] == previous, 'REVIEW_CHAIN_MISMATCH')
        c.a.p.check(path.read_bytes() == canonical_bytes(row), 'NONCANONICAL_REVIEW_BYTES')
        c.a.p.check(row == c.material(row['basis'], number, previous), 'REVIEW_ENVELOPE_MISMATCH')
        previous = c.a.p.identity(path)
        rows.append((path, row))
    return rows


def verify_locked(path, *, basis=None, require_current=True):
    path = Path(path).resolve()
    league, fixture = path.parent.parent.name, path.parent.name
    c.a.p.check(path.parent == c.RUN / 'reviews' / league / fixture, 'REVIEW_PATH_ESCAPE')
    c.identity(league, fixture)
    rows = chain(path.parent)
    selected = [(p, row) for p, row in rows if p == path]
    c.a.p.check(len(selected) == 1, 'REVIEW_NOT_IN_CHAIN')
    supplied = selected[0][1]
    if require_current:
        c.a.p.check(path == rows[-1][0], 'REVIEW_NOT_CURRENT_SUCCESSOR')
    count = len(supplied['basis']['lineage']['ordered_events'])
    reconstructed = (c.reconstruct(league, fixture, event_count=None if require_current else count)
                     if basis is None else basis)
    c.a.p.check(supplied == c.material(reconstructed, supplied['revision'], supplied['predecessor']),
                'REVIEW_RECONSTRUCTION_OR_LINEAGE_MISMATCH')
    return {'verified': True, 'review': c.a.p.identity(path), 'outcome': supplied['outcome'],
        'applies_to_current_lineage': path == rows[-1][0] and len(c.a.p.journal(c.a.COHORT, league)) == count,
        'clears_protocol_blocker': False, 'activation_authorized': False}


def verify(path):
    path = Path(path).resolve()
    league = path.parent.parent.name
    c.a.p.check(league in c.a.p.source.LEAGUES, 'WRONG_LEAGUE')
    with c.a.p.state_lock(c.a.COHORT / league / 'operation.lock'):
        return verify_locked(path, require_current=False)

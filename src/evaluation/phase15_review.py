"""Immutable scientific review publication; no manual resolution inputs."""
from src.evaluation import phase15_review_contract as c
from src.evaluation import phase15_review_verifier as v


def evaluate(league, fixture):
    c.identity(league, fixture)
    with c.a.p.state_lock(c.a.COHORT / league / 'operation.lock'):
        basis = c.reconstruct(league, fixture)
        directory = c.RUN / 'reviews' / league / str(int(fixture))
        rows = v.chain(directory)
        if rows and rows[-1][1]['basis_sha256'] == c.a.p.digest(basis):
            return {**v.verify_locked(rows[-1][0], basis=basis), 'idempotent': True}
        predecessor = c.a.p.identity(rows[-1][0]) if rows else None
        value = c.material(basis, len(rows)+1, predecessor)
        path = directory / f'review-{len(rows)+1:06d}.json'
        c.a.p.persist_new(path, value)
        return {**v.verify_locked(path, basis=basis), 'idempotent': False}


def status(league, fixture):
    c.identity(league, fixture)
    with c.a.p.state_lock(c.a.COHORT / league / 'operation.lock'):
        basis = c.reconstruct(league, fixture)
        rows = v.chain(c.RUN / 'reviews' / league / str(int(fixture)))
        review = None
        if rows:
            try:
                review = v.verify_locked(rows[-1][0], basis=basis)
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                review = {'verified': False, 'state': 'BLOCKED_STALE_OR_INVALID_REVIEW',
                          'reason': str(exc), 'clears_protocol_blocker': False}
        return {'league': league, 'fixture_id': basis['fixture_id'], 'basis': basis,
                'review': review, 'state': basis['outcome'], 'clears_protocol_blocker': False}

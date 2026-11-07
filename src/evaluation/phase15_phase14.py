"""Additive Phase14 consumption boundary, preserving its sealed implementation.

The frozen consumer supplies no review-clearance authority. A valid review
explains the blocker; malformed or stale reviews add a blocker, never waive one.
"""
from src.evaluation import phase14_decision as d
from src.evaluation import phase15_review_contract as c
from src.evaluation import phase15_review_verifier as v


def attach_verified(report, reviews):
    c.a.p.check(all(review['verified'] and review['clears_protocol_blocker'] is False
                   for review in reviews), 'UNAUTHORIZED_REVIEW_CLEARANCE')
    return {**report, 'incident_reviews': reviews, 'review_clearance_authorized': False}


def status(league, fixture):
    c.identity(league, fixture)
    with c.a.p.state_lock(c.a.COHORT / league / 'operation.lock'):
        report = d.inspect_locked(league)
        try:
            rows = v.chain(c.RUN / 'reviews' / league / str(int(fixture)))
            reviews = [v.verify_locked(rows[-1][0])] if rows else []
            return attach_verified(report, reviews)
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            return {**report, 'state': 'BLOCKED_REVIEW_INVALID',
                'blockers': report['blockers'] + [{'code': 'BLOCKED_REVIEW_INVALID', 'detail': str(exc)}],
                'review_clearance_authorized': False}

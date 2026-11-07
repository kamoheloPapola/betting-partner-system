"""Phase12 evidence adapter over the unchanged Phase8 transaction engine."""
from contextlib import contextmanager
from threading import local
from pathlib import Path
from src.ml.league_bundle_activation import LeagueBundleController
from src.evaluation import phase13_authorization as authorization


class ProspectiveBundleController(LeagueBundleController):
    def __init__(self, specification_path, *, league, root=authorization.p.ROOT):
        super().__init__(specification_path, league=league, root=root)
        authorization.p.check(self.root == authorization.p.ROOT, 'AUTHORIZATION_ROOT_MISMATCH')
        self._guard = local()

    @contextmanager
    def cohort_guard(self):
        if getattr(self._guard, 'held', False):
            yield
            return
        with authorization.p.state_lock(authorization.COHORT / self.league / 'operation.lock'):
            self._guard.held = True
            try:
                yield
            finally:
                self._guard.held = False

    def league_promotion_gate(self, evidence):
        # Missing decisions require binding validation but no collection lock.
        if not (Path(evidence) / 'prospective_result.json').exists():
            return authorization.gate(self, evidence)
        with self.cohort_guard():
            return authorization.gate(self, evidence)

    def activate(self, evidence, *, expected_active, operator):
        with self.cohort_guard():
            return super().activate(evidence, expected_active=expected_active, operator=operator)

    def dry_run(self, evidence, expected_active):
        with self.cohort_guard():
            return super().dry_run(evidence, expected_active=expected_active)

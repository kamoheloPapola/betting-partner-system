"""Clean v2 origin, immutable binding, and historical separation."""
from pathlib import Path
from tempfile import TemporaryDirectory
import ast

import pytest

from src.evaluation import phase12_prospective as p
from src.evaluation import phase16_cohort as c
from src.evaluation import phase16_verifier as v


@pytest.fixture
def private_run(monkeypatch):
    temporary = p.ROOT / 'tmp'
    temporary.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix='phase16-cohort-test-', dir=temporary) as name:
        directory = Path(name).resolve()
        assert directory.is_relative_to(temporary.resolve())
        monkeypatch.setattr(c, 'RUN', directory)
        monkeypatch.setattr(c, 'RULE_FILE', directory / 'fixture_revision_rule_v2.json')
        monkeypatch.setattr(v, 'RUN', directory)
        yield directory


def test_clean_cohort_initialization_and_independent_verification(private_run):
    c.freeze_rule()
    spec = c.initialize('BL1-clean-001', 'BL1')
    assert spec == c.initialize('BL1-clean-001', 'BL1')
    assert spec['initial_pair_prefix'] == [] and spec['initial_journal_head'] is None
    assert p.instant(spec['created_at']) < p.instant(spec['start_at']) < p.instant(spec['end_at'])
    proof = v.verify_cohort('BL1-clean-001')
    assert proof['verified'] and proof['pairs'] == proof['settlements'] == 0
    assert proof['scientific_state'] == 'PHASE16_PROTOCOL_V2_READY_FOR_CLEAN_COHORT'
    assert not proof['activation_authorized']


def test_existing_cohort_cannot_change_league_or_protocol(private_run):
    c.freeze_rule()
    c.initialize('BL1-clean-001', 'BL1')
    with pytest.raises(ValueError, match='COHORT_LEAGUE_SUBSTITUTION'):
        c.initialize('BL1-clean-001', 'SA')
    path = private_run / 'cohorts/BL1-clean-001/prospective_protocol.json'
    path.write_bytes(path.read_bytes().replace(b'ffps.league_prospective_protocol.v2',
                                             b'ffps.league_prospective_protocol.v1'))
    with pytest.raises(ValueError, match='ARTIFACT_DRIFT'):
        v.verify_cohort('BL1-clean-001')


def test_old_phase15_review_untouched_and_unresolved():
    from src.evaluation import phase15_review_contract as old
    path = old.RUN / 'reviews/BL1/565839/review-000001.json'
    review = p.read(path)
    p.frozen.sealed(review, 'review_sha256')
    assert review['outcome'] == 'UNRESOLVED_PROTOCOL_CONFLICT'
    assert review['clears_protocol_blocker'] is False


def test_independent_verifier_does_not_import_publisher():
    tree = ast.parse(Path(v.__file__).read_text(encoding='utf-8'))
    names = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
             for alias in node.names]
    names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
              for alias in node.names]
    assert 'phase16_cohort' not in names

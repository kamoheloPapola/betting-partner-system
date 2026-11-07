"""Current count-market regression contract established by approved FFPS Phase 2.

Phase 0's mixed golden remains historical evidence. Phase 1 protects PL goal/1X2.
This module treats the Phase 2 controlled golden as authoritative for corrected
corner/card behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.ml.count_domains import sanitize_card_count_mean, sanitize_corner_count_mean
from src.ml.distributions import NegativeBinomialEngine, ZeroInflatedEngine


ROOT = Path(__file__).resolve().parents[2]
PHASE2_GOLDEN_PATH = (
    ROOT
    / "baselines"
    / "phase2"
    / "phase2-count-markets-79110f9a-20260811-v1"
    / "controlled_goldens.json"
)
PHASE2_COUNT_GOLDEN_SHA256 = "49ec83a5ae42dbe86883f08892002e759c282eac27dcd1bd002e0ff9f14f53c9"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cases() -> dict[str, dict]:
    payload = json.loads(PHASE2_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert payload["repetition"]["exact_match"] is True
    assert payload["repetition"]["run_1"] == payload["repetition"]["run_2"]
    return {case["case"]: case for case in payload["cases"]}


def test_phase2_count_golden_is_current_and_immutable() -> None:
    assert PHASE2_GOLDEN_PATH.is_file()
    assert _sha256(PHASE2_GOLDEN_PATH) == PHASE2_COUNT_GOLDEN_SHA256
    assert set(_cases()) == {
        "corner_specialist_count_domain",
        "card_specialist_count_domain",
    }


def test_phase2_corrected_corner_golden_reproduces() -> None:
    case = _cases()["corner_specialist_count_domain"]
    total = sanitize_corner_count_mean(case["new_specialist_total"])
    assert total > 4.0
    inputs = case["new_distribution_inputs"]
    home_mean = inputs["home_mean"]
    away_mean = inputs["away_mean"]
    team_total = case["inputs"]["team_home_mean"] + case["inputs"]["team_away_mean"]
    assert home_mean == pytest.approx(total * case["inputs"]["team_home_mean"] / team_total)
    assert away_mean == pytest.approx(total * case["inputs"]["team_away_mean"] / team_total)
    assert home_mean + away_mean == pytest.approx(total)
    actual = NegativeBinomialEngine().calculate_probabilities(
        home_mean,
        home_mean * 1.3,
        away_mean,
        away_mean * 1.3,
    )
    assert actual == pytest.approx(case["new_probabilities"], abs=1e-12)


def test_phase2_corrected_card_golden_reproduces() -> None:
    case = _cases()["card_specialist_count_domain"]
    total = sanitize_card_count_mean(case["new_specialist_total"])
    assert total > 4.0
    actual = ZeroInflatedEngine().calculate_probabilities(
        total,
        pi_zero=case["inputs"]["pi_zero"],
        var_mult=case["inputs"]["intensity_multiplier"],
    )
    assert actual == pytest.approx(case["new_probabilities"], abs=1e-12)
    assert actual["cards_over_4_5"] <= actual["cards_over_2_5"]
    assert actual["cards_under_4_5"] <= actual["cards_under_5_5"]
    assert actual["cards_under_4_5"] + actual["cards_over_4_5"] == pytest.approx(1.0)

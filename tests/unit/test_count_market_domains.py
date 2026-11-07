from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.cli.commands.prediction as prediction_module
from src.cli.commands.prediction import _calc_cards, _calc_corners
from src.ml.count_domains import sanitize_card_count_mean, sanitize_corner_count_mean
from src.ml.distributions import NegativeBinomialEngine, ZeroInflatedEngine
from src.simulation.match_simulator import LAMBDA_MAX, clamp_lambda


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value
        self.meta = {
            "features": ["f1"],
            "name": "phase2_constant",
            "metrics": {"calibration_score": 0.05},
        }

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


class NoOffsets:
    def get_offset(self, team: object, league: str) -> None:
        return None


class Confidence:
    def calculate(self, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(confidence=0.5)


def match() -> pd.Series:
    return pd.Series(
        {
            "f1": 1.0,
            "match_id": "phase2-count-case",
            "home_team": "Home",
            "away_team": "Away",
            "h2h_match_count": 0,
        }
    )


def test_domain_isolation_preserves_goal_cap_but_not_count_means() -> None:
    assert clamp_lambda(9.2) == LAMBDA_MAX
    assert sanitize_corner_count_mean(9.2) == 9.2
    assert sanitize_card_count_mean(5.6) == 5.6


@pytest.mark.parametrize("sanitizer", [sanitize_corner_count_mean, sanitize_card_count_mean])
def test_count_sanitizers_floor_negative_values_and_reject_non_finite(sanitizer) -> None:
    assert sanitizer(-3.0) == 0.05
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="non-finite"):
            sanitizer(value)


def test_corner_specialist_above_four_survives_and_rescales_team_means(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float]] = []

    class CapturingNB:
        def calculate_probabilities(self, mu_h, var_h, mu_a, var_a):
            calls.append((float(mu_h), float(var_h), float(mu_a), float(var_a)))
            return {
                "corners_home_win": 0.45,
                "corners_draw": 0.20,
                "corners_away_win": 0.35,
                "corners_over_7_5": 0.70,
                "corners_under_11_5": 0.65,
            }

    monkeypatch.setattr(prediction_module, "NegativeBinomialEngine", CapturingNB)
    monkeypatch.setattr(prediction_module, "TeamOffsetManager", NoOffsets)
    monkeypatch.setattr(prediction_module, "get_confidence_calculator", lambda: Confidence())
    monkeypatch.setattr(prediction_module, "apply_soft_cap", lambda p, confidence, market: p)

    suite = {
        "mh_corn": ConstantModel(6.0),
        "ma_corn": ConstantModel(4.0),
        "m_corners_lgbm": ConstantModel(10.0),
        "m_corners_xgb": ConstantModel(8.0),
    }
    result, _ = _calc_corners(match(), suite, "Phase2", "PL")

    expected_total = 0.6 * 10.0 + 0.4 * 8.0
    assert expected_total == pytest.approx(9.2)
    assert calls[0][0] + calls[0][2] == pytest.approx(expected_total)
    assert calls[-1][0] + calls[-1][2] == pytest.approx(expected_total)
    assert result["corn_1x2_h"] + result["corn_1x2_d"] + result["corn_1x2_a"] == pytest.approx(1.0)
    assert all(0 <= value <= 1 for value in result.values() if value is not None)


def test_card_specialist_above_four_survives_to_zip(monkeypatch) -> None:
    calls: list[tuple[float, float, float]] = []

    class CapturingZIP:
        CLAMP_CEILING_U55 = 0.88

        def calculate_probabilities(self, mu, pi_zero=0.03, var_mult=1.0):
            calls.append((float(mu), float(pi_zero), float(var_mult)))
            return {
                "cards_under_4_5": 0.80,
                "cards_over_4_5": 0.20,
                "cards_over_2_5": 0.20,
                "cards_under_5_5": 0.60,
            }

    monkeypatch.setattr(prediction_module, "ZeroInflatedEngine", CapturingZIP)
    monkeypatch.setattr(prediction_module, "TeamOffsetManager", NoOffsets)
    monkeypatch.setattr(prediction_module, "get_confidence_calculator", lambda: Confidence())
    monkeypatch.setattr(prediction_module, "apply_soft_cap", lambda p, confidence, market: p)
    monkeypatch.setattr(prediction_module, "apply_calibration_cap", lambda p, market, ctx="": p)

    suite = {
        "m_cards_lgbm": ConstantModel(6.0),
        "m_cards_xgb": ConstantModel(5.0),
    }
    result, _ = _calc_cards(match(), suite, "Phase2", "PL")

    expected_total = 0.6 * 6.0 + 0.4 * 5.0
    assert expected_total == pytest.approx(5.6)
    assert calls[0][0] == pytest.approx(expected_total)
    assert calls[-1][0] == pytest.approx(expected_total)
    assert result["card_o25"] >= 1.0 - result["card_u45"]
    assert result["card_u45"] <= result["card_u55"]
    assert all(0 <= value <= 1 for value in result.values() if value is not None)


def test_corner_and_card_market_tail_invariants() -> None:
    corners = NegativeBinomialEngine().calculate_probabilities(5.5, 8.0, 4.5, 6.5)
    assert corners["corners_over_11_5"] <= corners["corners_over_7_5"]
    assert sum(corners[key] for key in ("corners_home_win", "corners_draw", "corners_away_win")) == pytest.approx(1.0)

    cards = ZeroInflatedEngine().calculate_probabilities(5.6, pi_zero=0.03)
    assert cards["cards_over_4_5"] <= cards["cards_over_2_5"]
    assert cards["cards_under_4_5"] <= cards["cards_under_5_5"]
    assert cards["cards_under_4_5"] + cards["cards_over_4_5"] == pytest.approx(1.0)


def test_distribution_parameter_validation_fails_closed() -> None:
    nb = NegativeBinomialEngine()
    zip_engine = ZeroInflatedEngine()
    with pytest.raises(ValueError):
        nb.calculate_probabilities(np.nan, 2.0, 4.0, 6.0)
    with pytest.raises(ValueError):
        nb.calculate_probabilities(-1.0, 2.0, 4.0, 6.0)
    with pytest.raises(ValueError):
        zip_engine.calculate_probabilities(np.inf)
    with pytest.raises(ValueError):
        zip_engine.calculate_probabilities(-1.0)
    with pytest.raises(ValueError):
        zip_engine.calculate_probabilities(5.0, pi_zero=1.0)
    with pytest.raises(ValueError):
        zip_engine.calculate_probabilities(5.0, var_mult=0.0)

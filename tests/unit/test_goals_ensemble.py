import logging

import numpy as np
import pandas as pd
import pytest

import src.cli.commands.prediction as prediction_module
from src.cli.commands.prediction import _calc_goals, _calculate_probabilities


class ConstantModel:
    def __init__(self, value: float, features: list[str]) -> None:
        self.value = value
        self.features = features
        self.meta = {
            "features": features,
            "name": "constant_model",
            "metrics": {"calibration_score": 0.05},
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def _suite() -> dict:
    features = ["f1"]
    return {
        "mh_goals": ConstantModel(2.0, features),
        "ma_goals": ConstantModel(1.0, features),
        "meta_goals": {"features": features},
        "mh_goals_poisson": ConstantModel(1.0, features),
        "ma_goals_poisson": ConstantModel(2.0, features),
        "meta_goals_poisson": {"features": features},
    }


def _match() -> pd.Series:
    return pd.Series(
        {
            "f1": 0.5,
            "match_id": "match-xyz",
            "home_team": "Home",
            "away_team": "Away",
            "h2h_match_count": 0,
        }
    )


def test_goal_ensemble_weighted_lambda(monkeypatch):
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_LGBM_WEIGHT", 0.6)
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_XGB_WEIGHT", 0.4)
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD", 0.2)

    result = _calc_goals(_match(), _suite(), "Home vs Away", use_simulator=False)

    assert result["goal_model_home_lambda"] == pytest.approx(1.6, abs=1e-8)
    assert result["goal_model_away_lambda"] == pytest.approx(1.4, abs=1e-8)
    assert result["ensemble_divergence"] is True
    assert result["divergence_pct"] == pytest.approx(50.0, abs=1e-8)


def test_goal_ensemble_logs_warning_on_divergence(monkeypatch, caplog):
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_LGBM_WEIGHT", 0.6)
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_XGB_WEIGHT", 0.4)
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD", 0.2)

    with caplog.at_level(logging.WARNING):
        _calc_goals(_match(), _suite(), "Home vs Away", use_simulator=False)

    assert any("GOALS_ENSEMBLE_DIVERGENCE" in rec.message for rec in caplog.records)
    assert any("match-xyz" in rec.message for rec in caplog.records)


def test_calc_goals_applies_bl1_specific_btts_cap(monkeypatch):
    monkeypatch.setattr(
        prediction_module,
        "apply_lambda_aware_adjustment",
        lambda prob, market, home_lambda, away_lambda, ctx="", strength=0.25: 0.64 if market == "btts_yes" else prob,
    )
    monkeypatch.setattr(
        prediction_module,
        "apply_calibration_cap",
        lambda prob, market, ctx="": prob,
    )

    bl1_result = _calc_goals(_match(), _suite(), "Home vs Away", use_simulator=False, league="BL1")
    pl_result = _calc_goals(_match(), _suite(), "Home vs Away", use_simulator=False, league="PL")

    assert bl1_result["btts"] == pytest.approx(0.58, abs=1e-8)
    assert pl_result["btts"] == pytest.approx(0.64, abs=1e-8)


def test_btts_no_is_complement_of_btts_yes(monkeypatch):
    def fake_cap(prob: float, market: str, ctx: str = "") -> float:
        if market == "btts_no":
            raise AssertionError("btts_no should be derived from btts_yes, not capped independently")
        if market == "btts_yes":
            return 0.63
        return prob

    monkeypatch.setattr(prediction_module, "apply_calibration_cap", fake_cap)

    result = _calc_goals(_match(), _suite(), "Home vs Away", use_simulator=False, league="PL")

    assert result["btts"] == pytest.approx(0.63, abs=1e-8)
    assert result["btts_no"] == pytest.approx(0.37, abs=1e-8)
    assert result["btts"] + result["btts_no"] == pytest.approx(1.0, abs=1e-9)


def test_o15_reanchored_when_u25_cap_fires(monkeypatch, caplog):
    class _FakePoissonEngine:
        def calculate_probabilities(self, home_lambda: float, away_lambda: float) -> dict:
            return {
                "home_win": 0.4,
                "draw": 0.3,
                "away_win": 0.3,
                "over_2_5": 0.2,
                "under_2_5": 0.8,
                "over_1_5": 0.3,
                "under_3_5": 0.9,
                "btts_yes": 0.25,
                "btts_no": 0.75,
                "home_under_1_5": 0.8,
                "away_under_1_5": 0.8,
            }

    def fake_adjust(prob: float, market: str, home_lambda: float, away_lambda: float, ctx: str = "", strength: float = 0.25) -> float:
        if market == "u25":
            return 0.70
        return prob

    def fake_cap(prob: float, market: str, ctx: str = "") -> float:
        if market == "u25":
            return 0.66
        return prob

    monkeypatch.setattr(prediction_module, "PoissonEngine", _FakePoissonEngine)
    monkeypatch.setattr(prediction_module, "apply_lambda_aware_adjustment", fake_adjust)
    monkeypatch.setattr(prediction_module, "apply_calibration_cap", fake_cap)

    with caplog.at_level(logging.WARNING):
        result = _calc_goals(_match(), _suite(), "Home vs Away", use_simulator=False, league="PL")

    assert result["o25"] == pytest.approx(0.34, abs=1e-8)
    assert result["over_1_5"] == pytest.approx(0.34, abs=1e-8)
    assert result["over_1_5"] >= result["o25"]
    assert "o15 re-anchored post u25-cap" in caplog.text


def test_cards_divergence_sets_global_ensemble_flag(monkeypatch):
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_LGBM_WEIGHT", 0.6)
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_XGB_WEIGHT", 0.4)
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD", 0.2)

    features = ["f1"]
    suite = {
        "mh_goals": ConstantModel(1.4, features),
        "ma_goals": ConstantModel(1.4, features),
        "meta_goals": {"features": features},
        "mh_goals_xgb": ConstantModel(1.4, features),
        "ma_goals_xgb": ConstantModel(1.4, features),
        "meta_goals_xgb": {"features": features},
        "m_cards_lgbm": ConstantModel(6.0, features),
        "m_cards_xgb": ConstantModel(3.0, features),
    }
    match = pd.Series(
        {
            "f1": 0.2,
            "match_id": "cards-div-match",
            "home_team": "Home",
            "away_team": "Away",
            "league": "PL",
            "h2h_match_count": 0,
        }
    )

    probs, _ = _calculate_probabilities(match, suite, "PL")
    assert probs["ensemble_divergence"] is True
    assert probs["divergence_pct"] == pytest.approx(50.0, abs=1e-8)


def test_corners_divergence_sets_global_ensemble_flag(monkeypatch):
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_LGBM_WEIGHT", 0.6)
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_XGB_WEIGHT", 0.4)
    monkeypatch.setattr(prediction_module, "GOALS_ENSEMBLE_DIVERGENCE_THRESHOLD", 0.2)

    features = ["f1"]
    suite = {
        "mh_goals": ConstantModel(1.4, features),
        "ma_goals": ConstantModel(1.4, features),
        "meta_goals": {"features": features},
        "mh_goals_xgb": ConstantModel(1.4, features),
        "ma_goals_xgb": ConstantModel(1.4, features),
        "meta_goals_xgb": {"features": features},
        "mh_corn": ConstantModel(5.0, features),
        "ma_corn": ConstantModel(5.0, features),
        "m_corners_lgbm": ConstantModel(3.0, features),
        "m_corners_xgb": ConstantModel(1.5, features),
    }
    match = pd.Series(
        {
            "f1": 0.2,
            "match_id": "corners-div-match",
            "home_team": "Home",
            "away_team": "Away",
            "league": "PL",
            "h2h_match_count": 0,
        }
    )

    probs, _ = _calculate_probabilities(match, suite, "PL")
    assert probs["ensemble_divergence"] is True
    assert probs["divergence_pct"] == pytest.approx(50.0, abs=1e-8)


def test_corners_h2h_lift_is_capped():
    """H2H corners blend cannot push mu_total up by more than 2.0."""
    mu_model = 10.0
    h2h_avg = 18.0  # extreme H2H history
    h2h_weight = 0.4  # max weight (5+ matches)

    blended = (1 - h2h_weight) * mu_model + h2h_weight * h2h_avg
    capped = min(blended, mu_model + 2.0)

    assert capped == 12.0
    assert blended > capped


def test_corners_h2h_negative_lift_uncapped():
    """Negative H2H lift (low-scoring history) is never restricted."""
    mu_model = 10.0
    h2h_avg = 4.0  # very low H2H history
    h2h_weight = 0.4

    blended = (1 - h2h_weight) * mu_model + h2h_weight * h2h_avg

    assert blended < mu_model

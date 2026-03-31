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

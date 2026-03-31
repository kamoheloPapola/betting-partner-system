from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cli.commands.prediction import ModelSuite, _calculate_probabilities
from src.features.pipeline import FeaturePipeline
from src.simulation.match_simulator import DEFAULT_N_SIMULATIONS


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "sample_matches.csv"


class ConstantRegressor:
    def __init__(self, value: float, features: list[str], name: str) -> None:
        self.value = value
        self.features = features
        self.meta = {
            "name": name,
            "version": "test",
            "metrics": {"calibration_score": 0.05},
            "features": features,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value, dtype=float)


def test_full_prediction_workflow_from_fixture_dataframe() -> None:
    sample_matches = pd.read_csv(FIXTURE_PATH)
    sample_matches["date"] = pd.to_datetime(sample_matches["date"])

    features = FeaturePipeline().transform(sample_matches)

    assert len(features) == len(sample_matches)
    assert {"h2h_goals_o25_rate", "h2h_btts_rate", "h2h_match_count"}.issubset(features.columns)
    assert features["h2h_goals_o25_rate"].tolist() == pytest.approx([0.53, 0.53, 0.53])
    assert features["h2h_btts_rate"].tolist() == pytest.approx([0.54, 0.54, 0.54])
    assert features["h2h_match_count"].tolist() == [0, 0, 0]

    goal_features = [
        "home_rolling_goals_scored_5",
        "home_rolling_goals_conceded_5",
        "away_rolling_goals_scored_5",
        "away_rolling_goals_conceded_5",
    ]
    suite: ModelSuite = {
        "mh_goals": ConstantRegressor(1.4, goal_features, "home_goals"),
        "ma_goals": ConstantRegressor(1.1, goal_features, "away_goals"),
        "meta_goals": {"features": goal_features},
    }

    for _, row in features.iterrows():
        probabilities, attribution = _calculate_probabilities(row, suite, row["league"])

        assert attribution == {}
        assert probabilities["home"] + probabilities["draw"] + probabilities["away"] == pytest.approx(1.0, abs=1e-6)
        assert probabilities["u25"] + probabilities["o25"] == pytest.approx(1.0, abs=1e-6)
        assert probabilities["over_1_5"] >= probabilities["o25"] >= 0.0
        assert 0.0 <= probabilities["btts"] <= 1.0
        assert 0.0 <= probabilities["dc_1x"] <= 1.0
        assert 0.0 <= probabilities["dc_x2"] <= 1.0
        assert 0.0 <= probabilities["dc_12"] <= 1.0
        assert probabilities["expected_home_goals"] > 0.0
        assert probabilities["expected_away_goals"] > 0.0
        assert probabilities["mc_n_simulations"] == DEFAULT_N_SIMULATIONS
        assert isinstance(probabilities["mc_top_scorelines"], list)
        assert len(probabilities["mc_top_scorelines"]) > 0

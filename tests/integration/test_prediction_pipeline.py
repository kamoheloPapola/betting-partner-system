from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cli.commands.prediction import (
    ModelSuite,
    _calculate_probabilities_v2,
)
from src.features.pipeline import FeaturePipeline
from src.simulation.match_simulator import DEFAULT_N_SIMULATIONS, MatchSimulator


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
        simulator = MatchSimulator(
            n_simulations=DEFAULT_N_SIMULATIONS,
            seed=42,
            league=row["league"],
        )
        probabilities, attribution = _calculate_probabilities_v2(
            row,
            suite,
            row["league"],
            False,
            {"missing_offsets": set(), "missing_card_offsets": set()},
            simulator=simulator,
            use_simulator=True,
        )

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


def test_simulation_shifts_probabilities_vs_analytical() -> None:
    sample_matches = pd.read_csv(FIXTURE_PATH)
    sample_matches["date"] = pd.to_datetime(sample_matches["date"])
    match = FeaturePipeline().transform(sample_matches).iloc[0]

    goal_features = [
        "home_rolling_goals_scored_5",
        "home_rolling_goals_conceded_5",
        "away_rolling_goals_scored_5",
        "away_rolling_goals_conceded_5",
    ]
    suite: ModelSuite = {
        "mh_goals": ConstantRegressor(1.6, goal_features, "home_goals"),
        "ma_goals": ConstantRegressor(1.6, goal_features, "away_goals"),
        "meta_goals": {"features": goal_features},
    }
    simulator = MatchSimulator(
        n_simulations=DEFAULT_N_SIMULATIONS,
        seed=42,
        league=match["league"],
    )

    sim_probabilities, _ = _calculate_probabilities_v2(
        match,
        suite,
        match["league"],
        False,
        {"missing_offsets": set(), "missing_card_offsets": set()},
        simulator,
        use_simulator=True,
    )
    analytical_probabilities, _ = _calculate_probabilities_v2(
        match,
        suite,
        match["league"],
        False,
        {"missing_offsets": set(), "missing_card_offsets": set()},
        simulator,
        use_simulator=False,
    )

    assert sim_probabilities["goal_model_home_lambda"] == pytest.approx(1.6)
    assert sim_probabilities["goal_model_away_lambda"] == pytest.approx(1.6)
    assert analytical_probabilities["goal_model_home_lambda"] == pytest.approx(1.6)
    assert analytical_probabilities["goal_model_away_lambda"] == pytest.approx(1.6)

    assert abs(sim_probabilities["home"] - analytical_probabilities["home"]) >= 0.01
    assert abs(sim_probabilities["draw"] - analytical_probabilities["draw"]) >= 0.01
    assert (
        abs(sim_probabilities["o25"] - analytical_probabilities["o25"]) >= 0.005
        or abs(sim_probabilities["btts"] - analytical_probabilities["btts"]) >= 0.005
    )

from types import SimpleNamespace

from src.strategies.forbidden_fruit import ForbiddenFruitEngine


class _StubCalibrator:
    def calibrate(self, market, probability):
        return probability


class _StubSharpness:
    def get_status(self, market):
        return "ACTIVE"


class _StubDriftMonitor:
    def check_drift(self, market):
        return True


class _StubConfidenceCalculator:
    def calculate(self, **kwargs):
        return SimpleNamespace(
            confidence=0.74,
            action_tier="TIER_A",
            components=["stubbed"],
        )


class _StubRejectionLogger:
    def log_rejection(self, *args, **kwargs):
        return None


def test_value_opportunities_return_edge_without_stake(
    isolated_repo_state,
    monkeypatch,
):
    import src.strategies.forbidden_fruit as forbidden_fruit_module

    engine = ForbiddenFruitEngine()

    monkeypatch.setattr(forbidden_fruit_module, "get_calibrator", lambda: _StubCalibrator())
    monkeypatch.setattr(
        forbidden_fruit_module,
        "get_sharpness_gate",
        lambda: _StubSharpness(),
    )
    monkeypatch.setattr(
        forbidden_fruit_module,
        "get_drift_monitor",
        lambda: _StubDriftMonitor(),
    )
    monkeypatch.setattr(
        forbidden_fruit_module,
        "get_rejection_logger",
        lambda: _StubRejectionLogger(),
    )
    monkeypatch.setattr(
        forbidden_fruit_module,
        "get_confidence_calculator",
        lambda: _StubConfidenceCalculator(),
    )
    monkeypatch.setattr(forbidden_fruit_module, "apply_soft_cap", lambda p, c, m: p)

    opportunities = engine.evaluate_value_opportunities(
        match={
            "home_team": "Arsenal",
            "away_team": "Everton",
            "league": "PL",
        },
        raw_predictions={"u25": 0.60},
        odds={"goals_u25": 2.10},
    )

    assert len(opportunities) == 1
    opportunity = opportunities[0]
    assert opportunity["market"] == "goals_u25"
    assert opportunity["passes_value_threshold"] is True
    assert opportunity["edge"] > 0
    assert opportunity["ev"] > 0
    assert opportunity["value_reason"] == "VALUE_OK"
    assert "stake" not in opportunity
    assert "stake_mult" not in opportunity
    assert "bet_id" not in opportunity

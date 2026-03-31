import pytest

from src.strategies.edge_engine import EdgeEngine


def test_edge_engine_computes_price_metrics():
    engine = EdgeEngine()

    assert engine.compute_implied_probability(2.0) == pytest.approx(0.5)
    assert engine.compute_edge(0.6, 2.0) == pytest.approx(0.1)
    assert engine.compute_ev(0.6, 2.0) == pytest.approx(0.2)
    assert engine.passes_value_threshold(0.6, 2.0) is True


def test_edge_engine_marks_unpriced_opportunities():
    engine = EdgeEngine()

    evaluation = engine.evaluate_opportunity(probability=0.6, odds=None)

    assert evaluation["priced"] is False
    assert evaluation["passes_value_threshold"] is False
    assert evaluation["value_reason"] == "MISSING_ODDS"


def test_edge_engine_ranks_priced_value_above_unpriced_candidates():
    engine = EdgeEngine()
    ranked = engine.rank_opportunities(
        [
            {
                "market": "goals_u25",
                "confidence": 0.7,
                "passes_value_threshold": False,
                "edge": None,
                "ev": None,
            },
            {
                "market": "btts_no",
                "confidence": 0.68,
                "passes_value_threshold": True,
                "edge": 0.08,
                "ev": 0.14,
            },
        ]
    )

    assert ranked[0]["market"] == "btts_no"
    assert ranked[0]["value_rank"] == 1

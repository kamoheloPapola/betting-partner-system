import pytest

from src.config.thresholds import Thresholds
from src.strategies.selection_gate import SelectionGate


@pytest.mark.parametrize(
    ("market", "expected"),
    [
        ("goals_over_2_5", "goals"),
        ("over_1_5", "over_1_5"),
        ("cards_over_2_5", "cards"),
        ("cards_under_4_5", "cards"),
        ("corners_under_11_5", "corners"),
        ("corners_over_9_5", "corners"),
        ("double_chance", "double_chance"),
    ],
)
def test_market_type_routing(market, expected):
    gate = SelectionGate()
    assert gate._get_market_type(market) == expected


def test_team_under_1_5_can_pass_edge_gate():
    """home_under_1_5 at cap value (0.80) must clear the 8% edge requirement."""
    cap = 0.80  # MARKET_PROB_CAPS['home_under_1_5']
    base = Thresholds.BASE_TEAM_U15
    min_edge = Thresholds.GATE_MIN_EDGE
    assert cap - base >= min_edge, (
        f"home_under_1_5 at cap ({cap}) cannot clear edge gate: "
        f"edge={cap - base:.2f} < min={min_edge}"
    )

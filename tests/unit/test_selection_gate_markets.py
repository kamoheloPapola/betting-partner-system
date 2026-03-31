import pytest

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

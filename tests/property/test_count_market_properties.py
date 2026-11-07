from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from src.ml.count_domains import sanitize_card_count_mean, sanitize_corner_count_mean
from src.ml.distributions import NegativeBinomialEngine, ZeroInflatedEngine


finite_means = st.floats(min_value=0.05, max_value=25.0, allow_nan=False, allow_infinity=False)


@given(finite_means)
def test_count_domain_sanitizers_never_apply_goal_ceiling(mean: float) -> None:
    assert sanitize_corner_count_mean(mean) == mean
    assert sanitize_card_count_mean(mean) == mean


@given(
    home=st.floats(min_value=0.05, max_value=12.0, allow_nan=False, allow_infinity=False),
    away=st.floats(min_value=0.05, max_value=12.0, allow_nan=False, allow_infinity=False),
)
def test_corner_distribution_is_finite_bounded_and_coherent(home: float, away: float) -> None:
    probabilities = NegativeBinomialEngine().calculate_probabilities(
        home,
        home * 1.3,
        away,
        away * 1.3,
    )
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values())
    assert probabilities["corners_over_11_5"] <= probabilities["corners_over_7_5"]
    assert sum(probabilities[key] for key in ("corners_home_win", "corners_draw", "corners_away_win")) == pytest.approx(1.0)


@given(st.floats(min_value=0.05, max_value=15.0, allow_nan=False, allow_infinity=False))
def test_card_distribution_is_finite_bounded_and_monotonic(mean: float) -> None:
    probabilities = ZeroInflatedEngine().calculate_probabilities(mean, pi_zero=0.03)
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities.values())
    assert probabilities["cards_over_4_5"] <= probabilities["cards_over_2_5"] + 1e-12
    assert probabilities["cards_under_4_5"] <= probabilities["cards_under_5_5"] + 1e-12
    assert probabilities["cards_under_5_5"] <= ZeroInflatedEngine.CLAMP_CEILING_U55 + 1e-12

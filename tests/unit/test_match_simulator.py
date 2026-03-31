"""
Unit tests for Monte Carlo Match Simulator.

Validates probability coherence, symmetry, home advantage effect,
scoreline coverage, reproducibility, lambda clamping, BTTS, entropy
classification, tail mass, and the global clamp_lambda utility.
"""

import pytest
import numpy as np

from src.simulation.match_simulator import (
    MatchSimulator,
    SimulationResult,
    clamp_lambda,
    LAMBDA_MIN,
    LAMBDA_MAX,
    DEFAULT_N_SIMULATIONS,
)


# Use a large number of simulations for stability in tests
N_SIM = DEFAULT_N_SIMULATIONS
TOLERANCE = 0.03  # 3pp tolerance for Monte Carlo variance


class TestClampLambda:
    """Tests for the global clamp_lambda utility."""

    def test_clamp_negative(self):
        assert clamp_lambda(-1.0) == LAMBDA_MIN

    def test_clamp_zero(self):
        assert clamp_lambda(0.0) == LAMBDA_MIN

    def test_clamp_too_high(self):
        assert clamp_lambda(10.0) == LAMBDA_MAX

    def test_clamp_within_range(self):
        assert clamp_lambda(1.5) == 1.5

    def test_clamp_boundary_low(self):
        assert clamp_lambda(LAMBDA_MIN) == LAMBDA_MIN

    def test_clamp_boundary_high(self):
        assert clamp_lambda(LAMBDA_MAX) == LAMBDA_MAX

    def test_clamp_array(self):
        arr = np.array([-1.0, 0.0, 1.5, 5.0])
        result = clamp_lambda(arr)
        expected = np.array([LAMBDA_MIN, LAMBDA_MIN, 1.5, LAMBDA_MAX])
        np.testing.assert_array_almost_equal(result, expected)


class TestSimulationResultCoherence:
    """Tests that probabilities are internally consistent."""

    @pytest.fixture
    def result(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        return sim.simulate(home_xg=1.5, away_xg=1.1)

    def test_1x2_sums_to_one(self, result):
        total = result.home_win_prob + result.draw_prob + result.away_win_prob
        assert abs(total - 1.0) < 1e-6, f"1X2 sum = {total}"

    def test_over_under_2_5_sums_to_one(self, result):
        total = result.over_2_5 + result.under_2_5
        assert abs(total - 1.0) < 1e-6, f"O2.5 + U2.5 = {total}"

    def test_over_under_1_5_sums_to_one(self, result):
        total = result.over_1_5 + result.under_1_5
        assert abs(total - 1.0) < 1e-6, f"O1.5 + U1.5 = {total}"

    def test_over_under_3_5_sums_to_one(self, result):
        total = result.over_3_5 + result.under_3_5
        assert abs(total - 1.0) < 1e-6, f"O3.5 + U3.5 = {total}"

    def test_over_ordering(self, result):
        """O1.5 >= O2.5 >= O3.5 always."""
        assert result.over_1_5 >= result.over_2_5 >= result.over_3_5, (
            f"Over ordering violated: O1.5={result.over_1_5:.3f}, "
            f"O2.5={result.over_2_5:.3f}, O3.5={result.over_3_5:.3f}"
        )

    def test_probabilities_in_bounds(self, result):
        """All probabilities must be in [0, 1]."""
        fields = [
            result.home_win_prob, result.draw_prob, result.away_win_prob,
            result.over_1_5, result.over_2_5, result.over_3_5,
            result.under_1_5, result.under_2_5, result.under_3_5,
            result.btts_prob,
            result.home_under_1_5_prob, result.away_under_1_5_prob,
        ]
        for p in fields:
            assert 0.0 <= p <= 1.0, f"Probability out of bounds: {p}"

    def test_n_simulations_stored(self, result):
        assert result.n_simulations == N_SIM

    def test_team_under_1_5_reasonable(self, result):
        assert result.home_under_1_5_prob >= 0.0
        assert result.away_under_1_5_prob >= 0.0
        assert result.home_under_1_5_prob <= 1.0
        assert result.away_under_1_5_prob <= 1.0


class TestSymmetry:
    """Equal xG should produce symmetric results."""

    def test_equal_xg_symmetric_1x2(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=123)
        result = sim.simulate(home_xg=1.3, away_xg=1.3)

        assert abs(result.home_win_prob - result.away_win_prob) < TOLERANCE, (
            f"Symmetry broken: H={result.home_win_prob:.3f}, A={result.away_win_prob:.3f}"
        )

    def test_equal_xg_symmetric_expected_goals(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=123)
        result = sim.simulate(home_xg=1.3, away_xg=1.3)

        assert abs(result.expected_home_goals - result.expected_away_goals) < 0.1


class TestHomeAdvantage:
    """Higher home xG should produce higher home win probability."""

    def test_home_advantage_reflected(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=2.0, away_xg=0.8)

        assert result.home_win_prob > result.away_win_prob, (
            f"Home advantage not reflected: H={result.home_win_prob:.3f}, "
            f"A={result.away_win_prob:.3f}"
        )
        # Strong home advantage — should be clearly dominant
        assert result.home_win_prob > 0.5

    def test_away_advantage_reflected(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=0.5, away_xg=2.5)

        assert result.away_win_prob > result.home_win_prob


class TestScorelines:
    """Scoreline probability tests."""

    def test_top_scorelines_count(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.5, away_xg=1.0)

        assert len(result.scoreline_probs) <= 15
        assert len(result.scoreline_probs) >= 5  # Should have at least some variety

    def test_scoreline_probs_are_valid(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.5, away_xg=1.0)

        for (h, a), prob in result.scoreline_probs.items():
            assert 0.0 < prob <= 1.0, f"Invalid scoreline prob ({h}-{a}): {prob}"
            assert isinstance(h, int) and isinstance(a, int)

    def test_scoreline_probs_sum_lte_one(self):
        """Sum of top scoreline probs must not exceed 1.0."""
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.5, away_xg=1.0)

        total = sum(result.scoreline_probs.values())
        assert total <= 1.0 + 1e-6, f"Scoreline probs sum > 1.0: {total}"

    def test_most_likely_scoreline_is_sensible(self):
        """For moderate xG, 1-0 or 1-1 should be among the top scorelines."""
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.3, away_xg=1.0)

        top_scores = sorted(result.scoreline_probs.items(), key=lambda x: -x[1])
        top_3 = [score for score, _ in top_scores[:3]]

        # At least one of 1-0, 0-0, 1-1 should be in top 3
        common = [(1, 0), (0, 0), (1, 1), (0, 1), (2, 1)]
        assert any(s in top_3 for s in common), (
            f"No common scoreline in top 3: {top_3}"
        )


class TestTailMass:
    """Tail mass tracks uncaptured probability."""

    def test_tail_mass_is_valid(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.5, away_xg=1.0)

        assert 0.0 <= result.tail_mass <= 1.0

    def test_tail_mass_plus_scorelines_equals_one(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.5, away_xg=1.0)

        total = sum(result.scoreline_probs.values()) + result.tail_mass
        assert abs(total - 1.0) < 0.01, f"Scorelines + tail != 1.0: {total}"

    def test_high_xg_has_more_tail_mass(self):
        """Higher xG spreads goals over more scorelines → more tail mass."""
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        low = sim.simulate(home_xg=0.5, away_xg=0.5)
        high = sim.simulate(home_xg=3.0, away_xg=3.0)

        assert high.tail_mass > low.tail_mass


class TestReproducibility:
    """Same seed should produce identical results."""

    def test_same_seed_same_result(self):
        r1 = MatchSimulator(n_simulations=N_SIM, seed=99).simulate(1.5, 1.0)
        r2 = MatchSimulator(n_simulations=N_SIM, seed=99).simulate(1.5, 1.0)

        assert r1.home_win_prob == r2.home_win_prob
        assert r1.draw_prob == r2.draw_prob
        assert r1.scoreline_probs == r2.scoreline_probs
        assert r1.match_type == r2.match_type

    def test_different_seed_different_result(self):
        r1 = MatchSimulator(n_simulations=N_SIM, seed=1).simulate(1.5, 1.0)
        r2 = MatchSimulator(n_simulations=N_SIM, seed=2).simulate(1.5, 1.0)

        # Results will likely differ
        assert r1.home_win_prob != r2.home_win_prob or r1.draw_prob != r2.draw_prob

    def test_stability_across_runs_with_50k_draws(self):
        r1 = MatchSimulator(n_simulations=N_SIM, seed=11).simulate(3.5, 0.3)
        r2 = MatchSimulator(n_simulations=N_SIM, seed=77).simulate(3.5, 0.3)

        assert abs(r1.home_win_prob - r2.home_win_prob) <= 0.005


class TestLambdaClamping:
    """Extreme inputs should be safely clamped."""

    def test_very_high_xg_clamped(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=10.0, away_xg=8.0)

        # Should produce valid probabilities despite extreme input
        total = result.home_win_prob + result.draw_prob + result.away_win_prob
        assert abs(total - 1.0) < 1e-6

    def test_very_low_xg_clamped(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=-1.0, away_xg=0.0)

        total = result.home_win_prob + result.draw_prob + result.away_win_prob
        assert abs(total - 1.0) < 1e-6
        # Very low xG → mostly 0-0 draws
        assert result.draw_prob > 0.3

    def test_zero_xg_safe(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=0.0, away_xg=0.0)

        # Clamped to 0.05 — should still produce valid results
        assert result.expected_home_goals > 0
        assert result.expected_away_goals > 0


class TestBTTS:
    """Both Teams To Score probability tests."""

    def test_high_xg_high_btts(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=2.5, away_xg=2.0)

        assert result.btts_prob > 0.6, f"BTTS too low for high xG: {result.btts_prob:.3f}"

    def test_low_xg_low_btts(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=0.05, away_xg=0.05)

        assert result.btts_prob < 0.1, f"BTTS too high for low xG: {result.btts_prob:.3f}"


class TestEntropy:
    """Entropy measures match uncertainty."""

    def test_balanced_match_high_entropy(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.2, away_xg=1.2)

        assert result.entropy > 1.3

    def test_dominant_match_lower_entropy(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        balanced = sim.simulate(home_xg=1.2, away_xg=1.2)
        dominant = sim.simulate(home_xg=3.5, away_xg=0.3)

        assert dominant.entropy < balanced.entropy

    def test_entropy_is_non_negative(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.5, away_xg=1.0)
        assert result.entropy >= 0


class TestMatchType:
    """Entropy-based match classification."""

    def test_balanced_match_classified(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.2, away_xg=1.2)

        assert result.match_type in ("high_uncertainty", "balanced")

    def test_dominant_match_classified(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=3.5, away_xg=0.3)

        assert result.match_type == "predictable"

    def test_match_type_is_valid_enum(self):
        sim = MatchSimulator(n_simulations=N_SIM, seed=42)
        result = sim.simulate(home_xg=1.5, away_xg=1.0)

        assert result.match_type in ("high_uncertainty", "balanced", "predictable")


class TestValidation:
    """Input validation tests."""

    def test_too_few_simulations_raises(self):
        with pytest.raises(ValueError, match="n_simulations must be >= 100"):
            MatchSimulator(n_simulations=50)

    def test_result_is_frozen(self):
        sim = MatchSimulator(n_simulations=1_000, seed=42)
        result = sim.simulate(1.5, 1.0)

        with pytest.raises(AttributeError):
            result.home_win_prob = 0.99  # type: ignore

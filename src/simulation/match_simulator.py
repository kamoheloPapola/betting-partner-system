"""
Monte Carlo Match Simulator.

Simulates thousands of match outcomes using Poisson-distributed goals
to produce stable probability estimates for:
- Match result (1X2)
- Scoreline probabilities (top 15)
- Over/Under goals thresholds
- Both Teams To Score (BTTS)
- Match entropy and uncertainty classification

Uses shared log-normal tempo correlation to model the non-independence
of home and away goal scoring in real football matches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)

__all__ = ["MatchSimulator", "SimulationResult", "clamp_lambda", "DEFAULT_N_SIMULATIONS"]

# Lambda guardrails (must match PoissonEngine conventions)
LAMBDA_MIN = 0.05
LAMBDA_MAX = 4.0

# Tempo correlation: log-normal noise models open/defensive game states.
# LogNormal(0, sigma) has mean=exp(sigma^2/2) ≈ 1.005 for sigma=0.1
# Guarantees tempo > 0 (unlike Normal which can go negative).
TEMPO_SIGMA = 0.1
LEAGUE_TEMPO_SIGMA: dict[str, float] = {
    "PL": 0.10,
    "BL1": 0.11,
    "FL1": 0.09,
    "PD": 0.09,
    "SA": 0.07,
}

# Max goals to track per team in scoreline matrix
MAX_GOALS = 10

# Top N scorelines to retain
TOP_SCORELINES = 15

# Entropy thresholds for match classification
ENTROPY_HIGH_UNCERTAINTY = 1.5
ENTROPY_BALANCED = 1.2

# Default draw count for Monte Carlo simulation.
DEFAULT_N_SIMULATIONS = 50_000


def clamp_lambda(x: float | np.ndarray) -> float | np.ndarray:
    """
    Global lambda clamping utility.

    Enforces λ ∈ [0.05, 4.0] everywhere — training evaluation,
    prediction pipeline, and simulator input.

    LightGBM Poisson can output slightly negative values.
    Zero or negative λ silently breaks Poisson sampling.

    Args:
        x: Scalar or array of lambda values.

    Returns:
        Clamped value(s) in [LAMBDA_MIN, LAMBDA_MAX].
    """
    return np.clip(x, LAMBDA_MIN, LAMBDA_MAX)


@dataclass(frozen=True)
class SimulationResult:
    """Immutable result container for a Monte Carlo match simulation."""

    # --- 1X2 Probabilities ---
    home_win_prob: float
    draw_prob: float
    away_win_prob: float

    # --- Scoreline Probabilities (top N) ---
    scoreline_probs: Dict[Tuple[int, int], float]

    # --- Over/Under Goals ---
    over_1_5: float
    over_2_5: float
    over_3_5: float
    under_1_5: float
    under_2_5: float
    under_3_5: float

    # --- BTTS ---
    btts_prob: float

    # --- Expected Values ---
    expected_home_goals: float
    expected_away_goals: float
    home_under_1_5_prob: float
    away_under_1_5_prob: float

    # --- Uncertainty ---
    entropy: float
    scoreline_entropy: float
    home_win_ci_90: float
    draw_ci_90: float
    away_win_ci_90: float
    match_type: str  # "high_uncertainty", "balanced", or "predictable"

    # --- Coverage ---
    tail_mass: float  # Probability mass NOT captured by top scorelines

    # --- Audit ---
    n_simulations: int
    rl_weights_applied: bool = False


class MatchSimulator:
    """
    Monte Carlo match simulator using independent Poisson draws
    with shared log-normal tempo correlation.

    Usage:
        sim = MatchSimulator(n_simulations=50_000, seed=42)
        result = sim.simulate(home_xg=1.72, away_xg=1.08)
        print(result.home_win_prob)   # e.g. 0.487
        print(result.scoreline_probs) # {(1, 0): 0.13, (2, 1): 0.11, ...}
        print(result.match_type)      # "balanced"
    """

    def __init__(
        self,
        n_simulations: int = DEFAULT_N_SIMULATIONS,
        max_goals: int = MAX_GOALS,
        top_scorelines: int = TOP_SCORELINES,
        tempo_sigma: float = TEMPO_SIGMA,
        league: str | None = None,
        seed: Optional[int] = None,
    ) -> None:
        if n_simulations < 100:
            raise ValueError(f"n_simulations must be >= 100, got {n_simulations}")

        self.n_simulations = n_simulations
        self.max_goals = max_goals
        self.top_scorelines = top_scorelines
        self.tempo_sigma = LEAGUE_TEMPO_SIGMA.get(league, tempo_sigma) if league else tempo_sigma
        self.rng = np.random.default_rng(seed)

    def simulate(
        self,
        home_xg: float,
        away_xg: float,
        rl_weights: Optional[dict[str, float]] = None,
    ) -> SimulationResult:
        """
        Simulate a match using Poisson-distributed goals.

        Args:
            home_xg: Expected home goals (lambda).
            away_xg: Expected away goals (lambda).

        Returns:
            SimulationResult with all derived probabilities.
        """
        weights = rl_weights or {}
        tempo_sigma = self.tempo_sigma * float(weights.get("tempo_sigma_scale", 1.0))
        home_xg = float(clamp_lambda(home_xg * float(weights.get("lambda_scale_home", 1.0))))
        away_xg = float(clamp_lambda(away_xg * float(weights.get("lambda_scale_away", 1.0))))

        # 2. Tempo correlation: log-normal shared noise factor per simulation.
        #    Models the reality that games have shared tempo (open vs defensive).
        #    LogNormal guarantees tempo > 0, unlike Normal which can go negative.
        # Asymmetric tempo: shared component drives correlation,
        # independent components allow divergence per team.
        sigma_shared = tempo_sigma * 0.8
        sigma_ind = tempo_sigma * 0.4
        shared = self.rng.lognormal(mean=0.0, sigma=sigma_shared, size=self.n_simulations)
        home_tempo = shared * self.rng.lognormal(mean=0.0, sigma=sigma_ind, size=self.n_simulations)
        away_tempo = shared * self.rng.lognormal(mean=0.0, sigma=sigma_ind, size=self.n_simulations)

        # 3. Per-simulation lambdas
        home_lambdas = home_xg * home_tempo
        away_lambdas = away_xg * away_tempo

        # 4. Draw goals from Poisson distributions (vectorized)
        home_goals = self.rng.poisson(home_lambdas)
        away_goals = self.rng.poisson(away_lambdas)

        # 5. Cap goals for scoreline tracking
        home_goals_capped = np.minimum(home_goals, self.max_goals)
        away_goals_capped = np.minimum(away_goals, self.max_goals)

        total_goals = home_goals + away_goals
        n = self.n_simulations

        # --- 1X2 ---
        home_win_prob = float(np.mean(home_goals > away_goals))
        draw_prob = float(np.mean(home_goals == away_goals))
        away_win_prob = float(np.mean(home_goals < away_goals))
        # 90% confidence intervals (Bernoulli normal approximation)
        home_win_ci_90 = float(1.645 * np.sqrt(home_win_prob * (1 - home_win_prob) / n))
        draw_ci_90 = float(1.645 * np.sqrt(draw_prob * (1 - draw_prob) / n))
        away_win_ci_90 = float(1.645 * np.sqrt(away_win_prob * (1 - away_win_prob) / n))

        # --- Over/Under (integer thresholds: >2 means ≥3, i.e. over 2.5) ---
        over_1_5 = float(np.mean(total_goals > 1))
        over_2_5 = float(np.mean(total_goals > 2))
        over_3_5 = float(np.mean(total_goals > 3))
        under_1_5 = float(np.mean(total_goals <= 1))
        under_2_5 = float(np.mean(total_goals <= 2))
        under_3_5 = float(np.mean(total_goals <= 3))

        # --- BTTS ---
        btts_prob = float(np.mean((home_goals > 0) & (away_goals > 0)))

        # --- Expected values ---
        expected_home = float(np.mean(home_goals))
        expected_away = float(np.mean(away_goals))
        home_under_1_5_prob = float(np.mean(home_goals <= 1))
        away_under_1_5_prob = float(np.mean(away_goals <= 1))

        # --- Scoreline probabilities (top N only) ---
        scoreline_probs = self._compute_scoreline_probs(
            home_goals_capped, away_goals_capped
        )

        # --- Tail mass: probability not captured by top scorelines ---
        tail_mass = round(1.0 - sum(scoreline_probs.values()), 4)

        # --- Scoreline entropy (measures goal volume diversity, not competitiveness) ---
        size = self.max_goals + 1
        counts_flat = np.zeros(size * size, dtype=np.int64)
        np.add.at(counts_flat,
                  home_goals_capped * size + away_goals_capped, 1)
        probs_flat = counts_flat[counts_flat > 0] / n
        scoreline_entropy = float(scipy_entropy(probs_flat, base=2))

        # --- 1X2 entropy (measures competitive uncertainty — used for match_type) ---
        p_1x2 = np.array([home_win_prob, draw_prob, away_win_prob])
        p_1x2_nonzero = p_1x2[p_1x2 > 0]
        match_entropy = float(scipy_entropy(p_1x2_nonzero, base=2))

        # --- Match type classification based on 1X2 entropy ---
        # 1X2 entropy range: 0 (certain) to log2(3)=1.585 (perfectly balanced).
        # Thresholds: predictable < 1.2 <= balanced < 1.5 <= high_uncertainty
        if match_entropy > 1.5:
            match_type = "high_uncertainty"
        elif match_entropy > 1.2:
            match_type = "balanced"
        else:
            match_type = "predictable"

        logger.debug(
            "Simulation complete: home_xg=%.2f, away_xg=%.2f → "
            "H=%.1f%% D=%.1f%% A=%.1f%% entropy=%.3f (%s) tail=%.3f",
            home_xg, away_xg,
            home_win_prob * 100, draw_prob * 100, away_win_prob * 100,
            match_entropy, match_type, tail_mass,
        )

        return SimulationResult(
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            scoreline_probs=scoreline_probs,
            over_1_5=over_1_5,
            over_2_5=over_2_5,
            over_3_5=over_3_5,
            under_1_5=under_1_5,
            under_2_5=under_2_5,
            under_3_5=under_3_5,
            btts_prob=btts_prob,
            expected_home_goals=expected_home,
            expected_away_goals=expected_away,
            home_under_1_5_prob=home_under_1_5_prob,
            away_under_1_5_prob=away_under_1_5_prob,
            entropy=match_entropy,
            scoreline_entropy=scoreline_entropy,
            home_win_ci_90=home_win_ci_90,
            draw_ci_90=draw_ci_90,
            away_win_ci_90=away_win_ci_90,
            match_type=match_type,
            tail_mass=tail_mass,
            n_simulations=self.n_simulations,
            rl_weights_applied=bool(rl_weights),
        )

    def _compute_scoreline_probs(
        self, home_goals: np.ndarray, away_goals: np.ndarray
    ) -> Dict[Tuple[int, int], float]:
        """
        Count scoreline frequencies and return the top N most common.

        Uses a 2D histogram for O(n) performance instead of Python Counter.
        """
        size = self.max_goals + 1
        counts = np.zeros((size, size), dtype=np.int64)

        # Vectorized 2D bincount via histogram
        np.add.at(counts, (home_goals, away_goals), 1)

        n = len(home_goals)

        # Flatten, sort by frequency, take top N
        flat_indices = np.argsort(counts.ravel())[::-1]
        result: Dict[Tuple[int, int], float] = {}

        for idx in flat_indices[: self.top_scorelines]:
            h_score = int(idx // size)
            a_score = int(idx % size)
            freq = counts[h_score, a_score]
            if freq == 0:
                break
            result[(h_score, a_score)] = round(float(freq / n), 4)

        return result

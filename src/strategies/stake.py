"""
Stake sizing utilities kept outside the core recommendation path.

This module remains available for manual workflows, but probability intelligence
stops before stake sizing.
"""

import logging
from typing import Dict, Optional, Tuple

from src.strategies.edge_engine import EdgeEngine

logger = logging.getLogger(__name__)

# === STAKE SIZING CONSTANTS ===
MAX_STAKE_FRACTION = 0.015
MAX_DAILY_MARKET_EXPOSURE = 0.07
DEFAULT_BANKROLL = 1000.0

# === TIER MULTIPLIERS ===
TIER_MULT = {
    "TIER_A": 1.0,
    "TIER_B": 0.6,
    "TIER_C": 0.4,
    "NO_BET": 0.0,
}

# === EV THRESHOLDS ===
MIN_EV = {
    "TIER_A": 0.05,
    "TIER_B": 0.08,
    "TIER_C": 0.12,
}

# === ODDS SANITY BOUNDS ===
ODDS_MIN = 1.30
ODDS_MAX = 6.00

_EDGE_ENGINE = EdgeEngine()


def calculate_ev(p_cal: float, odds: float) -> float:
    """Return expected value for stake calculations."""
    ev = _EDGE_ENGINE.compute_ev(p_cal, odds)
    return ev if ev is not None else 0.0


def passes_ev_filter(tier: str, p_cal: float, odds: float) -> Tuple[bool, float]:
    """
    Apply stake-specific EV filtering.

    This remains stricter than the core edge engine because it is intended for
    manual Kelly sizing, not recommendation eligibility.
    """
    if odds < ODDS_MIN or odds > ODDS_MAX:
        logger.debug("Odds %s outside bounds [%s, %s]", odds, ODDS_MIN, ODDS_MAX)
        return False, 0.0

    ev = calculate_ev(p_cal, odds)
    min_ev = MIN_EV.get(tier, 0.12)

    if ev < min_ev:
        logger.debug("%s EV=%.3f < min=%.3f", tier, ev, min_ev)
        return False, ev

    return True, ev


def calculate_kelly_stake(
    p_cal: float,
    odds: float,
    bankroll: float = DEFAULT_BANKROLL,
    tier: str = "TIER_A",
) -> float:
    """Calculate a tier-scaled fractional Kelly stake."""
    if odds <= 1.0:
        return 0.0

    f_kelly = (p_cal * odds - 1) / (odds - 1)
    f_kelly = max(0.0, min(f_kelly, MAX_STAKE_FRACTION))

    tier_mult = TIER_MULT.get(tier, 0.0)
    stake = bankroll * f_kelly * tier_mult
    return round(stake, 2)


class StakeManager:
    """Manual stake sizing with daily exposure tracking."""

    def __init__(self, bankroll: float = DEFAULT_BANKROLL):
        self.bankroll = bankroll
        self.daily_exposure: Dict[str, float] = {}
        self.current_date: Optional[str] = None

    def calculate_stake(
        self,
        p_cal: float,
        odds: float,
        tier: str,
        market: str,
    ) -> float:
        """Return a Kelly-style stake after EV and exposure checks."""
        from datetime import date

        today = date.today().isoformat()
        if self.current_date != today:
            self.daily_exposure = {}
            self.current_date = today

        passes, _ = passes_ev_filter(tier, p_cal, odds)
        if not passes:
            return 0.0

        stake = calculate_kelly_stake(p_cal, odds, self.bankroll, tier)
        if stake <= 0:
            return 0.0

        current_exposure = self.daily_exposure.get(market, 0.0)
        max_exposure = self.bankroll * MAX_DAILY_MARKET_EXPOSURE

        if current_exposure + stake > max_exposure:
            stake = max(0.0, max_exposure - current_exposure)
            if stake <= 0:
                logger.warning("%s: Daily exposure limit reached", market)
                return 0.0

        return round(stake, 2)

    def record_bet(self, market: str, stake: float):
        """Record stake for exposure tracking."""
        current = self.daily_exposure.get(market, 0.0)
        self.daily_exposure[market] = current + stake

    def get_remaining_exposure(self, market: str) -> float:
        """Get remaining daily exposure for a market."""
        max_exposure = self.bankroll * MAX_DAILY_MARKET_EXPOSURE
        used = self.daily_exposure.get(market, 0.0)
        return max(0.0, max_exposure - used)

    def set_bankroll(self, bankroll: float):
        """Update bankroll for manual staking workflows."""
        self.bankroll = bankroll
        logger.info("Bankroll updated to %.2f", bankroll)


_stake_manager: Optional[StakeManager] = None


def get_stake_manager(bankroll: float = DEFAULT_BANKROLL) -> StakeManager:
    """Get singleton stake manager for manual workflows."""
    global _stake_manager
    if _stake_manager is None:
        _stake_manager = StakeManager(bankroll)
    return _stake_manager

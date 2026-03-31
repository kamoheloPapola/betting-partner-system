"""
Immune Markets Configuration.

Markets in this set should NEVER be optimized further.
These represent high-ROI logic that is proven stable.

Rationale:
- Double Chance markets are derived and don't need tuning
- Team Goals U1.5 (Away) is dominant - don't risk destabilizing
"""
from typing import FrozenSet, Final

# Define public API
__all__ = ["IMMUNE_MARKETS", "is_immune"]

# --- IMMUNE MARKETS ---
# Do not optimize or modify prediction logic for these markets
IMMUNE_MARKETS: Final[FrozenSet[str]] = frozenset({
    # Double Chance (derived from 1x2, not tunable)
    "dc_1x",
    "dc_x2", 
    "dc_12",
    
    # High-ROI markets - proven stable, do not touch
    "away_under_1_5",
    "home_under_1_5",
})


def is_immune(market: str) -> bool:
    """
    Check if a market is immune from optimization.
    
    Args:
        market: Market identifier (e.g., 'dc_1x', 'away_under_1_5')
        
    Returns:
        True if market should not be modified.
    """
    return market.lower() in IMMUNE_MARKETS

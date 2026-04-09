"""
Core System Constants.

Defines immutable values used across the prediction system for match
statuses, data sources, and formatting conventions.

Note: For complete status sets (FINISHED_STATUSES, UPCOMING_STATUSES),
see src.config.leagues which provides frozenset collections.
"""
from typing import Final

# Define public API
__all__ = [
    # Match Statuses
    "STATUS_FT",
    "STATUS_AET",
    "STATUS_FINISHED",
    "STATUS_SCHEDULED",
    "STATUS_TIMED",
    "STATUS_IN_PLAY",
    # Data Sources
    "SOURCE_CSV",
    "SOURCE_ODDS_API",
    "SOURCE_FBREF",
    # Formatting
    "MATCH_SEPARATOR",
    # H2H Lift Policy
    "MAX_H2H_LIFT_DEFAULT",
    "MAX_H2H_LIFT_HIGH_SAMPLE",
    "H2H_HIGH_SAMPLE_THRESHOLD",
    # Resolver Lookback Policy
    "RESOLVER_LOOKBACK_DAYS",
]

# =============================================================================
# MATCH STATUSES
# =============================================================================
# Finished states
STATUS_FT: Final[str] = "FT"                  # Full Time (regular)
STATUS_AET: Final[str] = "AET"                # After Extra Time
STATUS_FINISHED: Final[str] = "FINISHED"      # Generic finished

# Upcoming/Live states
STATUS_SCHEDULED: Final[str] = "SCHEDULED"    # Match scheduled
STATUS_TIMED: Final[str] = "TIMED"            # Match time set
STATUS_IN_PLAY: Final[str] = "IN_PLAY"        # Currently playing

# =============================================================================
# DATA SOURCES
# =============================================================================
SOURCE_CSV: Final[str] = "csv_history"        # Football-data.co.uk historical CSVs
SOURCE_ODDS_API: Final[str] = "odds_api"      # The Odds API
SOURCE_FBREF: Final[str] = "fbref"            # FBRef scraper

# =============================================================================
# FORMATTING
# =============================================================================
MATCH_SEPARATOR: Final[str] = " vs "          # Home vs Away display separator

# =============================================================================
# H2H LIFT POLICY
# =============================================================================
MAX_H2H_LIFT_DEFAULT: Final[float] = 0.05      # 5pp default cap (conservative)
MAX_H2H_LIFT_HIGH_SAMPLE: Final[float] = 0.10  # 10pp cap for high-sample H2H
H2H_HIGH_SAMPLE_THRESHOLD: Final[int] = 8      # Matches required for high sample

# =============================================================================
# RESOLVER LOOKBACK POLICY
# =============================================================================
# Extended from 90 to 120 days after SYSTEM STOP incident (rolling_90d drift).
# Controls how far back the bandit loads resolved predictions for ECE training.
# Increase if leagues have long fixture gaps (e.g. international breaks).
RESOLVER_LOOKBACK_DAYS: Final[int] = 120


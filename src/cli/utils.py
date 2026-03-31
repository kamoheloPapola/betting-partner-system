"""
CLI Utility Functions.

Shared utilities for CLI commands including:
- LeagueCode enum for consistent league handling
- DateFilter for match date filtering
- MarketProbabilities type for structured probability data
- Helper functions for date/league resolution
"""
import logging
from enum import Enum
from typing import Dict, TypedDict, Optional, List, Union, Any

import pandas as pd
import pytz

from src.core.constants import MATCH_SEPARATOR
from src.config.leagues import LEAGUE_METADATA, FINISHED_STATUSES
from src.core.exceptions import DataValidationError

# Define public API
__all__ = [
    "LeagueCode",
    "DateFilter",
    "MarketProbabilities",
    "filter_matches_by_date",
    "resolve_league_code",
    "MATCH_SEPARATOR",  # Re-exported for convenience in CLI commands
]

logger = logging.getLogger(__name__)


class LeagueCode(str, Enum):
    """Supported soccer leagues with metadata integration."""
    PL = "PL"
    PD = "PD"
    SA = "SA"
    BL1 = "BL1"
    FL1 = "FL1"
    
    @property
    def full_name(self) -> str:
        """Get official competition name."""
        return LEAGUE_METADATA.get(self.value, {}).get('full_name', self.value)
    
    @property
    def country(self) -> str:
        """Get country/region."""
        return LEAGUE_METADATA.get(self.value, {}).get('country', 'Unknown')

    @classmethod
    def from_alias(cls, query: str) -> Optional["LeagueCode"]:
        """
        Fuzzy match a league code or name query to a valid LeagueCode using metadata aliases.
        Optimized to use a single pass through metadata.
        """
        if not query:
            return None
            
        q = query.strip().upper()
        
        # 1. Exact Code Match
        if q in cls.__members__:
            return cls[q]
            
        # 2. Alias & Partial Match in one pass
        partial_match = None
        for code_val, meta in LEAGUE_METADATA.items():
            # Alias match (priority)
            if q in [a.upper() for a in meta.get('aliases', [])]:
                return cls(code_val)
            
            # Name match (fallback)
            full_n = meta.get('full_name', '').upper()
            if q in full_n and not partial_match:
                try:
                    partial_match = cls(code_val)
                except ValueError:
                    continue
        
        return partial_match


def resolve_league_code(query: str) -> Optional[LeagueCode]:
    """Helper for backward compatibility and CLI resolution."""
    return LeagueCode.from_alias(query)


class DateFilter(str, Enum):
    """Supported date range filters for predictions matching CLI options."""
    YESTERDAY = "yesterday"
    TODAY = "today"
    TOMORROW = "tomorrow"
    WEEKEND = "weekend"
    WEEK = "week"
    MONTH = "month"
    ALL = "all"
    # Day-of-week filters
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


class MarketProbabilities(TypedDict):
    """
    Probability distribution for all supported betting markets.
    Keys are synchronized with prediction engine outputs.
    """
    home: float
    draw: float
    away: float
    u25: float
    o25: float
    btts: float
    home_under_1_5: float
    away_under_1_5: float
    over_1_5: float
    corn_u11: Optional[float]
    corn_o75: Optional[float]
    corn_1x2_h: Optional[float]
    corn_1x2_d: Optional[float]
    corn_1x2_a: Optional[float]
    card_u45: Optional[float]
    card_o25: Optional[float]
    card_u55: Optional[float]
    dc_1x: float
    dc_x2: float
    dc_12: float
    expected_home_goals: float
    expected_away_goals: float
    goal_model_home_lambda: float
    goal_model_away_lambda: float
    ensemble_divergence: bool
    divergence_pct: float
    mc_entropy: Optional[float]
    mc_tail_mass: Optional[float]
    mc_match_type: Optional[str]
    mc_n_simulations: Optional[float]
    mc_top_scorelines: List[Dict[str, Union[str, float]]]


def filter_matches_by_date(
    df: pd.DataFrame, 
    date_filter: Union[str, DateFilter], 
    show_all: bool = False, 
    user_timezone: str = 'UTC'
) -> pd.DataFrame:
    """
    Filter matches by date range in user's local timezone.
    
    Args:
        df: DataFrame with 'date' column (UTC) and 'status' column.
        date_filter: Range identifier or string from CLI.
        show_all: Override filter to show all future matches.
        user_timezone: IANA timezone name.
    
    Raises:
        DataValidationError: If critical columns missing or logic fails.
    """
    if df.empty:
        return df

    # 1. Column Validation
    required_cols = {'date', 'status'}
    missing = required_cols - set(df.columns)
    if missing:
        raise DataValidationError(f"Missing required columns for filtering: {missing}")

    # 2. Filter Normalization
    if isinstance(date_filter, str):
        try:
            date_filter = DateFilter(date_filter.lower())
        except ValueError:
            logger.warning(f"Invalid date filter '{date_filter}', defaulting to TODAY")
            date_filter = DateFilter.TODAY

    try:
        # 3. Timezone & Localization
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], utc=True)
        elif df['date'].dt.tz is None:
            df['date'] = df['date'].dt.tz_localize('UTC')
            
        try:
            if user_timezone.upper() == 'LOCAL':
                import datetime
                tz = datetime.datetime.now().astimezone().tzinfo
            else:
                tz = pytz.timezone(user_timezone)
        except (pytz.UnknownTimeZoneError, Exception) as e:
            logger.warning(f"Failed to resolve timezone '{user_timezone}': {e}, falling back to UTC")
            tz = pytz.UTC
            
        df['date_local'] = df['date'].dt.tz_convert(tz)
        df['status_upper'] = df['status'].str.upper()  # Normalize for FINISHED_STATUSES check
        now = pd.Timestamp.now(tz=tz)
        
        # 4. Filter Routing
        filters = {
            DateFilter.YESTERDAY: _filter_yesterday,
            DateFilter.TODAY: _filter_today,
            DateFilter.TOMORROW: _filter_tomorrow,
            DateFilter.WEEKEND: _filter_weekend,
            DateFilter.WEEK: _filter_week_range,
            DateFilter.MONTH: _filter_month_view,
            DateFilter.ALL: _filter_all_future,
            # Day-of-week filters
            DateFilter.MONDAY: lambda df, now: _filter_weekday(df, now, 0),
            DateFilter.TUESDAY: lambda df, now: _filter_weekday(df, now, 1),
            DateFilter.WEDNESDAY: lambda df, now: _filter_weekday(df, now, 2),
            DateFilter.THURSDAY: lambda df, now: _filter_weekday(df, now, 3),
            DateFilter.FRIDAY: lambda df, now: _filter_weekday(df, now, 4),
            DateFilter.SATURDAY: lambda df, now: _filter_weekday(df, now, 5),
            DateFilter.SUNDAY: lambda df, now: _filter_weekday(df, now, 6),
        }
        
        if show_all or date_filter == DateFilter.ALL:
            return _filter_all_future(df, now)
            
        filter_func = filters.get(date_filter, _filter_today)
        return filter_func(df, now)

    except Exception as e:
        raise DataValidationError(
            f"Date Filter logic failed: {e}",
            context={"filter_requested": date_filter, "timezone": user_timezone}
        )


# --- INTERNAL HELPERS (Reduced copies, optimized masks) ---

def _filter_today(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    today = now.normalize()
    tomorrow = today + pd.Timedelta(days=1)
    mask = (
        (df['date_local'] >= today) & 
        (df['date_local'] < tomorrow) & 
        (~df['status_upper'].isin(FINISHED_STATUSES))
    )
    return df[mask]

def _filter_yesterday(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    yesterday_start = (now - pd.Timedelta(days=1)).normalize()
    today_start = now.normalize()
    mask = (df['date_local'] >= yesterday_start) & (df['date_local'] < today_start)
    return df[mask]

def _filter_tomorrow(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    tomorrow_start = (now + pd.Timedelta(days=1)).normalize()
    next_day_start = tomorrow_start + pd.Timedelta(days=1)
    mask = (
        (df['date_local'] >= tomorrow_start) & 
        (df['date_local'] < next_day_start) & 
        (~df['status_upper'].isin(FINISHED_STATUSES))
    )
    return df[mask]

def _filter_weekend(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    current_weekday = now.dayofweek  # 0=Monday, 5=Saturday, 6=Sunday
    
    if current_weekday == 5:  # Sat
        saturday = now.normalize()
    elif current_weekday == 6:  # Sun
        saturday = (now - pd.Timedelta(days=1)).normalize()
    else:  # Mon-Fri
        saturday = (now + pd.Timedelta(days=5 - current_weekday)).normalize()
    
    monday = saturday + pd.Timedelta(days=2)
    mask = (
        (df['date_local'] >= saturday) & 
        (df['date_local'] < monday) & 
        (~df['status_upper'].isin(FINISHED_STATUSES))
    )
    return df[mask]

def _filter_all_future(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    start = (now - pd.Timedelta(days=2)).normalize()
    mask = (
        (df['date_local'] >= start) & 
        (~df['status_upper'].isin(FINISHED_STATUSES))
    )
    return df[mask]

def _filter_month_view(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    start = now.normalize()
    end = start + pd.Timedelta(days=30)
    mask = (
        (df['date_local'] >= start) & 
        (df['date_local'] < end) & 
        (~df['status_upper'].isin(FINISHED_STATUSES))
    )
    return df[mask]


def _filter_week_range(df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """
    Filter to current/upcoming Monday-Friday range, excluding weekends.
    
    If Mon-Fri: Today -> Friday 23:59:59
    If Sat-Sun: Next Monday -> Next Friday 23:59:59
    """
    current_weekday = now.dayofweek  # 0=Monday, 4=Friday, 5=Sat, 6=Sun
    
    if current_weekday <= 4:  # Mon-Fri
        start = now.normalize()
        # Days until Friday = 4 - current
        days_to_friday = 4 - current_weekday
        end = (start + pd.Timedelta(days=days_to_friday + 1))
    else:  # Sat-Sun
        days_to_monday = 7 - current_weekday
        start = (now + pd.Timedelta(days=days_to_monday)).normalize()
        end = (start + pd.Timedelta(days=5))  # Monday + 5 days = Saturday 00:00
        
    mask = (
        (df['date_local'] >= start) & 
        (df['date_local'] < end) & 
        (df['date_local'].dt.dayofweek < 5) &  # Extra safety: Monday-Friday only
        (~df['status_upper'].isin(FINISHED_STATUSES))
    )
    return df[mask]


def _filter_weekday(df: pd.DataFrame, now: pd.Timestamp, target_weekday: int) -> pd.DataFrame:
    """
    Filter to next occurrence of a specific weekday.
    
    Args:
        df: DataFrame with date_local column
        now: Current timestamp
        target_weekday: 0=Monday, 1=Tuesday, ..., 6=Sunday
    """
    current_weekday = now.dayofweek
    
    # Calculate days until target weekday
    if target_weekday >= current_weekday:
        days_ahead = target_weekday - current_weekday
    else:
        days_ahead = 7 - (current_weekday - target_weekday)
    
    # If today is the target weekday, show today
    if days_ahead == 0:
        target_start = now.normalize()
    else:
        target_start = (now + pd.Timedelta(days=days_ahead)).normalize()
    
    target_end = target_start + pd.Timedelta(days=1)
    
    mask = (
        (df['date_local'] >= target_start) & 
        (df['date_local'] < target_end) & 
        (~df['status_upper'].isin(FINISHED_STATUSES))
    )
    return df[mask]

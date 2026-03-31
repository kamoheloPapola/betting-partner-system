"""
League Standings Manager.

Reconstructs league tables at any historical point in time using
canonical match results for standings context in strategy decisions.
"""
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.utils.naming import normalize_team_name

# Define public API
__all__ = ["StandingsManager"]

logger = logging.getLogger(__name__)

# === Constants ===
# Rolling form window (last N matches)
RECENT_FORM_WINDOW = 5

# Momentum divergence thresholds
STRONG_POSITIVE_THRESHOLD = 0.6
MILD_POSITIVE_THRESHOLD = 0.2
STRONG_NEGATIVE_THRESHOLD = -0.6
MILD_NEGATIVE_THRESHOLD = -0.2

# Minimum matches for momentum calculation
MIN_MATCHES_FOR_MOMENTUM = 3

# Zone cutoffs (top/bottom N teams)
TOP_ZONE_SIZE = 4
BOTTOM_ZONE_SIZE = 3

# Points for match outcomes
WIN_POINTS = 3
DRAW_POINTS = 1
LOSS_POINTS = 0


class StandingsManager:
    """
    Reconstructs league tables at any point in time using canonical results.
    
    Strictly follows deterministic ordering and safety assertions.
    """
    
    MASTER_PATH: Path = DATA_DIR / "results" / "normalized" / "results_master.csv"

    def __init__(self, master_path: Optional[Path] = None) -> None:
        self.master_path = master_path or self.MASTER_PATH
        self._master_df: Optional[pd.DataFrame] = None

    def _load_data(self) -> pd.DataFrame:
        """Load and cache the master results file with date parsing."""
        if self._master_df is None:
            if not self.master_path.exists():
                logger.warning(f"Master results file not found at {self.master_path}")
                return pd.DataFrame()
            
            df = pd.read_csv(self.master_path, low_memory=False)
            
            # Robust date parsing with mixed format support
            dt_series = pd.to_datetime(df['match_date'], errors='coerce')
            invalid_mask = dt_series.isna() & df['match_date'].notna()
            
            if invalid_mask.any():
                # Fallback for '%Y-%m-%d %H:%M:%S' which sometimes confuses auto-infer
                dt_series.loc[invalid_mask] = pd.to_datetime(
                    df.loc[invalid_mask, 'match_date'], 
                    format='%Y-%m-%d %H:%M:%S', 
                    errors='coerce'
                )
            
            df['match_date'] = dt_series
            df = df.dropna(subset=['match_date'])
            
            # Type cast season to numeric
            if 'season' in df.columns:
                df['season'] = pd.to_numeric(df['season'], errors='coerce')

            # Canonical Name Normalization
            df['home_team'] = df['home_team'].apply(normalize_team_name)
            df['away_team'] = df['away_team'].apply(normalize_team_name)
            
            self._master_df = df
        return self._master_df

    def get_table(
        self, 
        league: str, 
        season: int, 
        as_of_date: datetime
    ) -> Dict[str, Dict[str, Any]]:
        """
        Reconstruct the league table as it stood BEFORE as_of_date.
        
        Args:
            league: League code (e.g., 'PL', 'BL1').
            season: Season start year (e.g., 2024 for 2024/25).
            as_of_date: Cutoff date (exclusive - no matches on this date).
            
        Returns:
            Dictionary mapping team names to their standings data.
        """
        df = self._load_data()
        if df.empty:
            return {}

        # 1. Scope Filtering
        target_league = str(league).strip()
        target_season = float(season)
        
        mask = (df['league'].astype(str).str.strip() == target_league) & \
               (df['season'].astype(float) == target_season)
        
        league_df = df[mask].copy()

        if league_df.empty:
            return {}

        # 2. Safety Assertion: No Look-Ahead
        as_of_ts = pd.Timestamp(as_of_date).tz_localize(None)
        past_matches = league_df[league_df['match_date'] < as_of_ts].copy()
        
        if past_matches.empty:
            return {}

        # 3. Deterministic Sort
        past_matches = past_matches.sort_values(["match_date", "match_hash"])

        # 4. Table Calculation
        stats = self._calculate_stats(past_matches)

        if not stats:
            return {}

        # 5. Build Final Table
        return self._build_final_table(stats)

    def _calculate_stats(self, matches: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate team statistics from match results."""
        stats: Dict[str, Dict[str, Any]] = {}

        def update_team(team: str, scored: int, conceded: int) -> None:
            if team not in stats:
                stats[team] = {
                    'points': 0, 'played': 0, 'gs': 0, 'gc': 0, 'gd': 0,
                    'recent_results': []
                }
            
            stats[team]['played'] += 1
            stats[team]['gs'] += scored
            stats[team]['gc'] += conceded
            stats[team]['gd'] += (scored - conceded)
            
            if scored > conceded:
                match_points = WIN_POINTS
            elif scored == conceded:
                match_points = DRAW_POINTS
            else:
                match_points = LOSS_POINTS
                
            stats[team]['points'] += match_points
            
            # Maintain rolling window
            stats[team]['recent_results'].append(match_points)
            if len(stats[team]['recent_results']) > RECENT_FORM_WINDOW:
                stats[team]['recent_results'].pop(0)

        for _, row in matches.iterrows():
            update_team(row['home_team'], row['home_goals'], row['away_goals'])
            update_team(row['away_team'], row['away_goals'], row['home_goals'])

        return stats

    def _build_final_table(self, stats: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Build the final standings table with momentum bands."""
        league_size = len(stats)
        standings_list: List[Dict[str, Any]] = []
        
        for team, data in stats.items():
            played = data['played']
            points = data['points']
            
            # Season PPG
            season_ppg = points / played if played > 0 else 0.0
            
            # Recent PPG (last N matches)
            recent_results = data.get('recent_results', [])
            recent_points = sum(recent_results)
            recent_count = len(recent_results)
            recent_ppg = recent_points / recent_count if recent_count > 0 else 0.0
            
            # Divergence & Momentum Band
            divergence = recent_ppg - season_ppg
            momentum_band = self._calculate_momentum_band(divergence, recent_count)

            standings_list.append({
                'team': team,
                'points': points,
                'played': played,
                'gd': data['gd'],
                'gs': data['gs'],
                'season_ppg': round(season_ppg, 2),
                'recent_ppg': round(recent_ppg, 2),
                'divergence': round(divergence, 2),
                'momentum_band': momentum_band,
                'momentum_volatility': round(abs(divergence), 2)
            })

        # Deterministic sort: Points -> GD -> GS -> Team Name
        standings_list.sort(key=lambda x: (-x['points'], -x['gd'], -x['gs'], x['team']))

        # Build final map with status bands
        final_table: Dict[str, Dict[str, Any]] = {}
        for i, entry in enumerate(standings_list):
            rank = i + 1
            status_band = self._calculate_status_band(rank, league_size)

            final_table[entry['team']] = {
                'rank': rank,
                'points': entry['points'],
                'played': entry['played'],
                'gd': entry['gd'],
                'gs': entry['gs'],
                'status_band': status_band,
                'season_ppg': entry['season_ppg'],
                'recent_ppg': entry['recent_ppg'],
                'divergence': entry['divergence'],
                'momentum_band': entry['momentum_band'],
                'momentum_volatility': entry['momentum_volatility']
            }

        return final_table

    @staticmethod
    def _calculate_momentum_band(divergence: float, recent_count: int) -> str:
        """Determine momentum band based on PPG divergence."""
        if recent_count < MIN_MATCHES_FOR_MOMENTUM:
            return "INSUFFICIENT_DATA"
        if divergence >= STRONG_POSITIVE_THRESHOLD:
            return "STRONG_POSITIVE"
        if divergence >= MILD_POSITIVE_THRESHOLD:
            return "MILD_POSITIVE"
        if divergence <= STRONG_NEGATIVE_THRESHOLD:
            return "STRONG_NEGATIVE"
        if divergence <= MILD_NEGATIVE_THRESHOLD:
            return "MILD_NEGATIVE"
        return "STABLE"

    @staticmethod
    def _calculate_status_band(rank: int, league_size: int) -> str:
        """Determine league position band (TOP/MID/LOW)."""
        if rank <= TOP_ZONE_SIZE:
            return "TOP"
        if rank >= (league_size - BOTTOM_ZONE_SIZE + 1):
            return "LOW"
        return "MID"


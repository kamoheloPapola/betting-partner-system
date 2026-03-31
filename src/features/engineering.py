"""
Feature Engineering Module.

Generates predictive features from historical match data including:
- Rolling statistics (goals, xG, corners, cards)
- Form ratings with exponential decay
- Rest days between matches
- Double chance market targets

Enforces strict no look-ahead bias through shift(1) operations.
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import List, Optional, Dict, Tuple

from dataclasses import dataclass, field
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for feature engineering defaults.
    """
    DAYS_REST_DEFAULT: int = 7
    DAYS_REST_MAX: int = 21
    FORM_NEUTRAL: float = 1.5
    ROLLING_WINDOW_DEFAULT: int = 5
    FORM_DECAY_DEFAULT: float = 0.9
    SEASON_START_MONTH: int = 8
    SEASON_TOTAL_MATCHES: int = 38
    METRIC_COLUMNS: Dict[str, Tuple[str, str]] = field(default_factory=lambda: {
        'goals': ('goals_scored', 'goals_conceded'),
        'xg': ('xg_scored', 'xg_conceded'),
        'cards': ('cards_scored', 'cards_conceded'),
        'corners': ('corners_scored', 'corners_conceded')
    })
FeatureDefaults = FeatureConfig()

class FeatureEngineer:
    """
    Generates predictive features from historical match data.
    Enforces no look-ahead bias.
    """

    # -------------------------------------------------------------
    # FEATURE ENGINEERING UNLOCKED: H2H Features Added (2026-01-13)
    # For Cards O2.5 Calibration Fix - JUVENTUS vs CREMONESE issue
    # -------------------------------------------------------------
    LOCKED = False

    def __init__(self) -> None:
        pass

    def _get_date_column(self, df: pd.DataFrame) -> str:
        """Get the date column name, with validation."""
        if 'date' in df.columns:
            return 'date'
        elif 'match_date' in df.columns:
            return 'match_date'
        else:
            raise ValueError(
                f"No date column found. Expected 'date' or 'match_date'. "
                f"Available: {df.columns.tolist()}"
            )

    def validate_features(self, df: pd.DataFrame) -> None:
        """Validate feature integrity and check for look-ahead bias."""
        if df.empty:
            return
            
        # Check 1: Non-negative rest days
        if 'days_rest' in df.columns:
            neg_rest = (df['days_rest'] < 0).sum()
            if neg_rest > 0:
                raise ValueError(f"Found {neg_rest} records with negative days_rest")
        
        # Check 2: Rolling stats shouldn't exist for first match
        rolling_cols = [c for c in df.columns if c.startswith('rolling_')]
        if rolling_cols and 'team_id' in df.columns:
            date_col = self._get_date_column(df)
            first_matches = df.groupby('team_id')[date_col].transform('min') == df[date_col]
            
            for col in rolling_cols:
                has_stats_on_debut = df.loc[first_matches, col].notna().any()
                if has_stats_on_debut:
                    raise ValueError(
                        f"Look-ahead bias detected: {col} has values on team debut matches"
                    )

        # Check 3: Form rating should be within valid range
        if 'form_rating' in df.columns:
            invalid_form = ((df['form_rating'] < 0) | (df['form_rating'] > 3)).sum()
            if invalid_form > 0:
                raise ValueError(f"Found {invalid_form} records with invalid form_rating (must be 0-3)")
                
    def _sanitize_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and sanitize identifier columns."""
        df = df.copy()
        targets = [c for c in ['team_id', 'home_team', 'away_team'] if c in df.columns]
        
        for col in targets:

            # Whitelist approach: allow alphanumeric, spaces, hyphens, underscores, dots, and apostrophes
            invalid_mask = ~df[col].astype(str).str.match(r"^[\w\s\-\.']+$", na=False)
            
            if invalid_mask.any():
                invalid_ids = df.loc[invalid_mask, col].unique()[:5]  # Show first 5
                raise ValueError(
                    f"Invalid characters in {col}: {invalid_ids}. "
                    f"Only alphanumeric, spaces, hyphens, underscores, and dots allowed."
                )

        return df

    def calculate_rolling_stats(
        self, 
        df: pd.DataFrame, 
        window: int = FeatureDefaults.ROLLING_WINDOW_DEFAULT, 
        metrics: List[str] = ['goals', 'xg']
    ) -> pd.DataFrame:
        """
        Calculates rolling averages for team performance with strict validation.
        
        Raises:
            ValueError: If required metric columns are missing (ISSUE #1)
        """
        df = df.copy()  # ISSUE #7: Prevent side effects
        
        # Column mapping for different metric types (ISSUE #8)
        METRIC_COLUMNS = FeatureDefaults.METRIC_COLUMNS
        
        # Ensure we are sorted by team_id and date to prevent chronological leakage
        date_col = self._get_date_column(df)
        df = df.sort_values(['team_id', date_col])
        
        # Build required column list based on mapping
        required_cols = []
        for m in metrics:
            cols = METRIC_COLUMNS.get(m, (f"{m}_scored", f"{m}_conceded"))
            required_cols.extend(list(cols))
            
        # Validate required columns (ISSUE #1)
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
            raise ValueError(
                f"Missing required columns for rolling stats: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        # We group only by team_id to allow cross-season continuity
        # FIX: Loop through columns to ensure correct naming (transform doesn't rename automatically)
        # Preserve NaNs for upcoming matches - they should NOT inherit rolling stats
        # Only filled values come from actual historical data via shift(1)
        for col in required_cols:
            df[f"rolling_{col}_{window}"] = df.groupby('team_id')[col].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
        
        # Calculate Rolling Confidence and Density features
        df[f'rolling_games_played_{window}'] = df.groupby('team_id')[date_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).count()
        ).fillna(0)
        
        df[f'rolling_confidence_{window}'] = df[f'rolling_games_played_{window}'] / window
        
        # Compute span in days to get rolling density
        dates = pd.to_datetime(df[date_col])
        days_array = dates.astype('int64') // (10**9 * 86400)
        temp_df = pd.DataFrame({'team_id': df['team_id'], 'days': days_array})
        
        df[f'rolling_days_span_{window}'] = temp_df.groupby('team_id')['days'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).max() - x.shift(1).rolling(window=window, min_periods=1).min()
        ).fillna(0)
        
        # Calculate density (games played / days span, +1 to avoid div by zero)
        df[f'rolling_density_{window}'] = df[f'rolling_games_played_{window}'] / (df[f'rolling_days_span_{window}'] + 1)
        
        # Track max source date to assert NO leakage
        # .rolling().max() doesn't officially support datetime64[ns] in older pandas, so we cast to int64, roll, then cast back.
        df[f'_rolling_max_source_date_{window}'] = pd.to_datetime(
            df.groupby('team_id')[date_col].transform(
                lambda x: x.astype('int64').shift(1).rolling(window=window, min_periods=1).max()
            ), 
            errors='coerce'
        ).where(df[f'rolling_games_played_{window}'] > 0, pd.NaT)
        
        return df

    def calculate_std_stats(
        self,
        df: pd.DataFrame,
        metrics: List[str] = ['goals', 'xg'],
        smoothing: int = 5
    ) -> pd.DataFrame:
        """
        Calculates Season-To-Date (STD) averages, blending with the previous season's average
        to stabilize early-season features and eliminate early-season noise.
        """
        df = df.copy()
        
        if 'season' not in df.columns:
            import logging
            logging.getLogger(__name__).warning("Missing 'season' column; skipping calculate_std_stats.")
            return df
            
        METRIC_COLUMNS = FeatureDefaults.METRIC_COLUMNS
        date_col = self._get_date_column(df)
        df = df.sort_values(['team_id', date_col])
        
        required_cols = []
        for m in metrics:
            cols = METRIC_COLUMNS.get(m, (f"{m}_scored", f"{m}_conceded"))
            required_cols.extend(list(cols))
            
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for STD stats: {missing}")

        # 1. Calculate historical team-season averages for the blending factor
        season_stats = df.groupby(['team_id', 'season'])[required_cols].mean().reset_index()
        season_stats = season_stats.sort_values(['team_id', 'season'])
        
        for col in required_cols:
            season_stats[f"last_season_{col}"] = season_stats.groupby('team_id')[col].shift(1)
        
        season_stats = season_stats.drop(columns=required_cols)
        df = df.merge(season_stats, on=['team_id', 'season'], how='left')
        
        # Fill missing prior seasons with global averages for cold-start teams
        for col in required_cols:
            global_avg = df[col].mean()
            df[f"last_season_{col}"] = df[f"last_season_{col}"].fillna(global_avg)
            
        # 2. Compute STD Current Season (expanding mean per season)
        df['std_games_played'] = df.groupby(['team_id', 'season'])[date_col].transform(
            lambda x: x.shift(1).expanding(min_periods=1).count()
        ).fillna(0)
        
        df['std_confidence'] = df['std_games_played'] / FeatureDefaults.SEASON_TOTAL_MATCHES
        
        for col in required_cols:
            current_std = df.groupby(['team_id', 'season'])[col].transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
            
            n_games = df['std_games_played']
            last_season_avg = df[f"last_season_{col}"]
            
            # Blended formulate: (current_std * n_games + last_season_avg * k) / (n_games + k)
            blended = (current_std * n_games + last_season_avg * smoothing) / (n_games + smoothing)
            df[f'std_{col}'] = blended
            
            # If current_std is NaN (Match 1), blended is NaN. Fill with last_season_avg.
            df[f'std_{col}'] = df[f'std_{col}'].fillna(last_season_avg)
            
            # Clean up
            df = df.drop(columns=[f"last_season_{col}"])
            
        # Track max source date for STD features to assert NO leakage
        df['_std_max_source_date'] = pd.to_datetime(
            df.groupby(['team_id', 'season'])[date_col].transform(
                lambda x: x.astype('int64').shift(1).expanding(min_periods=1).max()
            ),
            errors='coerce'
        ).where(df['std_games_played'] > 0, pd.NaT)
            
        return df

    def calculate_differentials(
        self, 
        df: pd.DataFrame, 
        metrics: List[str],
        strict: bool = False
    ) -> pd.DataFrame:
        """Calculate Home - Away differences for specified metrics."""
        df = df.copy()  # ISSUE #7: Prevent side effects
        
        if not metrics:
            return df
        
        created = []
        missing = []
        
        for metric in metrics:
            home_col = f"home_{metric}"
            away_col = f"away_{metric}"
            diff_col = f"{metric}_diff"
            
            if home_col in df.columns and away_col in df.columns:
                df[diff_col] = df[home_col] - df[away_col]
                created.append(diff_col)
            else:
                missing.append(metric)
        
        if strict and missing:
            raise ValueError(f"Missing columns for metrics: {missing}")
        elif missing:
            # Log at debug level so it doesn't spam
            import logging
            logging.getLogger(__name__).debug(
                f"Skipped differentials for missing metrics: {missing}"
            )
        return df

    def calculate_form(self, df: pd.DataFrame, decay: float = FeatureDefaults.FORM_DECAY_DEFAULT) -> pd.DataFrame:
        """
        Calculate exponentially weighted form rating.
        
        Logic:
        1. Convert results to points (W=3, D=1, L=0)
        2. Calculate EWM on historical results only (shift prevents look-ahead)
        3. Preserve NaN for upcoming matches (no history = no form)
        4. Default to neutral (1.5) for teams with no prior matches
        
        Args:
            df: Team-level DataFrame with goals_scored/conceded
            decay: EWM decay factor (0.9 = emphasize recent form)
        
        Returns:
            DataFrame with 'form_rating' column added
        """
        df = df.copy()  # ISSUE #7: Prevent side effects
        
        # Ensure we are sorted by date
        date_col = self._get_date_column(df)
        df = df.sort_values(date_col)
        
        # 1. Calculate points (ISSUE #4: Type-Safe Points)
        from typing import Optional
        
        def get_points(goals_scored: float, goals_conceded: float) -> Optional[float]:
            """Convert match result to points (W=3.0, D=1.0, L=0.0)."""
            if pd.isna(goals_scored) or pd.isna(goals_conceded):
                return None
            if goals_scored > goals_conceded:
                return 3.0
            elif goals_scored == goals_conceded:
                return 1.0
            else:
                return 0.0
        
        df['points'] = df.apply(
            lambda row: get_points(row['goals_scored'], row['goals_conceded']),
            axis=1
        )
        
        # 2. Calculate EWM per team (Using transform for robustness)
        # FIX: Split transform and ffill to ensure correct context
        # EWM alpha parameter: alpha = 1-decay
        # High decay (0.9) -> Low alpha (0.1) -> More weight on recent matches
        # Low decay (0.5) -> High alpha (0.5) -> Smoother averaging
        df['form_rating'] = df.groupby('team_id')['points'].transform(
            lambda x: (
                x.shift(1)
                .ewm(alpha=1-decay, min_periods=1, adjust=False)
                .mean()
                .where(x.shift(1).notna())  # Preserve NaN for upcoming matches - no forward-fill
            )
        )
        
        
        # Fill remaining NaNs with neutral
        df['form_rating'] = df['form_rating'].fillna(FeatureDefaults.FORM_NEUTRAL)
        
        # Clean up
        df = df.drop(columns=['points'])
        
        return df

    def calculate_rest_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate days since last match for the team.
        
        For first match of season, defaults to FeatureDefaults.DAYS_REST_DEFAULT.
        Capped at FeatureDefaults.DAYS_REST_MAX to prevent international break anomalies.
        """
        df = df.copy()  # ISSUE #7: Prevent side effects
        

        
        date_col = self._get_date_column(df)
        df = df.sort_values(date_col)
        
        df['last_match_date'] = df.groupby('team_id')[date_col].shift(1)
        df['days_rest'] = (df[date_col] - df['last_match_date']).dt.days
        
        # Fill missing with default (ISSUE #3)
        df['days_rest'] = df['days_rest'].fillna(FeatureDefaults.DAYS_REST_DEFAULT)
        
        # Cap at max to prevent international break anomalies (ISSUE #3)
        df['days_rest'] = df['days_rest'].clip(upper=FeatureDefaults.DAYS_REST_MAX)
        
        return df.drop(columns=['last_match_date'])

    def add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add contextual features like time of season, day of week, etc."""
        df = df.copy()
        
        date_col = self._get_date_column(df)
        
        # Day of week (weekday vs weekend effect)
        df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Month (seasonality effects)
        df['month'] = pd.to_datetime(df[date_col]).dt.month
        
        # Position in season (early/mid/late season dynamics)
        # August start approximation (0.0 = Start of season, 1.0 = End)
        df['season_progress'] = (pd.to_datetime(df[date_col]).dt.month - FeatureDefaults.SEASON_START_MONTH) % 12 / 12
        
        return df

    def add_referee_features(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Add referee context features at match level.

        Features:
        - referee_id: canonical referee identifier
        - referee_card_rate_10: rolling mean of total cards handled by the referee
          over the previous `window` matches (shifted to avoid leakage)
        """
        df = df.copy()
        date_col = self._get_date_column(df)
        df = df.sort_values(date_col)

        # Canonical referee key
        if 'referee_id' in df.columns:
            referee_series = df['referee_id']
        elif 'referee' in df.columns:
            referee_series = df['referee']
        else:
            referee_series = pd.Series(pd.NA, index=df.index)

        df['referee_id'] = referee_series.astype('string').str.strip()
        df.loc[df['referee_id'].isin(['', 'nan', 'None', '<NA>']), 'referee_id'] = pd.NA
        df['referee_id'] = df['referee_id'].fillna('UNKNOWN')

        # Match-level cards total (yellow + red where available)
        if {'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 'away_red_cards'}.issubset(df.columns):
            cards_total = (
                pd.to_numeric(df['home_yellow_cards'], errors='coerce').fillna(0.0)
                + pd.to_numeric(df['away_yellow_cards'], errors='coerce').fillna(0.0)
                + pd.to_numeric(df['home_red_cards'], errors='coerce').fillna(0.0)
                + pd.to_numeric(df['away_red_cards'], errors='coerce').fillna(0.0)
            )
        elif {'home_cards', 'away_cards'}.issubset(df.columns):
            cards_total = (
                pd.to_numeric(df['home_cards'], errors='coerce').fillna(0.0)
                + pd.to_numeric(df['away_cards'], errors='coerce').fillna(0.0)
            )
        elif {'home_total_cards', 'away_total_cards'}.issubset(df.columns):
            cards_total = (
                pd.to_numeric(df['home_total_cards'], errors='coerce').fillna(0.0)
                + pd.to_numeric(df['away_total_cards'], errors='coerce').fillna(0.0)
            )
        elif 'match_total_cards' in df.columns:
            cards_total = pd.to_numeric(df['match_total_cards'], errors='coerce').fillna(0.0)
        else:
            cards_total = pd.Series(0.0, index=df.index, dtype=float)

        df['_ref_match_cards_total'] = cards_total.astype(float)

        df[f'referee_card_rate_{window}'] = df.groupby('referee_id')['_ref_match_cards_total'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        baseline = float(df['_ref_match_cards_total'].mean()) if len(df) else 0.0
        if pd.isna(baseline):
            baseline = 0.0
        df[f'referee_card_rate_{window}'] = df[f'referee_card_rate_{window}'].fillna(baseline)

        return df.drop(columns=['_ref_match_cards_total'])

    def _get_stadium_coordinate_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Return the first available (lat, lon) column pair."""
        candidates = [
            ('stadium_latitude', 'stadium_longitude'),
            ('stadium_lat', 'stadium_lon'),
            ('stadium_lat', 'stadium_lng'),
            ('latitude', 'longitude'),
            ('lat', 'lon'),
        ]
        for lat_col, lon_col in candidates:
            if lat_col in df.columns and lon_col in df.columns:
                return lat_col, lon_col
        return None, None

    def _fetch_openweather_snapshot(
        self,
        lat: float,
        lon: float,
        match_date: pd.Timestamp,
        api_key: str,
        timeout_seconds: float = 3.0,
    ) -> Tuple[float, float]:
        """
        Fetch pre-match weather snapshot from OpenWeatherMap forecast API.

        Returns:
            (precipitation_mm, wind_speed_kmh)
        """
        try:
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/forecast",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": api_key,
                    "units": "metric",
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json() or {}
            points = payload.get("list", [])
            if not points:
                return 0.0, 0.0

            match_ts = pd.to_datetime(match_date, errors='coerce', utc=True)
            if pd.isna(match_ts):
                return 0.0, 0.0

            selected = None
            selected_ts = None
            for item in points:
                point_ts = pd.to_datetime(item.get("dt"), unit="s", errors='coerce', utc=True)
                if pd.isna(point_ts):
                    continue
                if point_ts <= match_ts and (selected_ts is None or point_ts > selected_ts):
                    selected = item
                    selected_ts = point_ts

            if selected is None:
                selected = points[0]

            rain_3h = float((selected.get("rain") or {}).get("3h", 0.0) or 0.0)
            snow_3h = float((selected.get("snow") or {}).get("3h", 0.0) or 0.0)
            precipitation_mm = max(0.0, rain_3h + snow_3h)

            wind_ms = float((selected.get("wind") or {}).get("speed", 0.0) or 0.0)
            wind_speed_kmh = max(0.0, wind_ms * 3.6)
            return precipitation_mm, wind_speed_kmh
        except Exception as exc:
            logger.debug("OpenWeatherMap lookup failed: %s", exc)
            return 0.0, 0.0

    def add_weather_features(
        self,
        df: pd.DataFrame,
        api_key: Optional[str] = None,
        timeout_seconds: float = 3.0,
    ) -> pd.DataFrame:
        """
        Add pre-match weather context features.

        Features:
        - precipitation_mm
        - wind_speed_kmh

        Fail-safe:
        - Returns 0.0 for both fields when API key/coords are missing or API is unavailable.
        """
        df = df.copy()
        df['precipitation_mm'] = 0.0
        df['wind_speed_kmh'] = 0.0

        resolved_api_key = (
            api_key
            or os.getenv("OPENWEATHERMAP_API_KEY")
            or os.getenv("OPENWEATHER_API_KEY")
        )
        if not resolved_api_key:
            return df

        lat_col, lon_col = self._get_stadium_coordinate_columns(df)
        if lat_col is None or lon_col is None:
            return df

        date_col = self._get_date_column(df)
        cache: Dict[Tuple[float, float, str], Tuple[float, float]] = {}

        for idx, row in df.iterrows():
            lat = pd.to_numeric(row.get(lat_col), errors='coerce')
            lon = pd.to_numeric(row.get(lon_col), errors='coerce')
            match_date = pd.to_datetime(row.get(date_col), errors='coerce', utc=True)

            if pd.isna(lat) or pd.isna(lon) or pd.isna(match_date):
                continue

            key = (round(float(lat), 4), round(float(lon), 4), match_date.isoformat())
            if key not in cache:
                cache[key] = self._fetch_openweather_snapshot(
                    lat=float(lat),
                    lon=float(lon),
                    match_date=match_date,
                    api_key=resolved_api_key,
                    timeout_seconds=timeout_seconds,
                )

            precipitation_mm, wind_speed_kmh = cache[key]
            df.at[idx, 'precipitation_mm'] = float(precipitation_mm)
            df.at[idx, 'wind_speed_kmh'] = float(wind_speed_kmh)

        return df

    def add_team_availability_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pre-match squad availability index.

        Stub implementation for Phase B:
        - team_availability_pct = 1.0 (full squad)
        """
        df = df.copy()
        df['team_availability_pct'] = 1.0
        return df

    def normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different data sources.
        
        Mappings:
        - home_goals -> home_score
        - away_goals -> away_score
        - match_date -> date
        - match_hash -> match_id (if match_id is missing)
        """
        df = df.copy()  # Prevent side effects
        
        rename_map = {
            'home_goals': 'home_score',
            'away_goals': 'away_score',
            'match_date': 'date'
        }
        
        # Only rename columns that exist
        existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        
        if existing_renames:
            df = df.rename(columns=existing_renames)

        # Ensure canonical match identifier exists for downstream joins.
        if 'match_id' not in df.columns:
            if 'match_hash' in df.columns:
                df['match_id'] = df['match_hash']
            else:
                df['match_id'] = df.index.astype(str)
        elif df['match_id'].isna().any():
            if 'match_hash' in df.columns:
                df['match_id'] = df['match_id'].fillna(df['match_hash'])
            df['match_id'] = df['match_id'].fillna(df.index.astype(str))
        
        return df

    def transform_match_to_team_rows(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """
        Explodes a match row (Home vs Away) into two rows (Home Team, Away Team)
        to facilitate team-level feature engineering.
        """
        # Normalize first (ISSUE #6: Enforced naming)
        matches_df = self.normalize_column_names(matches_df)
        
        # Security: Input Sanitization
        matches_df = self._sanitize_input(matches_df)
        
        # Check source DataFrame once for optional features
        has_xg = 'home_xg' in matches_df.columns and 'away_xg' in matches_df.columns
        has_corners = 'home_corners' in matches_df.columns and 'away_corners' in matches_df.columns
        has_cards = 'home_total_cards' in matches_df.columns and 'away_total_cards' in matches_df.columns

        # Add match identifier if missing (to preserve relationships)
        if 'match_hash' not in matches_df.columns and 'match_id' not in matches_df.columns:
             # Create a temporary match ID from index if real ID missing
             matches_df['match_hash'] = matches_df.index.astype(str)
        
        # Home side
        home = matches_df.copy()
        home['_match_row_type'] = 'home'
        home['team_id'] = home['home_team']
        home['opponent_id'] = home['away_team']
        
        # Scores (guaranteed normalized to _score)
        home['goals_scored'] = home['home_score']
        home['goals_conceded'] = home['away_score']
        
        if has_xg:
            home['xg_scored'] = home['home_xg']
            home['xg_conceded'] = home['away_xg']
        
        home['is_home'] = 1
        
        # Away side
        away = matches_df.copy()
        away['_match_row_type'] = 'away'
        away['team_id'] = away['away_team']
        away['opponent_id'] = away['home_team']
        away['goals_scored'] = away['away_score']
        away['goals_conceded'] = away['home_score']
        
        if has_xg:
            away['xg_scored'] = away['away_xg']
            away['xg_conceded'] = away['home_xg']
        
        # Standardize ID column (already done by normalize_column_names if match_hash existed)
        # Standardize date column (already done by normalize_column_names if match_date existed)
        
        # CORNERS (ISSUE #8: Cleaner Terminology - scored/conceded)
        if has_corners:
            home['corners_scored'] = home['home_corners']
            home['corners_conceded'] = home['away_corners']
            away['corners_scored'] = away['away_corners']
            away['corners_conceded'] = away['home_corners']
            
        # CARDS (Total) (ISSUE #8: Cleaner Terminology - received/drawn)
        if has_cards:
            home['cards_scored'] = home['home_total_cards']
            home['cards_conceded'] = home['away_total_cards']
            away['cards_scored'] = away['away_total_cards']
            away['cards_conceded'] = away['home_total_cards']
            
        away['is_home'] = 0
        
        team_df = pd.concat([home, away], ignore_index=True)
        # date is guaranteed normalized (handled by _get_date_column internal validation)
        date_col = self._get_date_column(team_df)
        team_df = team_df.sort_values(date_col)
        
        return team_df

    def create_double_chance_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variables for double chance betting markets.
        
        Markets:
        - 1X: Home win OR draw (home_score >= away_score)
        - X2: Away win OR draw (away_score >= home_score)  
        - 12: Either team wins (home_score != away_score)
        
        Only calculated for completed matches (non-null scores).
        Upcoming matches have NaN targets (excluded from training).
        
        Args:
            df: Match-level DataFrame with home_score, away_score
        
        Returns:
            DataFrame with target_1x, target_x2, target_12 columns
        """
        df = df.copy()  # ISSUE #7: Prevent side effects
        
        # Validate required columns
        required = ['home_score', 'away_score']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        # Identify completed matches (ISSUE #5: Precise Targeting)
        mask_played = df['home_score'].notna() & df['away_score'].notna()
        
        # Initialize target columns with NaN
        df['target_1x'] = np.nan
        df['target_x2'] = np.nan
        df['target_12'] = np.nan
        
        if not mask_played.any():
            return df
            
        # Calculate targets using explicit boolean logic (ISSUE #5)
        df.loc[mask_played, 'target_1x'] = (
            df.loc[mask_played, 'home_score'] >= df.loc[mask_played, 'away_score']
        ).astype(int)
        
        df.loc[mask_played, 'target_x2'] = (
            df.loc[mask_played, 'away_score'] >= df.loc[mask_played, 'home_score']
        ).astype(int)
        
        df.loc[mask_played, 'target_12'] = (
            df.loc[mask_played, 'home_score'] != df.loc[mask_played, 'away_score']
        ).astype(int)
        
        return df

    def calculate_h2h_features(
        self, 
        df: pd.DataFrame, 
        window: int = 5
    ) -> pd.DataFrame:
        """
        Calculate head-to-head (H2H) statistics between teams.
        
        Features computed:
        - h2h_avg_cards: Average total cards in H2H meetings
        - h2h_avg_goals: Average total goals in H2H meetings  
        - h2h_cards_o25_rate: Rate of H2H games with >2.5 cards
        - h2h_match_count: Number of historical H2H meetings
        
        Uses shift(1) to prevent look-ahead bias.
        
        Args:
            df: Match-level DataFrame with home_team, away_team, date
            window: Number of recent H2H matches to consider
            
        Returns:
            DataFrame with H2H features added
        """
        df = df.copy()
        date_col = self._get_date_column(df)
        
        # Ensure date is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col)
        
        # Create canonical matchup key (alphabetically sorted teams)
        def get_h2h_key(row):
            teams = sorted([str(row['home_team']), str(row['away_team'])])
            return f"{teams[0]}__vs__{teams[1]}"
        
        df['_h2h_key'] = df.apply(get_h2h_key, axis=1)
        
        # Calculate total cards for each match (if available)
        has_cards = 'home_total_cards' in df.columns and 'away_total_cards' in df.columns
        if has_cards:
            df['_match_total_cards'] = df['home_total_cards'].fillna(0) + df['away_total_cards'].fillna(0)
        elif 'total_cards' in df.columns:
            df['_match_total_cards'] = df['total_cards']
        else:
            df['_match_total_cards'] = np.nan
        
        # Calculate total corners for each match (if available)
        has_corners = 'home_corners' in df.columns and 'away_corners' in df.columns
        if has_corners:
            df['_match_total_corners'] = df['home_corners'].fillna(0) + df['away_corners'].fillna(0)
        else:
            df['_match_total_corners'] = np.nan
            
        # Calculate total goals
        df['_match_total_goals'] = df['home_score'].fillna(0) + df['away_score'].fillna(0)
        
        # Initialize H2H columns
        df['h2h_avg_cards'] = np.nan
        df['h2h_avg_goals'] = np.nan
        df['h2h_avg_corners'] = np.nan
        df['h2h_goals_o25_rate'] = np.nan
        df['h2h_btts_rate'] = np.nan
        df['h2h_cards_o25_rate'] = np.nan
        df['h2h_corners_u115_rate'] = np.nan
        df['h2h_match_count'] = 0
        
        # Group by H2H matchup and compute rolling stats
        # Use shift(1) to exclude current match
        for h2h_key in df['_h2h_key'].unique():
            mask = df['_h2h_key'] == h2h_key
            h2h_df = df.loc[mask].copy()
            
            if len(h2h_df) < 2:
                # Need at least 1 prior meeting to compute H2H
                continue
            
            # Rolling average cards (shifted to prevent look-ahead)
            cards_shifted = h2h_df['_match_total_cards'].shift(1)
            df.loc[mask, 'h2h_avg_cards'] = cards_shifted.expanding(min_periods=1).mean()
            
            # Rolling average corners (shifted to prevent look-ahead)
            corners_shifted = h2h_df['_match_total_corners'].shift(1)
            df.loc[mask, 'h2h_avg_corners'] = corners_shifted.expanding(min_periods=1).mean()
            
            # Rolling average goals
            goals_shifted = h2h_df['_match_total_goals'].shift(1)
            df.loc[mask, 'h2h_avg_goals'] = goals_shifted.expanding(min_periods=1).mean()
            
            # Goals O2.5 rate (for Total Goals markets)
            goals_o25 = (goals_shifted > 2.5).astype(float)
            df.loc[mask, 'h2h_goals_o25_rate'] = goals_o25.expanding(min_periods=1).mean()
            
            # BTTS rate (Both Teams To Score)
            home_scored_shifted = h2h_df['home_score'].shift(1)
            away_scored_shifted = h2h_df['away_score'].shift(1)
            btts = ((home_scored_shifted > 0) & (away_scored_shifted > 0)).astype(float)
            df.loc[mask, 'h2h_btts_rate'] = btts.expanding(min_periods=1).mean()
            
            # Cards O2.5 rate (using expanding window)
            cards_o25 = (cards_shifted > 2.5).astype(float)
            df.loc[mask, 'h2h_cards_o25_rate'] = cards_o25.expanding(min_periods=1).mean()
            
            # Corners U11.5 rate (using expanding window)
            corners_u115 = (corners_shifted < 11.5).astype(float)
            df.loc[mask, 'h2h_corners_u115_rate'] = corners_u115.expanding(min_periods=1).mean()
            
            # Match count (prior meetings)
            df.loc[mask, 'h2h_match_count'] = range(len(h2h_df))
        
        # Clean up temporary columns
        df = df.drop(columns=['_h2h_key', '_match_total_cards', '_match_total_goals', '_match_total_corners'])
        
        # Fill NaN H2H with league defaults (no prior meetings)
        # Use neutral values that won't strongly influence predictions
        df['h2h_avg_cards'] = df['h2h_avg_cards'].fillna(df['home_total_cards'].mean() + df['away_total_cards'].mean() if has_cards else 4.0)
        df['h2h_avg_corners'] = df['h2h_avg_corners'].fillna(df['home_corners'].mean() + df['away_corners'].mean() if has_corners else 10.0)
        df['h2h_avg_goals'] = df['h2h_avg_goals'].fillna(2.5)  # League average
        df['h2h_goals_o25_rate'] = df['h2h_goals_o25_rate'].fillna(0.53)  # League base rate
        df['h2h_btts_rate'] = df['h2h_btts_rate'].fillna(0.54)  # League base rate
        df['h2h_cards_o25_rate'] = df['h2h_cards_o25_rate'].fillna(0.77)  # Base rate
        df['h2h_corners_u115_rate'] = df['h2h_corners_u115_rate'].fillna(0.72)  # Base rate
        
        return df


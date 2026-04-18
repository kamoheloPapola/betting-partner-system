"""
Feature Pipeline Module.

Orchestrates the feature engineering workflow from raw match data to
ML-ready features. Includes intelligent caching and data validation.

Key Classes:
    FeaturePipeline: Main entry point with run(), transform() methods.
"""
import pandas as pd
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict, Union, Callable, TypeAlias
from src.cli.utils import LeagueCode

# Type Aliases for code clarity (ISSUE #2)
MatchData: TypeAlias = Union[pd.DataFrame, List[Dict[str, Any]], None]

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_DIR
from src.core.exceptions import DataValidationError
from src.data.validators import ProcessedMatch
from src.features.engineering import FeatureEngineer
# from src.data.merger import TeamMerger # To be integrated later if needed

logger = logging.getLogger(__name__)

class FeaturePipeline:
    # -------------------------------------------------------------
    # PIPELINE UNLOCKED: H2H Features Added (2026-01-13)
    # For Cards O2.5 Calibration Fix - JUVENTUS vs CREMONESE issue
    # -------------------------------------------------------------
    LOCKED = False
    VALID_LEAGUES = {'PL', 'PD', 'SA', 'BL1', 'FL1'}
    CACHE_TTL_MINUTES = 5  # Adjust based on data update frequency
    MAX_CACHE_SIZE = 10  # Maximum cached league datasets
    MASTER_FEATURE_ROTATION_KEEP = 3

    class StaticPriors:
        """
        Baseline statistics for imputation.
        
        Sources:
        - rolling_goals: Historical avg from 2018-2024 PL/PD/SA/BL1/FL1 (N=15,423)
        - rolling_corners:  data 2020-2024
        - rolling_cards: Calculated from normalized yellow card rates
        
        Last updated: 2024-12-21
        Review frequency: Quarterly
        """
        ROLLING_GOALS = 1.35
        ROLLING_CORNERS = 4.5
        ROLLING_CARDS = 2.2
        FORM_NEUTRAL = 1.0
        REST_DAYS_DEFAULT = 7.0

    class FeatureColumns:
        """Explicit feature whitelist for pipeline stability (ISSUE #7)"""
        BASE_METRICS = [
            'rolling_goals_scored_3', 'rolling_goals_conceded_3',
            'rolling_goals_scored_5', 'rolling_goals_conceded_5',
            'rolling_goals_scored_10', 'rolling_goals_conceded_10',
        ]
        
        XG_METRICS = [
            'rolling_xg_scored_3', 'rolling_xg_conceded_3',
            'rolling_xg_scored_5', 'rolling_xg_conceded_5',
            'rolling_xg_scored_10', 'rolling_xg_conceded_10',
        ]
        
        CORNER_METRICS = [
            'rolling_corners_scored_3', 'rolling_corners_conceded_3',
            'rolling_corners_scored_5', 'rolling_corners_conceded_5',
            'rolling_corners_scored_10', 'rolling_corners_conceded_10',
        ]
        
        CARD_METRICS = [
            'rolling_cards_scored_3', 'rolling_cards_conceded_3',
            'rolling_cards_scored_5', 'rolling_cards_conceded_5',
            'rolling_cards_scored_10', 'rolling_cards_conceded_10',
        ]
        
        # H2H features for card calibration (2026-01-13)
        H2H_METRICS = [
            'h2h_avg_cards', 'h2h_avg_goals', 
            'h2h_cards_o25_rate', 'h2h_match_count'
        ]
        
        META_FEATURES = ['form_rating', 'days_rest', 'team_availability_pct', 'team_id', 'match_id']
        
        @classmethod
        def get_home_columns(cls, include_xg: bool = False, include_corners: bool = False, include_cards: bool = False) -> List[str]:
            cols = cls.BASE_METRICS + cls.META_FEATURES
            if include_xg:
                cols += cls.XG_METRICS
            if include_corners:
                cols += cls.CORNER_METRICS
            if include_cards:
                cols += cls.CARD_METRICS
            return cols
    
    def __init__(self, data_loader: Optional[Callable[[Optional[str]], pd.DataFrame]] = None):
        self.engineer = FeatureEngineer()
        self.data_loader = data_loader or self._default_loader
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamp: Dict[str, datetime] = {}
    
    def _should_invalidate_cache(self, cache_key: str) -> bool:
        """
        Check if cache should be invalidated based on:
        1. File modification times
        2. TTL expiration
        """
        if cache_key not in self._cache:
            return True
        
        cache_time = self._cache_timestamp[cache_key]
        
        # Check TTL
        age = datetime.now() - cache_time
        if age > timedelta(minutes=self.CACHE_TTL_MINUTES):
            logger.info(f"[Pipeline] Cache expired for {cache_key} (age: {age})")
            return True
        
        # Check file modification times
        matches_dir = PROCESSED_DATA_DIR / "matches"
        if not matches_dir.exists():
            return False
        
        pattern = f"{cache_key}_*.csv" if cache_key != "global" else "*.csv"
        
        for csv_file in matches_dir.glob(pattern):
            file_mtime = datetime.fromtimestamp(csv_file.stat().st_mtime)
            if file_mtime > cache_time:
                logger.info(
                    f"[Pipeline] Cache invalidated: {csv_file.name} modified "
                    f"at {file_mtime} (cache from {cache_time})"
                )
                return True
        
        return False
    
    def _evict_lru_cache(self):
        """Remove least recently used cache entry when limit reached."""
        if len(self._cache) >= self.MAX_CACHE_SIZE:
            oldest_key = min(self._cache_timestamp, key=self._cache_timestamp.get)
            
            # Calculate cache size for logging
            cache_mb = sum(
                df.memory_usage(deep=True).sum() / 1024 / 1024 
                for df in self._cache.values()
            )
            
            del self._cache[oldest_key]
            del self._cache_timestamp[oldest_key]
            
            logger.info(
                f"[Pipeline] LRU eviction: {oldest_key} "
                f"(total cache: {cache_mb:.1f} MB)"
            )
        
    def _default_loader(self, league: str = None) -> pd.DataFrame:
        """
        Load all normalized matches from data/processed/matches/*.csv.
        API-Football -> Raw -> Normalizer -> CSV -> HERE.
        Optional: Filter by league (filename pattern: {league}_*.csv).
        """
        matches_dir = PROCESSED_DATA_DIR / "matches"
        if not matches_dir.exists():
            logger.warning(f"[Pipeline] No processed match data found at {matches_dir}")
            return pd.DataFrame()

        pattern = f"{league}_*.csv" if league else "*.csv"
        # Sort files to ensure deterministic loading order
        all_files = sorted(list(matches_dir.glob(pattern)))
        logger.info(f"[Pipeline] Found {len(all_files)} files matching {pattern}")
        
        if not all_files:
            logger.warning(f"[Pipeline] No CSV files found in processed/matches/ matching {pattern}")
            return pd.DataFrame()

        dfs = []
        for p in all_files:
            try:
                # Read CSV
                df = pd.read_csv(p)
                
                # 0.7. SCHEMA DEFENSE: Extract season/competition from filename if missing or misaligned
                # Expected format: LEAGUE_SEASON.csv (e.g. PL_2024.csv)
                try:
                    parts = p.stem.split('_')
                    file_league = parts[0]
                    file_season_str = parts[1]
                    
                    if 'competition' not in df.columns or df['competition'].isnull().all():
                        df['competition'] = file_league
                    
                    # If season is missing or not a number (e.g. contains 'FT')
                    if 'season' not in df.columns or not pd.to_numeric(df['season'], errors='coerce').notnull().all():
                        if file_season_str.isdigit():
                            df['season'] = int(file_season_str)
                except (IndexError, ValueError) as e:
                    logger.warning(
                        f"Could not parse league/season from filename: {p.name}",
                        extra={"error": str(e), "stem": p.stem}
                    )
                    # Skip file if critical metadata parsing fails to prevent downstream pollution
                    continue
                
                # Track source for debugging data loss
                df['source_file'] = p.name
                
                logger.debug(f"[Pipeline] Loaded {p.name} ({len(df)} rows)")
                dfs.append(df)
            except Exception as e:
                raise DataValidationError(
                    f"Error reading processed match CSV: {p.name}",
                    context={"file_path": str(p), "error": str(e)}
                )
        
        if not dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Consistent Competition Mapping for Backtest/Audit
        if 'competition' in combined_df.columns:
             # Normalized mapping (Bug 3.10 Fix) using LeagueCode
             normalized_map = {lc.value: lc.full_name for lc in LeagueCode}
             combined_df['competition'] = combined_df['competition'].replace(normalized_map)
        
        if 'date' in combined_df.columns:
            before_count = len(combined_df)
            combined_df['date'] = pd.to_datetime(combined_df['date'], errors='coerce', utc=True)
            
            # ISSUE #5: Catch Silent Data Loss
            invalid_dates = combined_df[combined_df['date'].isna()]
            if not invalid_dates.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Create quarantine directory
                quarantine_dir = DATA_DIR / "quarantine"
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                
                logger.error(
                    f"Found {len(invalid_dates)} matches with invalid dates",
                    extra={
                        "sample_ids": invalid_dates['match_id'].head(3).tolist() if 'match_id' in invalid_dates else [],
                        "files_affected": invalid_dates['source_file'].unique().tolist() if 'source_file' in invalid_dates else "Unknown"
                    }
                )
                
                # Save to quarantine for investigation
                quarantine_path = quarantine_dir / f"invalid_dates_{timestamp}.csv"
                invalid_dates.to_csv(quarantine_path, index=False)
                logger.warning(f"Invalid data preservation: Saved to {quarantine_path}")

            combined_df = combined_df.dropna(subset=['date'])
            after_count = len(combined_df)
            
            if before_count != after_count:
                 logger.warning(f"[Pipeline] Dropped {before_count - after_count} matches due to invalid dates.")

        # DEDUPLICATION (Fix for Bug 3.12 & Issue #8)
        combined_df = self._deduplicate_matches(combined_df)

        # Extract season range for summary
        seasons = []
        for d in dfs:
            if 'season' in d.columns:
                seasons.extend(d['season'].dropna().unique().tolist())
        # Normalize to strings for consistent comparison
        seasons_str = sorted(set(str(s) for s in seasons))
        season_range = f"Seasons {seasons_str[0]}–{seasons_str[-1]}" if seasons_str else ""
        
        # Include league in summary if available
        league_tag = f"[{league}]" if league else ""
        logger.info(f"[Pipeline]{league_tag} Loaded {len(dfs)} files | {len(combined_df):,} matches | {season_range}")
        return combined_df

    def process_matches(
        self, 
        matches: MatchData
    ) -> pd.DataFrame:
        """
        Normalize match data to DataFrame format.
        
        Args:
            matches: Either a DataFrame or list of match dictionaries
            
        Returns:
            Normalized DataFrame with required columns
            
        Raises:
            DataValidationError: If schema validation fails
        """
        if isinstance(matches, list):
             # Legacy/Mock path or direct injection
             # Validate schema if coming from list of dicts
             processed = [ProcessedMatch(**m).model_dump() for m in matches]
             df = pd.DataFrame(processed)
        else:
             df = matches

        if 'match_hash' in df.columns and 'match_id' not in df.columns:
            df['match_id'] = df['match_hash']
            
        # Ensure date is datetime
        date_col = 'date' if 'date' in df.columns else 'match_date'
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Sort by date essential for rolling stats
        df = df.sort_values(date_col)
        
        # Sanitization (Issue #10)
        df = self._sanitize_dataframe(df)
        
        return df

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply robust sanitization to the DataFrame:
        1. Trim leading/trailing whitespace from string columns.
        2. Remove control characters (ASCII 0-31, 127).
        3. Validate identifier whitelist (team_id, home_team, away_team).
        """
        df = df.copy()
        
        # 1 & 2: Trim and remove control characters from string columns
        string_cols = df.select_dtypes(include=['object']).columns
        for col in string_cols:
            # Remove control characters using regex
            # \x00-\x1F are control chars, \x7F is DEL
            df[col] = df[col].astype(str).str.replace(r'[\x00-\x1F\x7F]', '', regex=True).str.strip()
            
            # Action 8.1: Global Normalization (Uppercase identifiers)
            if col in ['team_id', 'home_team', 'away_team', 'status', 'league', 'competition']:
                df[col] = df[col].str.upper()

        # 3. Identifier validation (reusing logic from FeatureEngineer)
        id_targets = ['team_id', 'home_team', 'away_team', 'match_id', 'competition']
        present_targets = [c for c in id_targets if c in df.columns]
        
        for col in present_targets:

            # Whitelist: Alphanumeric, spaces, hyphens, underscores, DOTS for abbreviations (St. Pauli, 1. FC)
            invalid_mask = ~df[col].astype(str).str.match(r'^[\w\s\-\.]+$', na=False)
            
            if invalid_mask.any():
                invalid_samples = df.loc[invalid_mask, col].unique()[:3]
                raise DataValidationError(
                    f"Sanitization error in '{col}': Invalid characters found in samples {invalid_samples}. "
                    "Only alphanumeric, spaces, hyphens, underscores, and dots are allowed."
                )

                
        return df

    def load_stored_matches(self, league: str = None) -> pd.DataFrame:
        """
        Public API for loading matches, delegating to the configured loader.
        This enables dependency injection for testing (ISSUE #9).
        """
        return self.data_loader(league)

    @staticmethod
    def _deduplicate_matches(df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate matches with the following priority (ISSUE #8):
        1. Status: FT/FINISHED > others
        2. Completeness: Non-null scores > null scores
        3. Recency: Latest date wins
        
        Returns deduplicated DataFrame.
        """
        if 'match_id' not in df.columns:
            logger.warning("No match_id column, skipping deduplication")
            return df
        
        pre_count = len(df)
        
        # Priority ranking
        # FT/FINISHED = 0 (Highest)
        # LIVE = 1
        # SCHEDULED/TBD = 2
        # Others = 3
        # Action 8.3: Robust Case-Insensitive Deduplication
        status_norm = df['status'].str.upper()
        df['_status_priority'] = status_norm.map({
            'FT': 0, 
            'FINISHED': 0, 
            'LIVE': 1, 
            'SCHEDULED': 2,
            'TBD': 2
        }).fillna(3)
        
        # Score completeness check
        # Prefer rows that actually have scores (historical) over upcoming placeholders
        if 'home_score' in df.columns:
            df['_has_score'] = (~df['home_score'].isna()).astype(int)
        else:
            df['_has_score'] = 0
        
        # Sort: best status first (ascending), with score first (descending), latest date first (descending)
        df = df.sort_values(
            by=['_status_priority', '_has_score', 'date'],
            ascending=[True, False, False]
        )
        
        df = df.drop_duplicates(subset=['match_id'], keep='first')
        df = df.drop(columns=['_status_priority', '_has_score'])
        
        post_count = len(df)
        if pre_count != post_count:
            logger.info(f"[Pipeline] Deduplicated matches. Dropped {pre_count - post_count} duplicates.")
        
        return df

    def _validate_feature_completeness(
        self,
        df: pd.DataFrame,
        expected_groups: List[str]
    ) -> None:
        """
        Validate that all expected feature groups are present.
        
        Raises:
            DataValidationError: If critical features are missing
        """
        missing_features = []
        
        feature_map = {
            'BASE': self.FeatureColumns.BASE_METRICS,
            'XG': self.FeatureColumns.XG_METRICS,
            'CORNERS': self.FeatureColumns.CORNER_METRICS,
            'CARDS': self.FeatureColumns.CARD_METRICS,
        }
        
        for group in expected_groups:
            if group not in feature_map:
                continue
            
            for metric in feature_map[group]:
                home_col = f'home_{metric}'
                away_col = f'away_{metric}'
                
                if home_col not in df.columns:
                    missing_features.append(f"{group}: {home_col}")
                if away_col not in df.columns:
                    missing_features.append(f"{group}: {away_col}")
        
        if missing_features:
            raise DataValidationError(
                f"Feature engineering incomplete. Missing {len(missing_features)} features: "
                f"{missing_features[:5]}..."  # Show first 5
            )
        
        logger.info(f"[Pipeline] Feature validation passed. Groups: {expected_groups}")

    def _load_and_process(self, match_data: MatchData = None, league: str = None) -> pd.DataFrame:
        """
        Internal processing logic (formerly run).
        """
        if match_data is not None:
            logger.info(f"[Pipeline] Processing in-memory data")
            df = self.process_matches(match_data)
        else:
            logger.info(f"[Pipeline] Loading data from storage (Processed CSVs)... League: {league or 'All'}")
            df = self.load_stored_matches(league=league)
            if df.empty:
                logger.warning("No data found. Returning empty DataFrame.")
                return pd.DataFrame()
            df = self.process_matches(df)

        # 0.5. Load and Merge FBref Enrichment
        # Updated to new path data/fbref/processed
        enrich_dir = DATA_DIR / "fbref" / "processed"
        fbref_files = list(enrich_dir.glob("fbref_*.csv"))
        
        if fbref_files:
            try:
                enrich_dfs = []
                for f in fbref_files:
                    enrich_dfs.append(pd.read_csv(f))
                    
                if enrich_dfs:
                    enrich_df = pd.concat(enrich_dfs, ignore_index=True)
                    # Ensure match_id present
                    if 'match_id' in enrich_df.columns:
                        # Deduplicate (latest wins)
                        enrich_df = enrich_df.drop_duplicates(subset=['match_id'], keep='last')
                        
                        cols_to_use = ['match_id', 'home_xg', 'away_xg', 'home_shots', 'away_shots', 'home_possession', 'away_possession']
                        # Filter existing columns
                        cols_to_use = [c for c in cols_to_use if c in enrich_df.columns]
                        
                        if len(cols_to_use) > 1:
                            logger.info(f"Merging FBref features for {len(enrich_df)} matches from {len(fbref_files)} files...")
                            df = pd.merge(df, enrich_df[cols_to_use], on='match_id', how='left')
                            
                            # Update metadata for rows that got enrichment?
                            # Pipeline doesn't handle metadata column per row usually, just passes data.
                            # Trainer handles metadata summary.
                        else:
                            logger.warning("FBref enrichment files found but missing meaningful columns.")
                    else:
                        logger.warning("FBref enrichment CSVs missing 'match_id'.")
            except Exception as e:
                logger.error(f"Error merging FBref data: {e}")
        else:
            logger.info("No FBref enrichment data found. Proceeding with BASE features.")

        # 1. Transform to Team-Centric view
        team_df = self.engineer.transform_match_to_team_rows(df)
        
        # 2. Add Features
        logger.info("Calculating Rolling Stats...")
        
        metrics = ['goals']
        # Feature Group Check
        if 'xg_scored' in team_df.columns and 'xg_conceded' in team_df.columns:
            metrics.append('xg')
            logger.info("Feature Group Active: BASE + XG")
        else:
            logger.info("xG not available — skipping xG feature group. Active: BASE ONLY")
            
        if 'corners_scored' in team_df.columns:
            metrics.append('corners')
            logger.info("Feature Group Active: CORNERS (scored/conceded)")
            
        if 'cards_scored' in team_df.columns:
            metrics.append('cards')
            logger.info("Feature Group Active: CARDS (received/drawn)")
            
        for window in [3, 5, 10]:
            team_df = self.engineer.calculate_rolling_stats(team_df, window=window, metrics=metrics)
        
        logger.info("Calculating Form...")
        team_df = self.engineer.calculate_form(team_df)
        
        logger.info("Calculating Rest Days...")
        team_df = self.engineer.calculate_rest_days(team_df)

        logger.info("Calculating Team Availability...")
        team_df = self.engineer.add_team_availability_feature(team_df)
        
        # 2.5 Dynamic League Priors (Bug 3.1 & 3.3 Fix)
        # Strategy: Use Static Priors for Absolute Stability (Bug 3.1 & 3.3 Fix)
        # We avoid calculating means from the entire the current dataset to prevent 
        # look-ahead bias and ensure localized stability.
        logger.info("Imputing missing features with Static Priors...")
        
        rubric_cols = [c for c in team_df.columns if 'rolling' in c or 'form' in c or 'days_rest' in c]
        for col in rubric_cols:
            if team_df[col].isnull().any():
                if 'form' in col:
                    fill_val = self.StaticPriors.FORM_NEUTRAL
                elif 'corners' in col:
                    fill_val = self.StaticPriors.ROLLING_CORNERS
                elif 'cards' in col:
                    fill_val = self.StaticPriors.ROLLING_CARDS
                elif 'xg' in col or 'goals' in col:
                    fill_val = self.StaticPriors.ROLLING_GOALS
                elif 'days_rest' in col:
                    fill_val = self.StaticPriors.REST_DAYS_DEFAULT
                else:
                    fill_val = self.StaticPriors.ROLLING_GOALS
                    
                with pd.option_context("future.no_silent_downcasting", True):
                    team_df[col] = team_df[col].infer_objects(copy=False).fillna(fill_val)
        
        # 3. Re-assemble to Match-Centric view
        # Use explicit whitelist (ISSUE #7)
        include_xg = 'xg_scored' in team_df.columns and 'xg_conceded' in team_df.columns
        include_corners = 'corners_scored' in team_df.columns
        include_cards = 'cards_scored' in team_df.columns
        
        home_cols = self.FeatureColumns.get_home_columns(
            include_xg=include_xg,
            include_corners=include_corners,
            include_cards=include_cards
        )
        
        # Extract Home and Away features separately
        home_feats = team_df[team_df['is_home'] == 1][home_cols].add_prefix('home_')
        away_feats = team_df[team_df['is_home'] == 0][home_cols].add_prefix('away_')
        
        # Fix match_id join keys
        home_feats = home_feats.rename(columns={'home_match_id': 'match_id'})
        away_feats = away_feats.rename(columns={'away_match_id': 'match_id'})
        
        # Validate before merge (ISSUE #6)
        if home_feats['match_id'].duplicated().any():
            dupes = home_feats[home_feats['match_id'].duplicated(keep=False)]
            raise DataValidationError(
                f"Duplicate match_ids in home features: {dupes['match_id'].unique().tolist()[:5]}"
            )

        if away_feats['match_id'].duplicated().any():
            dupes = away_feats[away_feats['match_id'].duplicated(keep=False)]
            raise DataValidationError(
                f"Duplicate match_ids in away features: {dupes['match_id'].unique().tolist()[:5]}"
            )

        final_df = pd.merge(df, home_feats, on='match_id', how='left', validate='1:1')
        final_df = pd.merge(final_df, away_feats, on='match_id', how='left', validate='1:1')
        
        # Validate no matches lost during feature engineering
        original_match_ids = set(df['match_id'])
        final_match_ids = set(final_df['match_id'])
        
        if original_match_ids != final_match_ids:
            lost = original_match_ids - final_match_ids
            gained = final_match_ids - original_match_ids
            
            error_parts = []
            if lost:
                error_parts.append(f"Lost {len(lost)} matches: {list(lost)[:5]}")
            if gained:
                error_parts.append(f"Gained {len(gained)} unexpected matches: {list(gained)[:5]}")
            
            raise DataValidationError(
                f"Match ID integrity violation after feature engineering. " + 
                " | ".join(error_parts)
            )
        
        logger.info(f"[Pipeline] Match ID integrity verified: {len(final_match_ids)} matches preserved")
        
        # 4. Calculate Differentials (Home - Away)
        # Identify features that have both Home and Away versions
        base_features = [c.replace('home_', '') for c in home_feats.columns if c.startswith('home_') and c != 'home_match_id' and c!= 'home_team_id']
        
        final_df = self.engineer.calculate_differentials(final_df, base_features)
        
        # 4.5. Validate Feature Completeness
        expected_groups = ['BASE']
        if include_xg:
            expected_groups.append('XG')
        if include_corners:
            expected_groups.append('CORNERS')
        if include_cards:
            expected_groups.append('CARDS')
        
        self._validate_feature_completeness(final_df, expected_groups)
        
        # 5. EXPANSION: Double Chance Targets
        final_df = self.engineer.create_double_chance_targets(final_df)
        
        # 6. H2H Features for Card Calibration (2026-01-13)
        # Critical for matchups like JUVENTUS vs CREMONESE with unique H2H card patterns
        logger.info("Calculating H2H Features...")
        try:
            final_df = self.engineer.calculate_h2h_features(final_df)
            logger.info(f"H2H features added: h2h_avg_cards, h2h_cards_o25_rate, h2h_match_count")
        except KeyError as e:
            logger.warning(
                "H2H feature missing: %s - predictions will fall back to league priors",
                e,
            )
        except Exception as e:
            logger.warning(f"H2H feature calculation failed (non-fatal): {e}")

        # 7. Referee context (rolling cards per referee over last 10 matches)
        logger.info("Calculating Referee Features...")
        try:
            final_df = self.engineer.add_referee_features(final_df, window=10)
            logger.info("Referee features added: referee_id, referee_card_rate_10")
        except Exception as e:
            logger.warning(f"Referee feature calculation failed (non-fatal): {e}")

        # 8. Weather context (OpenWeatherMap, fail-safe zeros)
        logger.info("Calculating Weather Features...")
        try:
            final_df = self.engineer.add_weather_features(final_df)
            logger.info("Weather features added: precipitation_mm, wind_speed_kmh")
        except Exception as e:
            logger.warning(f"Weather feature calculation failed (non-fatal): {e}")
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = DATA_DIR / f"master_features_{timestamp}.csv"
        final_df.to_csv(output_path, index=False)
        self._rotate_master_feature_exports()
        logger.info(f"Saved master features to {output_path}")
        
        return final_df

    def _rotate_master_feature_exports(self, keep_last: Optional[int] = None) -> None:
        """
        Keep only the latest N master feature exports in DATA_DIR.

        Files matched: master_features_*.csv
        Older files are deleted after a new export is written.
        """
        keep = keep_last if keep_last is not None else self.MASTER_FEATURE_ROTATION_KEEP
        keep = max(1, int(keep))

        exports = sorted(
            DATA_DIR.glob("master_features_*.csv"),
            key=lambda p: (p.stat().st_mtime, p.name),
            reverse=True,
        )
        stale = exports[keep:]
        for old_path in stale:
            try:
                old_path.unlink()
                logger.info("Pruned old master feature export: %s", old_path.name)
            except OSError as exc:
                logger.warning("Failed to prune old master feature export %s: %s", old_path, exc)

    def run(
        self, 
        match_data: MatchData = None,
        league: Optional[str] = None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Execute feature engineering pipeline with intelligent caching.
        
        This is the main entry point for converting raw match data into
        ML-ready features with rolling statistics, form ratings, and 
        differentials.
        
        Args:
            match_data: Optional raw match data. If None, loads from disk.
                Can be DataFrame or list of dicts matching ProcessedMatch schema.
            league: Filter to specific league (PL/PD/SA/BL1/FL1). 
                None = load all leagues.
            force_refresh: Bypass cache and reload from disk. Use when
                underlying data has changed.
        
        Returns:
            DataFrame with ~186 engineered features (varies by data availability):
            - Rolling stats (windows: 3, 5, 10 matches)
            - Form ratings
            - Rest days
            - Home/away differentials
            - Double chance targets
        
        Raises:
            DataValidationError: If match data fails schema validation
            FileNotFoundError: If processed match files missing
        
        Example:
            >>> pipeline = FeaturePipeline()
            >>> features = pipeline.run(league='PL')  # Cached
            >>> features = pipeline.run(league='PL', force_refresh=True)  # Fresh
        
        Cache Policy:
            - TTL: 5 minutes
            - Keyed by: league (or 'global' if None)
            - Bypassed when: match_data provided or force_refresh=True
        
        Performance:
            - Cached: ~50ms
            - Fresh load: 2-5s depending on league size
        """
        if league and league not in self.VALID_LEAGUES:
            raise ValueError(
                f"Invalid league: {league}. Valid options: {self.VALID_LEAGUES}"
            )

        if match_data is not None:
            # Bypass cache for manual data injection
            return self._load_and_process(match_data, league)

        cache_key = league or "global"
        
        # Check cache with file modification awareness
        if not force_refresh:
            if not self._should_invalidate_cache(cache_key):
                logger.info(f"[Pipeline] Cache hit for {cache_key}")
                return self._cache[cache_key]
        
        # Load fresh
        logger.info(f"[Pipeline] Cache miss/expired for {cache_key}. Loading fresh data...")
        df = self._load_and_process(match_data, league)
        
        # Update Cache with LRU eviction
        self._evict_lru_cache()
        self._cache[cache_key] = df
        self._cache_timestamp[cache_key] = datetime.now()
        
        return df

    def run_global(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Explicit entry point for running the pipeline across all leagues.
        """
        return self.run(league=None, force_refresh=force_refresh)

    def transform(self, match_data: MatchData) -> pd.DataFrame:
        """
        Alias for run(match_data=...) to support sklearn-like API.
        """
        return self.run(match_data=match_data)

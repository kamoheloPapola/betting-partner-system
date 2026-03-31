"""
[DEPRECATED] CSV Data Importer.

NOTE: This file is deprecated.
The pipeline now uses `src/data_pipeline/build_matches_dataset.py` directly
to convert raw CSV files into `data/processed/matches.csv` with a strict, odds-free schema.

Imports and normalizes match data from football-data.co.uk CSV files.
Handles team name normalization, match ID generation, and schema
standardization for the feature pipeline.
"""

import pandas as pd
import logging
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from functools import partial

from src.config import DATA_DIR, PROCESSED_DATA_DIR
from src.core.constants import STATUS_FT, SOURCE_CSV
from src.utils.naming import calculate_season, normalize_team_name, generate_match_fingerprint

logger = logging.getLogger(__name__)

class HistoryImporter:
    """
    [DEPRECATED] Use `CanonicalDatasetBuilder` instead.
    Imports historical match data from local CSV files (Football-Data.co.uk format).
    Path: data/historical/{LEAGUE}/*.csv
    """
    
    # Standard Map: CSV Header -> System Column
    COL_MAP = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_score',
        'FTAG': 'away_score',
        'FTR': 'result',    # H, D, A
        # New Stats
        'HC': 'home_corners',
        'AC': 'away_corners',
        'HY': 'home_yellow_cards',
        'AY': 'away_yellow_cards',
        'HR': 'home_red_cards',
        'AR': 'away_red_cards',
        # Shots
        'HS': 'home_shots',
        'AS': 'away_shots',
        'HST': 'home_shots_on_target',
        'AST': 'away_shots_on_target',
    }

    def import_league(self, league_code: str) -> Dict[str, Any]:
        """
        Loads all CSVs for a league, normalizes, and saves to processed matches.
        
        Args:
            league_code: Internal league identifier (e.g., 'PL').
            
        Returns:
            Summary dictionary of the ingestion results.
        """
        source_dir = DATA_DIR / "historical" / league_code
        if not source_dir.exists():
            logger.error(f"Historical directory not found: {source_dir}")
            return {"status": "error", "message": "Source directory missing"}

        csv_files = list(source_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {source_dir}")
            return {"status": "warning", "message": "No CSV files found"}

        processed_dfs: List[pd.DataFrame] = []
        files_processed = 0
        files_skipped = []
        total_matches = 0
        
        for p in csv_files:
            try:
                # Read CSV using unicode_escape to handle special characters in team names 
                # (e.g. accents in Spanish/German teams) frequently found in legacy CSV exports.
                df = pd.read_csv(p, encoding='unicode_escape') 
                
                # Check required BASE cols (Scores); extras are optional but desirable
                base_cols = {'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'}
                if not base_cols.issubset(df.columns):
                    logger.warning(f"Skipping {p.name}: Missing base columns")
                    files_skipped.append(p.name)
                    continue
                
                # Rename available columns
                available_cols = [c for c in self.COL_MAP.keys() if c in df.columns]
                renamed_cols = {c: self.COL_MAP[c] for c in available_cols}
                
                df = df.rename(columns=renamed_cols)
                
                # Verify numeric safety for scores
                numeric_score_cols = ['home_score', 'away_score']
                df[numeric_score_cols] = df[numeric_score_cols].apply(pd.to_numeric, errors='coerce')
                df = df.dropna(subset=numeric_score_cols)
                
                # Copy with new names
                keep_cols = list(renamed_cols.values())
                df = df[keep_cols].copy()
                
                # Fill Missing new stats with 0.0 if they don't exist in older files
                new_stats = [
                    'home_corners', 'away_corners', 
                    'home_yellow_cards', 'away_yellow_cards', 
                    'home_red_cards', 'away_red_cards',
                    'home_shots', 'away_shots',
                    'home_shots_on_target', 'away_shots_on_target',
                ]
                for stat in new_stats:
                    if stat not in df.columns:
                        df[stat] = 0.0 
                
                # Normalize Dates
                df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['date'])
                
                # Calculate season from date and league context
                df['season'] = df['date'].apply(partial(calculate_season, league=league_code))
                
                # Add Metadata
                df['league'] = league_code
                df['status'] = STATUS_FT
                df['source'] = SOURCE_CSV
                
                # Calculate Totals (Hardened Validation)
                card_cols = ['home_yellow_cards', 'home_red_cards', 'away_yellow_cards', 'away_red_cards']
                if all(col in df.columns for col in card_cols):
                    df['home_total_cards'] = df['home_yellow_cards'] + df['home_red_cards']
                    df['away_total_cards'] = df['away_yellow_cards'] + df['away_red_cards']
                    df['match_total_cards'] = df['home_total_cards'] + df['away_total_cards']
                
                if 'home_corners' in df.columns and 'away_corners' in df.columns:
                    df['total_corners'] = df['home_corners'] + df['away_corners']
                
                # Shots totals and derived signals
                if 'home_shots' in df.columns and 'away_shots' in df.columns:
                    df['total_shots'] = df['home_shots'] + df['away_shots']
                    # Shot accuracy (on-target / total, guarded against div-by-zero)
                    df['shot_accuracy_home'] = df['home_shots_on_target'] / df['home_shots'].replace(0, float('nan'))
                    df['shot_accuracy_away'] = df['away_shots_on_target'] / df['away_shots'].replace(0, float('nan'))
                    # Shot pressure (team share of total shots)
                    df['shot_pressure_home'] = df['home_shots'] / df['total_shots'].replace(0, float('nan'))
                    df['shot_pressure_away'] = df['away_shots'] / df['total_shots'].replace(0, float('nan'))

                # Normalize Teams (Optimized based on season variation)
                if df['season'].nunique() == 1:
                    season = df['season'].iloc[0]
                    # Vectorized apply is faster than axis=1 for large files with single season
                    df['home_team'] = df['home_team'].apply(
                        partial(normalize_team_name, league=league_code, season=season)
                    )
                    df['away_team'] = df['away_team'].apply(
                        partial(normalize_team_name, league=league_code, season=season)
                    )
                else:
                    def normalize_row_teams(row: pd.Series) -> pd.Series:
                        row['home_team'] = normalize_team_name(
                            row['home_team'], 
                            league=league_code, 
                            season=row['season']
                        )
                        row['away_team'] = normalize_team_name(
                            row['away_team'], 
                            league=league_code, 
                            season=row['season']
                        )
                        return row
                    df = df.apply(normalize_row_teams, axis=1)
                
                # Create match_id
                def make_id(row: pd.Series) -> str:
                    return generate_match_fingerprint(
                        league_code, 
                        row['date'], 
                        row['home_team'], 
                        row['away_team'],
                        season=row['season']
                    )
                
                df['match_id'] = df.apply(make_id, axis=1)
                
                processed_dfs.append(df)
                files_processed += 1
                total_matches += len(df)
                logger.info(f"Loaded {len(df)} matches from {p.name}")
                
            except Exception as e:
                logger.error(f"Error processing {p.name}: {e}", exc_info=True)
                files_skipped.append(p.name)
                
        if not processed_dfs:
            return {
                "status": "error", 
                "message": "No data processed from CSVs",
                "files_skipped": files_skipped
            }
            
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        combined_df = combined_df.sort_values('date')
        
        seasons_saved = []
        for season, group in combined_df.groupby('season'):
            filename = f"{league_code}_{season}.csv"
            save_path = PROCESSED_DATA_DIR / "matches" / filename
            
            # Ensure directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Atomic Write Pattern: prevent race conditions during concurrent imports
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=save_path.parent, suffix=".tmp") as tmp:
                tmp_path = Path(tmp.name)
            
            try:
                # Write to closed temp file to prevent handle leaks/locks on some OS
                group.to_csv(tmp_path, index=False)
                
                # os.replace is atomic and cross-platform (whether target exists or not)
                os.replace(tmp_path, save_path)
                
                logger.info(f"Saved {len(group)} matches to {filename}")
                seasons_saved.append(int(season))
            except Exception as e:
                logger.error(f"Failed to atomically update {save_path}: {e}")
                if tmp_path.exists():
                    os.unlink(tmp_path)
            
        return {
            "status": "success",
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "total_matches": total_matches,
            "seasons": sorted(seasons_saved)
        }

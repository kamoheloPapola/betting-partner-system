"""
Data CLI Commands.

Commands for data acquisition and ingestion:
- fetch-data: Download historical CSVs from football-data.co.uk
- fetch-upcoming: Get upcoming fixtures via Odds API
- fetch-latest-season: Refresh current season data
- ingest-results: Process raw results through distillery and labeler
"""
import typer
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.cli.base import app
from src.cli.utils import LeagueCode
from src.core.container import ServiceContainer
from src.core.validators import validate_match_dataframe
from src.core.exceptions import PredictionSystemError, DataValidationError
from src.config import PROCESSED_DATA_DIR, DATA_DIR

logger = logging.getLogger(__name__)

def ensure_output_path(path: Path) -> Path:
    """
    Ensure directory exists and path is writable.
    
    Args:
        path: Target file path
        
    Returns:
        Validated path
        
    Raises:
        PredictionSystemError: If directory can't be created or isn't writable
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Test writability of directory
        if not os.access(path.parent, os.W_OK):
            raise PermissionError(f"No write permission for directory: {path.parent}")
            
        # Test writability of file if it exists
        if path.exists() and not os.access(path, os.W_OK):
            raise PermissionError(f"No write permission for file: {path}")
        
        return path
        
    except (PermissionError, OSError) as e:
        raise PredictionSystemError(
            f"Cannot write to path: {path}",
            context={"error": str(e), "path": str(path)}
        )

@app.command()
def fetch_data(
    league: LeagueCode = typer.Option(LeagueCode.PL, help="League Code (e.g. PL)"),
    season: int = typer.Option(2024, help="Season Year (e.g. 2024)")
) -> None:
    """
    Download raw historical CSV data for a specific league/season.
    
    Uses `football-data.co.uk` as the source.
    
    Args:
        league: League Code (PL, PD, SA, etc.)
        season: Start year of season (e.g., 2024 for 24/25)
        
    Raises:
        DataValidationError: If download fails or data is invalid.
    """
    try:
        from src.fetch.football_data import FootballDataFetcher
        
        logger.info("Fetching data", extra={"league": league.value, "season": season})
        fetcher = FootballDataFetcher()
        path = fetcher.fetch_league_season(league.value, season)
        
        if path:
            logger.info("Data fetch successful", extra={"path": path})
        else:
            raise DataValidationError(
                f"Failed to fetch data for {league.value} {season}",
                context={
                    "league": league.value,
                    "season": season,
                    "fix": "Check your internet connection and verify that the league/season combination is supported by football-data.co.uk."
                }
            )
            
    except PredictionSystemError as e:
        logger.error("Data fetch failed", extra={"error": e.message, "context": e.context})
        raise typer.Exit(code=1)



def _process_fixtures_to_dataframe(
    fixtures: List[Dict[str, Any]], 
    league: LeagueCode
) -> pd.DataFrame:
    """
    Convert raw fixture dicts to standardized DataFrame.
    
    Helper function to avoid duplication between fetch_fixtures and fetch_upcoming.
    """
    comp_name = league.full_name
    
    processed_list: List[Dict[str, Any]] = []
    for f in fixtures:
        item = {
            "match_id": f["match_id"],
            "date": f["date_utc"][:10],
            "kickoff_utc": f["date_utc"],
            "home_team": f["home_team"],
            "away_team": f["away_team"],
            "home_team_id": None, 
            "away_team_id": None, 
            "competition": comp_name,
            "league": f["league"],
            "season": f["season"],
            "status": "SCHEDULED",
            "source": "odds_api_cached"
        }
        processed_list.append(item)
    
    df = pd.DataFrame(processed_list)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def _save_upcoming_matches(df: pd.DataFrame, league: LeagueCode) -> None:
    """Save upcoming matches to CSV."""
    output_path = ensure_output_path(
        PROCESSED_DATA_DIR / "matches" / f"{league.value}_upcoming.csv"
    )
    
    if df.empty:
        # This is a safety guard; validation in the caller should catch this first.
        raise DataValidationError("Attempted to save an empty upcoming matches DataFrame")
   
    df.to_csv(output_path, index=False)
    logger.info(
        "Saved upcoming matches", 
        extra={
            "count": len(df),
            "path": str(output_path),
            "league": league.value,
            "date_range": f"{df['date'].min()} to {df['date'].max()}" if not df.empty else None,
            "teams": df[['home_team', 'away_team']].values.tolist()[:3]  # First 3 matches
        }
    )

@app.command()
def fetch_upcoming(
    league: LeagueCode = typer.Option(LeagueCode.PL, help="League Code (e.g. PL)"),
    skip_validation: bool = typer.Option(False, "--skip-validation", help="Skip schema validation (faster, for debugging)"),
    refresh: bool = typer.Option(False, "--refresh", help="Force refresh, bypass cache")
) -> None:
    """
    Fetch upcoming fixtures via The Odds API.
    
    Steps:
    1. Calls FixtureManager (which hits Odds API)
    2. Standardizes team names and formats
    3. Validates schema integrity (unless --skip-validation)
    4. Saves to `data/processed/matches/{league}_upcoming.csv`
    """
    try:
        from src.ingestion.fixture_manager import FixtureManager
        
        logger.info("Fetching upcoming fixtures", extra={"league": league.value, "refresh": refresh})
        
        manager = FixtureManager()
        fixtures = manager.get_fixtures(league.value, force_refresh=refresh)
        
        if not fixtures:
            logger.warning("No upcoming fixtures retrieved for league", extra={"league": league.value})
            return
        
        df = _process_fixtures_to_dataframe(fixtures, league)
        
        # This should NEVER happen if fixtures was not empty
        if df.empty:
            raise DataValidationError(
                "DataFrame is empty after processing fixtures",
                context={"fixture_count": len(fixtures), "league": league.value}
            )
        
        if not skip_validation:
            validate_match_dataframe(df, context="fetch_upcoming")
        
        _save_upcoming_matches(df, league)
        
    except PredictionSystemError as e:
        logger.error("Fetching upcoming fixtures failed", extra={"error": e.message})
        raise typer.Exit(code=1)

@app.command("fetch-latest-season")
def fetch_latest_season(season: str = "2526") -> None:
    """
    Download latest season results from primary source.
    
    Helper command to refresh current season data for analysis/training.
    Files are saved to `data/results/raw/{LEAGUE}/season_{season}.csv`.
    
    Args:
        season: Season suffix (e.g. '2526' for 2025/2026)
    """
    from src.fetch.fetch_latest_season_csvs import SeasonFetcher
    fetcher = SeasonFetcher()
    logger.info("Fetching latest season data", extra={"season": season})
    fetcher.fetch_season(season)
    logger.info("Fetch complete")

@app.command("ingest-results")
def ingest_results(
    league: Optional[LeagueCode] = typer.Option(None, help="League to ingest (e.g. PL). If None, all in data/historical/.")
) -> None:
    """
    Process raw result CSVs into labeling-ready format.
    
    The 'Result Distillery' normalizes data from different sources 
    (historical, daily updates) into a unified result schema.
    Then, the 'Labeler' calculates targets (Win/Loss, BTTS, Corners, etc.)
    for model training.
    
    Args:
        league: Specific league to ingest. None = All leagues.
    """
    from src.data.processors.result_distillery import ResultDistillery
    from src.data.processors.labeler import Labeler
    
    distillery = ResultDistillery()
    labeler = Labeler()
    
    leagues: List[str] = [league.value] if league else ['PL', 'PD', 'SA', 'BL1', 'FL1']
   
    logger.info("Starting Result Distillery", extra={"leagues": leagues})
    
    for lg in leagues:
        # 1. Check Historical
        hist_dir = DATA_DIR / "historical" / lg
        if hist_dir.exists():
            for csv_file in hist_dir.glob("*.csv"):
                distillery.ingest_raw_csv(csv_file, lg)
        
        # 2. Check Raw Snapshots (Latest Season)
        raw_dir = DATA_DIR / "results" / "raw" / lg
        if raw_dir.exists():
            for csv_file in raw_dir.glob("*.csv"):
                distillery.ingest_raw_csv(csv_file, lg)
            
    # Step 2: Labeling
    logger.info("Generating Deterministic Labels")
    labeler.generate_labels()
    logger.info("Ingestion & Labeling complete")


@app.command("build-canonical")
def build_canonical() -> None:
    """
    Build clean canonical matches.csv from raw Football-Data.co.uk CSVs.

    Reads data/historical/{LEAGUE}/*.csv, strips odds, normalises schema.
    Output:
    - data/processed/matches.csv  (canonical dataset — no odds)
    - data/processed/matches_coverage.json  (provenance metadata)
    """
    from src.data_pipeline import CanonicalDatasetBuilder

    builder = CanonicalDatasetBuilder()
    result = builder.build()

    if result["status"] == "success":
        typer.echo(
            f"✅ Canonical dataset built: {result['total_matches']} matches "
            f"({result['files_processed']} files, "
            f"{result['duplicates_removed']} duplicates removed)"
        )
        if result["files_skipped"]:
            typer.echo(f"⚠️  Skipped files: {result['files_skipped']}")
    elif result["status"] == "empty":
        typer.echo("⚠️  No source files found. Run 'ingest-results' first.")
    else:
        typer.echo(f"❌ Build failed: {result.get('message', 'Unknown error')}")
        raise typer.Exit(code=1)


@app.command("build-features")
def build_features() -> None:
    """
    Build the ML-ready feature matrix from canonical/processed match data.

    Runs the 7-layer pipeline:
    1. Raw match stats  2. Team tables  3. Rolling windows
    4. SoS normalisation  5. Dominance metrics
    6. Matchup deltas  7. Context signals

    Output:
    - data/features/feature_matrix.csv
    - data/features/feature_matrix_hash.json
    - data/features/feature_matrix_coverage.json
    """
    from src.features.build_feature_matrix import FeatureMatrixBuilder

    builder = FeatureMatrixBuilder()
    result = builder.build()

    if result["status"] == "success":
        typer.echo(
            f"✅ Feature matrix built: {result['rows']} rows × {result['features']} features\n"
            f"   SHA256: {result['sha256'][:16]}…\n"
            f"   Path:   {result['matrix_path']}"
        )
    else:
        typer.echo(f"❌ Build failed: {result.get('message', 'Unknown error')}")
        raise typer.Exit(code=1)

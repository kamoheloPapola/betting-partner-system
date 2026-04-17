"""
Training CLI Commands.

Provides CLI commands for ML model training and historical data import.
Commands:
- train: Execute the full ML training pipeline
- import_history: Bulk import historical match data from CSVs
"""
import logging
import time
from typing import Optional

import typer
from rich.console import Console

from src.cli.base import app
from src.cli.utils import LeagueCode
from src.core.exceptions import PredictionSystemError, DataValidationError
from src.ml.training.model_configs import ModelType, TrainingMode

# Define public API
__all__ = ["train", "import_history"]

logger = logging.getLogger(__name__)


@app.command(hidden=False)  # Unhidden for visibility
def train(
    model_type: ModelType = typer.Argument(ModelType.POISSON, help="Model type architecture to use"), 
    league: Optional[LeagueCode] = typer.Option(None, "--league", "-l", help="Specific league to train. If None, trains Global + Defaults."),
    mode: TrainingMode = typer.Option(TrainingMode.DEBUG, "--mode", "-m", help="Training mode: 'debug' or 'production'"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Path to specific feature file to train on (skips generation)")
) -> None:
    """
    Train predictive models for specified leagues or global scope.
    
    This command orchestrates the full ML training lifecycle:
    1. Generates/Loads feature matrix
    2. Filters for valid historical data (Finished matches only)
    3. Executes training loop for target models
    4. Persists trained models to the ModelRegistry
    """
    console = Console()
    console.rule(f"[bold blue]Training Pipeline: {model_type.value.upper()}[/bold blue]")
    
    start_time = time.time()
    
    try:
        from src.ml.training.data_validator import filter_historical_matches, load_feature_file
        from src.ml.training.orchestrator import TrainingOrchestrator

        # LOCK ENFORCEMENT: Fail hard if system is locked
        from src.config.model_state import require_unlocked
        require_unlocked("Training")
        
        orchestrator = TrainingOrchestrator()
        logger.info(f"Starting Training Pipeline [mode={mode}, type={model_type.value}]")
        
        # 1. Acquire Data
        with console.status("[bold green]Acquiring training data...[/bold green]"):
            if file:
                console.print(f"Loading features from: [cyan]{file}[/cyan]")
                df = load_feature_file(file)
            else:
                console.print("Generating fresh features (Global)...")
                df = orchestrator.container.pipeline.run(league=None) 
        
        if df is None or df.empty:
            raise DataValidationError("No training data available after feature generation.")
            
        # 2. Safety Filters (Historical Only, Non-Mutating)
        with console.status("[bold green]Validating data integrity...[/bold green]"):
            df = filter_historical_matches(df)
        
        if df.empty:
            raise DataValidationError(
                "No historical matches available for training after filtering",
                context={"reason": "empty_after_filter"}
            )
        
        console.print(f"Training on [bold cyan]{len(df)}[/bold cyan] historical matches.")
            
        # 3. Training Loop
        l_code = league.value if league else None
        
        with console.status(f"[bold green]Training models ({'Global' if not l_code else l_code})...[/bold green]"):
            # Orchestrator run currently returns None, but wrapped for future metrics extensibility
            orchestrator.run(df, model_type, league_code=l_code, mode=mode)
            
        duration = time.time() - start_time
        logger.info("Batch Training Complete")
        console.print(f"\n[bold green]✓ Training Complete![/bold green] (Duration: {duration:.1f}s)")
        if l_code:
            console.print(f"Models updated for [cyan]{l_code}[/cyan]")
        else:
            console.print("Models updated for [cyan]ALL LEAGUES[/cyan]")

    except (PredictionSystemError, DataValidationError) as e:
        logger.error(f"Training failed: {e.message}", extra={"context": getattr(e, "context", {})})
        console.print(f"\n[bold red]Training Failed:[/bold red] {e.message}")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        console.print(f"\n[bold red]Error:[/bold red] File not found: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Unexpected error during training")
        console.print(f"\n[bold red]Unexpected Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def import_history(
    league: Optional[LeagueCode] = typer.Option(None, "--league", "-l", help="Limit import to specific league only")
) -> None:
    """
    Bulk import historical match data from local raw CSVs.
    
    Processing path: data/historical/{LEAGUE}/*.csv -> data/processed/matches/*.csv
    """
    console = Console()
    console.rule("[bold blue]Historical Data Import[/bold blue]")
    
    try:
        # Correct import path based on file system discovery
        from src.ingestion.csv_importer import HistoryImporter
        from src.config.leagues import ACTIVE_LEAGUES
        
        importer = HistoryImporter()
        
        leagues_to_process = [league] if league else [LeagueCode(l) for l in ACTIVE_LEAGUES]
        
        with console.status(f"[bold green]Processing {len(leagues_to_process)} leagues...[/bold green]"):
            for lg in leagues_to_process:
                console.print(f"Processing [cyan]{lg.value}[/cyan]...")
                result = importer.import_league(lg.value)
                
                if result['status'] == 'success':
                    console.print(f"  ✓ Processed {result['files_processed']} files ({result['total_matches']} matches)")
                elif result['status'] == 'warning':
                    console.print(f"  ! {result['message']}", style="yellow")
                else:
                    console.print(f"  ✗ {result['message']}", style="red")
                 
        logger.info("Historical import complete")
        console.print("\n[bold green]✓ Import Workflow Complete![/bold green]")
        
    except PredictionSystemError as e:
        logger.error(f"Historical import failed: {e.message}", extra={"context": e.context})
        console.print(f"\n[bold red]Import Failed:[/bold red] {e.message}")
        raise typer.Exit(code=1)
    except ImportError as e:
        logger.error(f"Module import failed: {e}")
        console.print(f"\n[bold red]System Error:[/bold red] Could not load importer module. {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.exception("Unexpected error during import")
        console.print(f"\n[bold red]Unexpected Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


"""
Evaluation CLI Commands.

Commands for model evaluation, calibration analysis, and prediction reconciliation.
Includes health checks, accuracy metrics, and Wilson confidence intervals.
"""
import typer
import pandas as pd
import logging
import os
import numpy as np
from scipy import stats
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.ml.registry import ModelRegistry
from src.cli.base import app
from src.cli.utils import LeagueCode, resolve_league_code
from src.core.container import ServiceContainer
from src.core.validators import validate_match_dataframe
from src.core.exceptions import PredictionSystemError, DataValidationError, InsufficientDataError
from src.config import PROCESSED_DATA_DIR, DATA_DIR
from src.config.thresholds import Thresholds

logger = logging.getLogger(__name__)

def _calculate_wilson_ci(accuracy: float, n: int, confidence: float = 0.95) -> tuple:
    """Calculate Wilson score confidence interval."""
    if n == 0:
        return (0.0, 0.0)
    
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / n
    center = (accuracy + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt(
        accuracy * (1 - accuracy) / n + z**2 / (4 * n**2)
    ) / denominator
    
    return (center - margin, center + margin)

def _calculate_baseline_accuracy(results: pd.DataFrame) -> float:
    """
    Calculate accuracy of naive 'most common outcome' baseline.
    """
    if results.empty:
        return 0.0
    most_common = results['actual'].mode()[0]
    baseline_pred = [most_common] * len(results)
    baseline_acc = (baseline_pred == results['actual']).mean()
    return baseline_acc

def _validate_date_format(date_str: Optional[str], format: str = "%Y%m%d") -> str:
    """
    Validate and normalize date string.
    """
    if date_str is None:
        return datetime.now().strftime(format)
    
    try:
        datetime.strptime(date_str, format)
        return date_str
    except ValueError:
        raise ValueError(
            f"Invalid date format: '{date_str}'. Expected format: {format} (e.g., 20250107)"
        )

@dataclass
class ModelHealthStatus:
    """Health check result for a league's models."""
    league: str
    goals_status: str
    corners_status: str
    cards_status: str
    flags: List[str]
    
    @property
    def health(self) -> str:
        """Overall health classification."""
        global_count = len([f for f in self.flags if "Global" in f])
        if global_count >= 2:
            return "[bold red]FAIL[/bold red]"
        elif any("Global" in f for f in self.flags):
            return "[yellow]PARTIAL[/yellow]"
        return "[green]HEALTHY[/green]"

def _check_model_health(
    registry: ModelRegistry, 
    league: str, 
    thresholds: Dict[str, int]
) -> ModelHealthStatus:
    """
    Perform health check for a single league.
    """
    flags = []
    
    # Goals check
    m_hg = registry.get_production_model_for_league(league, "poisson_home_base")
    if not m_hg or m_hg.get('league') in ['Global', None]:
        goals_status = "[yellow]GLOBAL[/yellow]"
        flags.append("Global-Goals")
    elif (m_hg.get('train_size', 0) + m_hg.get('test_size', 0)) < thresholds['goals']:
        goals_status = "[orange3]SMALL[/orange3]"
        flags.append("Low-Sample-Goals")
    else:
        goals_status = "OK"
    
    # Corners check
    m_hc = registry.get_production_model_for_league(league, "nb_home_corners_base")
    if not m_hc or m_hc.get('league') in ['Global', None]:
        corners_status = "[yellow]GLOBAL[/yellow]"
        flags.append("Global-Corners")
    elif (m_hc.get('train_size', 0) + m_hc.get('test_size', 0)) < thresholds['corners']:
        corners_status = "[orange3]SMALL[/orange3]"
        flags.append("Low-Sample-Corners")
    else:
        corners_status = "OK"
    
    # Cards check
    m_tc = registry.get_production_model_for_league(league, "poisson_total_cards_base")
    cards_status = "[green]OK[/green]" if m_tc and m_tc.get('league') == league else "[dim]Global[/dim]"
    
    # Mixed state detection
    if "Global-Goals" in flags and "Global-Corners" not in flags:
        flags.append("Mixed-Alignment")
    
    return ModelHealthStatus(
        league=league,
        goals_status=goals_status,
        corners_status=corners_status,
        cards_status=cards_status,
        flags=flags
    )

@app.command("resolve-predictions")
def resolve_predictions() -> None:
    """
    Map predictions to actual labeled results (Won/Lost/Void).
    
    This authoritative command updates the prediction outcomes by matching
    pending predictions against finalized match results.
    
    Output:
        - Updates `data/eval/prediction_outcomes.csv`
        - Logs resolution statistics
    """
    try:
        from src.evaluation.resolve_results import AuthoritativeResolver
        logger.info("Resolving predictions into authoritative outcomes")
        resolver = AuthoritativeResolver()
        resolver.resolve_all()
        logger.info("Resolution complete")
    except PredictionSystemError as e:
        logger.error("Resolution failed", extra={"error": e.message, "context": e.context})
        raise typer.Exit(code=1)

@app.command("audit-coverage")
def audit_coverage(
    save: bool = typer.Option(False, "--save", "-s", help="Save the audit matrix to CSV.")
) -> None:
    """
    Perform structural integrity audit of trained models.
    
    Detects critical issues:
    - Silent Global Fallbacks (using generic models instead of league-specific)
    - Training Gaps (insufficient sample size)
    - Mixed Model States (e.g., Goals using Global but Corners using Local)
    
    Args:
        save: If True, saves detailed audit matrix to `data/audit/coverage_audit_{DATE}.csv`
    """
    try:
        console = Console()
        container = ServiceContainer.get_instance()
        registry = container.registry
        
        # Discover leagues
        match_dir = PROCESSED_DATA_DIR / "matches"
        if not match_dir.exists():
            raise DataValidationError(f"Match directory missing: {match_dir}")
            
        leagues = sorted(list(set([
            f.split('_')[0] 
            for f in os.listdir(match_dir) 
            if f.endswith('.csv') and '_' in f
        ])))
        
        # Run health checks
        thresholds = {
            "goals": Thresholds.PROMOTION_GOALS, 
            "corners": Thresholds.PROMOTION_CORNERS
        }
        
        health_results = [
            _check_model_health(registry, league, thresholds)
            for league in leagues
        ]
        
        # Build table
        table = Table(title="[bold cyan]League Model Coverage Audit[/bold cyan]")
        table.add_column("League", style="bold")
        table.add_column("Goals (H/A)", justify="center")
        table.add_column("Corners (H/A)", justify="center")
        table.add_column("Cards", justify="center")
        table.add_column("Health", justify="center")
        table.add_column("Red Flags")
        
        for status in health_results:
            table.add_row(
                status.league,
                status.goals_status,
                status.corners_status,
                status.cards_status,
                status.health,
                ", ".join(status.flags) if status.flags else "-"
            )
        
        console.print(table)
        console.print(f"[dim]Thresholds: Goals >= {thresholds['goals']} | Corners >= {thresholds['corners']}[/dim]")
        
        # Save if requested
        if save:
            df = pd.DataFrame([
                {
                    "league": s.league,
                    "health": s.health,
                    "flags": "|".join(s.flags)
                }
                for s in health_results
            ])
            path = DATA_DIR / "audit" / f"coverage_audit_{datetime.now().strftime('%Y%m%d')}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            logger.info("Audit matrix saved", extra={"path": str(path)})
    
    except PredictionSystemError as e:
        logger.error("Coverage audit failed", extra={"error": e.message, "context": e.context})
        raise typer.Exit(code=1)

@app.command("reconcile", hidden=True)
def reconcile(
    date_str: str = typer.Option(None, help="Prediction date to reconcile (YYYY-MM-DD). Defaults to yesterday.")
) -> None:
    """
    Reconcile past predictions with actual outcomes (Hash-Match).
    
    Validates prediction accuracy by comparing stored prediction hashes
    against fulfilled results. Generates detailed evaluation records.
    
    Args:
        date_str: Target date (YYYY-MM-DD). Defaults to yesterday.
    """
    try:
        from src.monitoring.reconciler import Reconciler
        
        target_date = date_str or (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        logger.info("Reconciling predictions", extra={"date": target_date})
        
        reconciler = Reconciler()
        reconciler.reconcile_date(target_date)
        logger.info("Reconciliation complete")
    except PredictionSystemError as e:
        logger.error("Reconciliation failed", extra={"error": e.message, "context": e.context})
        raise typer.Exit(code=1)

@app.command(hidden=True)
def backtest(
    test_season: int = typer.Option(..., help="Season to test (e.g. 2023)"),
    league: LeagueCode = typer.Option(LeagueCode.PL, help="League Code")
) -> None:
    """
    Run Walk-Forward Validation (Backtest) for a full season.
    
    Simulates the prediction pipeline week-by-week for the target season,
    retraining models on all prior data before each round.
    
    Args:
        test_season: Season year to test (e.g. 2023 for 23/24)
        league: League code to validate
        
    Side Effects:
        Saves results to `data/processed/backtest/results_{league}_{season}.csv`
        Logs accuracy and Brier score metrics to console and log file.
        
    Raises:
        InsufficientDataError: If no prior seasons available for training
        PredictionSystemError: If backtest execution fails
    """
    try:
        from src.backtest.engine import Backtester
        
        processed_dir = PROCESSED_DATA_DIR / "matches"
        available_files = list(processed_dir.glob(f"{league.value}_*.csv"))
        
        all_seasons = []
        for f in available_files:
            try:
                s = int(f.stem.split('_')[1])
                all_seasons.append(s)
            except:
                pass
        all_seasons.sort()
        train_seasons = [s for s in all_seasons if s < test_season]
        
        if not train_seasons:
            raise InsufficientDataError(
                f"No prior seasons found to train for testing {test_season}.",
                context={"available_seasons": all_seasons, "league": league.value}
            )
            
        logger.info("Starting backtesting", extra={"league": league.value, "test_season": test_season, "train_seasons": train_seasons})
        
        tester = Backtester()
        
        # Wrap with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=Console()
        ) as progress:
            task = progress.add_task(
                f"[cyan]Initializing backtest for {league.value} {test_season}...[/cyan]", 
                total=None
            )
            results = tester.run(train_seasons, test_season, league=league.value)
            progress.update(task, completed=True)
            
        if results.empty:
            logger.warning("Backtest produced no results")
            return
            
        total = len(results)
        acc = (results['pred'] == results['actual']).mean()
        brier = results['brier_home'].mean()
        
        # Baseline Comparison (ISSUE #8)
        baseline_acc = _calculate_baseline_accuracy(results)
        improvement = acc - baseline_acc
        
        # Calculate Wilson Score Confidence Interval
        ci_lower, ci_upper = _calculate_wilson_ci(acc, total)
        
        logger.info("Backtest Results", extra={
            "matches_tested": total,
            "accuracy_winner": f"{acc:.2%}",
            "baseline_accuracy": f"{baseline_acc:.2%}",
            "improvement": f"{improvement:+.1%}",
            "relative_improvement": f"{(improvement / baseline_acc if baseline_acc > 0 else 0):.1%}",
            "ci_95_lower": f"{ci_lower:.2%}",
            "ci_95_upper": f"{ci_upper:.2%}",
            "brier_score_home": f"{brier:.4f}"
        })
        
        if improvement < 0.02:
            logger.warning(
                "Model barely beats baseline. Review model quality.",
                extra={"improvement": f"{improvement:+.1%}"}
            )
        
        # UI Display
        console = Console()
        console.print(f"\n[bold]Backtest Summary[/bold]")
        console.print(f"  Accuracy:    [green]{acc:.1%}[/green] (95% CI: [{ci_lower:.1%}, {ci_upper:.1%}])")
        console.print(f"  Baseline:    {baseline_acc:.1%} (Most Common Outcome)")
        console.print(f"  Improvement: [bold]{improvement:+.1%}[/bold]")
        console.print(f"  Brier Score: [cyan]{brier:.4f}[/cyan]")
        console.print(f"  Sample Size: {total} matches\n")
       
        res_path = PROCESSED_DATA_DIR / "backtest" / f"results_{league.value}_{test_season}.csv"
        res_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(res_path, index=False)
        logger.info("Backtest results saved", extra={"path": res_path})

    except PredictionSystemError as e:
        logger.error("Backtesting failed", extra={"error": e.message, "context": e.context})
        raise typer.Exit(code=1)

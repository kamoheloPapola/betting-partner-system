"""
Maintenance CLI Commands.

System maintenance and monitoring commands:
- inspect-drift: Read the persisted drift guard state without mutating it
- refresh-drift: Calculate rolling drift metrics and update Drift Guardrail
- health-check: Verify system integrity and data freshness
- reconcile: Compare predictions with actual outcomes for evaluation
"""
import typer
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
from typing import TypedDict, Optional

from src.cli.base import app
from src.core.validators import validate_match_dataframe
from src.core.exceptions import PredictionSystemError, DataValidationError
from src.config import PROCESSED_DATA_DIR, Thresholds

logger = logging.getLogger(__name__)

LEAGUE_DRIFT_MIN_SAMPLES = 200


class DriftMetrics(TypedDict):
    """Schema for statistical drift analysis results."""
    goals_shift: float
    corners_shift: float
    hist_goals_avg: float
    recent_goals_avg: float
    sample_size: int


def _calculate_binned_ece(
    probabilities: pd.Series,
    outcomes: pd.Series,
    n_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE) with equal-width bins.
    """
    probs = pd.to_numeric(probabilities, errors="coerce").to_numpy(dtype=float)
    hits = pd.to_numeric(outcomes, errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(probs) & np.isfinite(hits)
    probs = probs[mask]
    hits = hits[mask]
    if probs.size == 0:
        return 0.0

    probs = np.clip(probs, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bin_edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    total = float(probs.size)
    for idx in range(n_bins):
        in_bin = bin_ids == idx
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_conf = float(probs[in_bin].mean())
        bin_acc = float(hits[in_bin].mean())
        ece += (count / total) * abs(bin_conf - bin_acc)

    return float(ece)


def _default_season_token(now: Optional[datetime] = None) -> str:
    """Infer current season token in YYZZ format."""
    utc_now = now or datetime.now(timezone.utc)
    season_start_year = utc_now.year if utc_now.month >= 7 else utc_now.year - 1
    next_year = season_start_year + 1
    return f"{season_start_year % 100:02d}{next_year % 100:02d}"


def _send_pipeline_failure_alert(
    *,
    failed_step: str,
    league: str,
    season: str,
    lookback: int,
    prediction_date: str,
    reconcile_date: Optional[str],
    error: Exception,
    force: bool = False,
) -> bool:
    """Dispatch a CRITICAL alert when orchestration fails."""
    from src.monitoring.alerter import Alerter

    message = f"Daily pipeline failed at step: {failed_step}"
    context = {
        "league": league,
        "season": season,
        "lookback_days": lookback,
        "prediction_date_filter": prediction_date,
        "reconcile_date": reconcile_date or "yesterday(default)",
        "error_type": type(error).__name__,
        "error": str(error) or repr(error),
    }
    try:
        return Alerter().send_alert(
            message=message,
            context=context,
            severity="CRITICAL",
            force=force,
        )
    except Exception:
        logger.exception("Failed to dispatch pipeline failure alert")
        return False

@app.command("inspect-drift")
def inspect_drift() -> None:
    """
    Show the persisted drift state and guard thresholds without mutating state.
    """
    from rich.console import Console
    from rich.table import Table
    from src.strategies.drift_guard import DriftGuardrail

    console = Console()
    guard = DriftGuardrail()
    state = guard.inspect_state()
    baseline_info = guard.describe_baselines()

    console.print(f"[bold]Status file:[/bold] {state['path']}")
    console.print(f"[bold]Baseline file:[/bold] {baseline_info['path']}")
    console.print(f"[bold]Baseline source:[/bold] {baseline_info.get('source', 'unknown')}")
    console.print(
        f"[bold]Baseline training date:[/bold] {baseline_info.get('training_date') or 'unavailable'}"
    )

    thresholds = baseline_info["thresholds"]
    baselines = baseline_info["values"]
    current_metrics = state.get("metrics", {})
    stop_boundaries = {
        "hit_rate": baselines["hit_rate"] + thresholds["hit_rate_stop"],
        "ece": baselines["ece"] + thresholds["ece_stop"],
        "mean_conf": baselines["mean_conf"] + thresholds["conf_inflation_stop"],
    }

    metrics_table = Table(title="[bold cyan]Drift Guard Thresholds[/bold cyan]")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Current", justify="right")
    metrics_table.add_column("Baseline", justify="right")
    metrics_table.add_column("Stop Boundary", justify="right")

    for metric in ("hit_rate", "ece", "mean_conf"):
        current = current_metrics.get(metric)
        current_display = f"{float(current):.3f}" if current is not None else "-"
        metrics_table.add_row(
            metric,
            current_display,
            f"{baselines[metric]:.3f}",
            f"{stop_boundaries[metric]:.3f}",
        )

    console.print(metrics_table)

    if not state["exists"]:
        console.print("[yellow]No persisted drift state found.[/yellow]")
        return

    status_style = {
        "OK": "green",
        "WATCH": "yellow",
        "STOP": "bold red",
    }.get(state["status"], "white")
    console.print(f"[bold]Status:[/bold] [{status_style}]{state['status']}[/{status_style}]")

    if state.get("evaluated_at"):
        console.print(f"[bold]Evaluated at:[/bold] {state['evaluated_at']}")
    elif state.get("legacy_date"):
        console.print(f"[bold]Legacy date field:[/bold] {state['legacy_date']}")

    if state.get("note"):
        console.print(f"[yellow]{state['note']}[/yellow]")

    alerts = state.get("alerts", [])
    if alerts:
        console.print("[bold]Alerts:[/bold]")
        for alert in alerts:
            console.print(f" - {alert}")
    else:
        console.print("[green]No alerts recorded.[/green]")

@app.command("refresh-drift")
def refresh_drift(
    window: int = typer.Option(30, help="Rolling window days", min=7, max=365)
) -> None:
    """
    Update global statistical drift metrics.
    
    Calculates rolling average shifts in key indicators to detect market regime changes.
    """
    try:
        df = _load_features()
        df = _prepare_drift_data(df)
        metrics = _analyze_drift(df, window)
        _report_drift_results(metrics)
            
    except PredictionSystemError as e:
        logger.error("Drift refresh failed", extra={"error": e.message, "context": e.context}, exc_info=True)
        raise typer.Exit(code=1) from e
    except Exception as e:
        logger.error("Unexpected drift error", exc_info=True)
        raise typer.Exit(code=1) from e

def _load_features() -> pd.DataFrame:
    """Load features from disk or generate if missing."""
    features_path = PROCESSED_DATA_DIR / "features" / "master_features.csv"
    
    if features_path.exists():
        logger.info("Loading existing features", extra={"path": str(features_path)})
        return pd.read_csv(features_path, parse_dates=['date'])
    
    logger.warning("Features not found, generating from scratch...")
    from src.features.pipeline import FeaturePipeline
    return FeaturePipeline().run_global()

def _prepare_drift_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate data for drift analysis.
    
    Filters for settled matches with complete corner data.
    """
    if df.empty:
        raise DataValidationError(
            "No feature data available for drift analysis",
            context={"expected_path": str(PROCESSED_DATA_DIR / "features" / "master_features.csv")}
        )
    
    # Pre-filter: Remove matches without final scores (Issue #12)
    # Validator expects only settled matches, so we filter first to avoid false validation errors
    if 'home_score' in df.columns and 'away_score' in df.columns:
        pre_filter_size = len(df)
        df = df.dropna(subset=['home_score', 'away_score']).copy()
        filtered_count = pre_filter_size - len(df)
        if filtered_count > 0:
            logger.info(
                "Pre-filtered unsettled matches",
                extra={"removed": filtered_count, "remaining": len(df)}
            )
    
    # Validation layer
    validate_match_dataframe(df, require_settled=True, context="refresh_drift")
    
    # Check for required columns
    required_cols = ['home_corners', 'away_corners', 'date']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Missing required columns for drift analysis: {missing}",
            context={"available_columns": list(df.columns)}
        )
    
    # Remove rows with missing corner data
    # Phase 10: Settled Outcomes Only
    # Ensure we only learn from finished matches
    from src.config.leagues import FINISHED_STATUSES
    if 'status' in df.columns:
        df = df[df['status'].isin(FINISHED_STATUSES)].copy()
        
    initial_size = len(df)
    df = df.dropna(subset=['home_corners', 'away_corners']).copy()
    dropped_count = initial_size - len(df)
    
    if dropped_count > 0:
        logger.warning(
            "Dropped rows with missing corner data",
            extra={"dropped": dropped_count, "remaining": len(df)}
        )
    
    if df.empty:
        raise DataValidationError(
            "No valid data remaining after filtering",
            context={"reason": "All rows had missing corner data"}
        )
    
    return df.sort_values('date').copy()

def _analyze_drift(df: pd.DataFrame, window_days: int) -> DriftMetrics:
    """Perform statistical drift detection on match data."""
    from src.strategies.drift_guard import DriftGuardrail
    return DriftGuardrail().detect_drift(df, window_days=window_days)

def _report_drift_results(metrics: DriftMetrics) -> None:
    """Log and alert based on drift metrics."""
    logger.info("Drift analysis complete", extra={"metrics": metrics})
    
    goals_shift = metrics['goals_shift']
    if goals_shift > Thresholds.DRIFT_WARNING_THRESHOLD:
        logger.warning(
            "Significant goal drift detected!", 
            extra={
                "shift": f"{goals_shift:.1%}", 
                "threshold": f"{Thresholds.DRIFT_WARNING_THRESHOLD:.1%}",
                "avg_hist": f"{metrics['hist_goals_avg']:.2f}",
                "avg_recent": f"{metrics['recent_goals_avg']:.2f}",
                "recommendation": "Review model performance and consider retraining"
            }
        )


@app.command("check-drift")
def check_drift(
    league: str = typer.Option(None, "--league", "-l", help="Specific league to check (default: all)"),
    lookback: int = typer.Option(120, help="Days to look back", min=7, max=120)
) -> None:
    """
    Run drift detection with type classification.
    
    Detects:
    - CALIBRATION_DRIFT: ECE increased beyond threshold
    - SHARPNESS_DECAY: Predictions becoming less confident  
    - MARKET_DEGRADATION: Per-market performance issues
    
    Alerts are logged to data/monitoring/drift_alerts.csv
    """
    from rich.console import Console
    from rich.table import Table
    
    try:
        from src.monitoring.drift_orchestrator import DriftOrchestrator
        from src.evaluation.resolve_results import AuthoritativeResolver
        
        console = Console()
        use_persisted_state = False
        
        # Get resolved predictions
        resolver = AuthoritativeResolver()
        predictions = resolver.load_outcomes()
        
        if predictions.empty:
            logger.warning("No resolved predictions found for drift analysis")
            console.print("[yellow]No resolved predictions available. Run 'resolve-predictions' first.[/yellow]")
            return

        if "kickoff_date" in predictions.columns:
            predictions = predictions.copy()
            predictions["kickoff_date"] = pd.to_datetime(
                predictions["kickoff_date"], utc=True, errors="coerce"
            )
            predictions = predictions.dropna(subset=["kickoff_date"])

            # Use a calendar-day cutoff so "last N days" includes the full boundary day.
            cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback)).normalize()
            filtered = predictions[predictions["kickoff_date"] >= cutoff]
            predictions = filtered
            if predictions.empty:
                console.print(
                    f"[yellow]No resolved predictions in the last {lookback} days — drift status read from persisted state. Run 'resolve-predictions' to populate recent outcomes.[/yellow]"
                )
                use_persisted_state = True
        
        # Filter by league if specified
        if league and not use_persisted_state:
            predictions = predictions[predictions['league'] == league]
            if predictions.empty:
                console.print(f"[yellow]No predictions found for league {league}[/yellow]")
                return

        if not use_persisted_state:
            console.print(
                f"[dim]Using {len(predictions)} resolved predictions from the last {lookback} days.[/dim]"
            )
        
        # Run drift checks via unified orchestrator
        monitor = DriftOrchestrator()
        alerts: list[str] = []
        metrics_for_display: dict[str, float] = {}
        status = DriftOrchestrator.GO

        def _evaluate_current_state() -> str:
            if league:
                return monitor.evaluate_league_drift(league)
            return monitor.evaluate_global_drift()

        def _evaluate_metrics(metrics: dict[str, float]) -> str:
            if league:
                league_metrics = dict(metrics)
                evaluated_status = monitor.evaluate_league_drift(league, league_metrics)
                monitor.persist_league_state(league)
                return evaluated_status
            return monitor.evaluate_global_drift(metrics)

        if use_persisted_state:
            status = _evaluate_current_state()
        elif {"probability", "outcome"}.issubset(predictions.columns):
            scored = predictions[["probability", "outcome"]].copy()
            scored["probability"] = pd.to_numeric(scored["probability"], errors="coerce")
            scored = scored.dropna(subset=["probability", "outcome"])

            if not scored.empty:
                # Outcomes may be stored as numeric 0/1 or strings WON/LOST
                outcome_col = scored["outcome"]
                if pd.api.types.is_numeric_dtype(outcome_col):
                    scored["hit"] = pd.to_numeric(outcome_col, errors="coerce")
                    scored.loc[~scored["hit"].isin([0.0, 1.0]), "hit"] = float("nan")
                else:
                    outcome_map = {"WON": 1.0, "LOST": 0.0, "PUSH": 0.5, "VOID": float("nan")}
                    scored["hit"] = outcome_col.astype(str).str.upper().map(outcome_map)
                scored = scored.dropna(subset=["hit"])

                # For league-specific drift, require minimum sample size for reliable ECE.
                # Below threshold, inherit global drift state rather than compute noisy ECE.
                if league and len(scored) < LEAGUE_DRIFT_MIN_SAMPLES:
                    console.print(
                        f"[yellow]Insufficient sample for league ECE: {len(scored)} predictions "
                        f"(minimum {LEAGUE_DRIFT_MIN_SAMPLES}). Inheriting global drift state.[/yellow]"
                    )
                    status = monitor.evaluate_global_drift()
                    monitor._league_status[league] = status
                    monitor._league_metrics[league] = {
                        "sample_size": len(scored),
                        "minimum_sample_size": LEAGUE_DRIFT_MIN_SAMPLES,
                        "inherited_from": "global",
                        "global_status": status,
                    }
                    monitor.persist_league_state(league)
                    metrics_for_display = {}
                    alerts = []
                elif not scored.empty:
                    probs = scored["probability"].clip(lower=0.0, upper=1.0)
                    hits = scored["hit"]
                    metrics = {
                        "hit_rate": float(hits.mean()),
                        "ece": _calculate_binned_ece(probs, hits),
                        "mean_conf": float(probs.mean()),
                    }
                    metrics_for_display = dict(metrics)
                    status = _evaluate_metrics(metrics)
                    if league:
                        alerts = monitor.get_league_alerts(league)
                    else:
                        alerts = list(monitor.global_alerts)
                else:
                    status = _evaluate_current_state()
            else:
                status = _evaluate_current_state()
        else:
            status = _evaluate_current_state()

        status_scope = "LEAGUE" if league else "GLOBAL"
        if status == DriftOrchestrator.STOP and not alerts:
            alerts = [f"{status_scope}_STOP: DriftOrchestrator returned STOP state"]
        elif status == DriftOrchestrator.WATCH and not alerts:
            alerts = [f"{status_scope}_WATCH: DriftOrchestrator returned WATCH state"]

        if alerts:
            monitor.append_drift_alerts(alerts, league=league, status=status)

        # Display results
        if not alerts:
            console.print("[green]✓ No drift detected[/green]")
            if metrics_for_display:
                table = Table(title="[bold green]Drift Metrics[/bold green]")
                table.add_column("Metric", style="cyan")
                table.add_column("Value")
                table.add_row("Status", status)
                table.add_row("Hit Rate", f"{metrics_for_display['hit_rate']:.3f}")
                table.add_row("ECE", f"{metrics_for_display['ece']:.3f}")
                table.add_row("Mean Confidence", f"{metrics_for_display['mean_conf']:.3f}")
                console.print(table)
            return

        table = Table(title="[bold red]Drift Alerts Detected[/bold red]")
        table.add_column("Type", style="cyan")
        table.add_column("Details")
        
        for alert in alerts:
            alert_type, _, details = str(alert).partition(":")
            table.add_row(alert_type.strip(), (details or str(alert)).strip())
        
        console.print(table)
        
    except PredictionSystemError as e:
        logger.error("Drift check failed", extra={"error": e.message, "context": e.context})
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error("Unexpected drift check failure", exc_info=True, extra={"error": str(e)})
        raise typer.Exit(code=1) from e


@app.command("refresh-offsets")
def refresh_offsets(
    league: str = typer.Option(None, "--league", "-l", help="Specific league (default: all)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force recompute even if frozen")
) -> None:
    """
    Recompute team-level corner/card offsets.
    
    Uses Bayesian shrinkage toward league mean. Only updates if:
    - Pre-season (offsets unfrozen), OR
    - Team has ≥50 new matches since last update
    
    Offsets are saved to data/models/team_offsets.csv
    """
    from rich.console import Console
    
    try:
        # LOCK ENFORCEMENT: Fail hard if system is locked
        from src.config.model_state import require_unlocked
        require_unlocked("Offset refresh")
        
        from src.ml.models.corners.team_offsets import TeamOffsetManager
        from src.config import DEFAULT_TRAINING_LEAGUES
        
        console = Console()
        manager = TeamOffsetManager()
        
        leagues = [league] if league else DEFAULT_TRAINING_LEAGUES
        total_valid = 0
        
        for lg in leagues:
            console.print(f"[cyan]Computing offsets for {lg}...[/cyan]")
            valid = manager.compute_offsets(lg, force=force)
            total_valid += valid
            console.print(f"  → {valid} teams with valid offsets")
        
        console.print(f"\n[green]✓ Total: {total_valid} teams with valid offsets[/green]")
        
    except Exception as e:
        logger.error("Offset refresh failed", exc_info=True)
        raise typer.Exit(code=1)


@app.command("freeze-models")
def freeze_models(
    confirm: bool = typer.Option(False, "--confirm", "-y", help="Confirm freeze action")
) -> None:
    """
    Lock model pipeline state.
    
    After freezing:
    - MODEL_STATE is set to LOCKED_vX.X
    - Any training, tuning, or offset refresh will FAIL HARD
    - Team offsets are frozen mid-season
    
    This prevents "just one tweak" syndrome.
    """
    from rich.console import Console
    
    console = Console()
    
    if not confirm:
        console.print("[yellow]This will freeze the model pipeline. Use --confirm to proceed.[/yellow]")
        console.print("Current state: ", end="")
        from src.config.model_state import is_locked, get_model_state, NEXT_LOCK_VERSION
        state = get_model_state()
        if is_locked():
            console.print(f"[red]LOCKED ({state})[/red]")
        else:
            console.print(f"[green]UNLOCKED[/green]")
            console.print(f"[dim]Next lock version: {NEXT_LOCK_VERSION}[/dim]")
        return
    
    try:
        from src.config.model_state import freeze, get_model_state, NEXT_LOCK_VERSION
        from src.ml.models.corners.team_offsets import TeamOffsetManager
        from src.config import DEFAULT_TRAINING_LEAGUES
        
        # Freeze model state
        freeze()
        console.print(f"[green]✓ Model state frozen: {get_model_state()}[/green]")
        
        # Freeze all team offsets
        manager = TeamOffsetManager()
        for league in DEFAULT_TRAINING_LEAGUES:
            manager.freeze_all(league)
        
        console.print("[green]✓ All team offsets frozen[/green]")
        console.print("\n[bold red]Pipeline is now LOCKED.[/bold red]")
        console.print("[dim]All training, tuning, and offset operations will FAIL until unlocked.[/dim]")
        
    except Exception as e:
        logger.error("Freeze operation failed", exc_info=True)
        raise typer.Exit(code=1)


@app.command("sync-models")
def sync_models(
    push: bool = typer.Option(False, "--push", help="Upload manifest-listed local .pkl models to S3."),
    pull: bool = typer.Option(False, "--pull", help="Download missing manifest-listed models from S3."),
) -> None:
    from rich.console import Console
    import os
    from src.ml.registry import ModelRegistry

    console = Console()
    if push == pull:
        console.print("[red]Specify exactly one of --push or --pull.[/red]")
        raise typer.Exit(code=1)

    bucket = os.environ.get("S3_BUCKET")
    prefix = os.environ.get("S3_PREFIX")
    missing = [name for name, value in {"S3_BUCKET": bucket, "S3_PREFIX": prefix}.items() if not value]
    if missing:
        console.print(
            f"[red]Missing required S3 environment variables: {', '.join(missing)}[/red]"
        )
        raise typer.Exit(code=1)

    try:
        registry = ModelRegistry()
        if push:
            uploaded = registry.push_to_s3(bucket=bucket, prefix=prefix)
            console.print(
                f"[green]Uploaded {uploaded} model artifacts to s3://{bucket.strip().strip('/')}/{prefix.strip().strip('/')}[/green]"
            )
        else:
            downloaded = registry.pull_from_s3(bucket=bucket, prefix=prefix)
            console.print(
                f"[green]Downloaded {downloaded} missing model artifacts from s3://{bucket.strip().strip('/')}/{prefix.strip().strip('/')}[/green]"
            )
    except Exception as exc:
        logger.error("Model sync failed", exc_info=True)
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command("check-models")
def check_models() -> None:
    """
    Validate manifest coverage: check every active_models entry resolves
    to an artifact file that exists on disk. Reports missing and stale models
    per league. Exit code 1 if any artifacts are missing.
    """
    import json
    from pathlib import Path
    from src.config import DATA_DIR

    MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "ml" / "models"
    MANIFEST_PATH = MODELS_DIR / "manifest.json"
    LEAGUES = ["PL", "BL1", "FL1", "SA", "PD"]
    # Model keys expected per league in active_models
    EXPECTED_SUFFIXES = [
        "match_outcome", "home_goals", "home_goals_xgb",
        "away_goals", "away_goals_xgb",
        "corners", "corners_xgb", "cards", "cards_xgb",
        "mh_corn", "ma_corn", "m_corners_lgbm", "m_corners_xgb",
    ]

    if not MANIFEST_PATH.exists():
        typer.echo(f"[ERROR] Manifest not found: {MANIFEST_PATH}", err=True)
        raise typer.Exit(1)

    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        typer.echo(f"[ERROR] Failed to parse manifest: {exc}", err=True)
        raise typer.Exit(1)

    active = manifest.get("active_models", {})
    if not active:
        typer.echo("[ERROR] active_models section missing or empty in manifest.", err=True)
        raise typer.Exit(1)

    missing: list[str] = []
    found: list[str] = []
    not_in_active: list[str] = []

    for league in LEAGUES:
        for suffix in EXPECTED_SUFFIXES:
            key = f"{suffix}_{league}"
            artifact_key = active.get(key)
            if artifact_key is None:
                not_in_active.append(f"  [{league}] {key} — not in active_models")
                continue
            meta = manifest.get(artifact_key)
            if not isinstance(meta, dict):
                missing.append(f"  [{league}] {key} → {artifact_key} — no manifest entry")
                continue
            filename = meta.get("filename")
            if not filename:
                missing.append(f"  [{league}] {key} → {artifact_key} — no filename in entry")
                continue
            artifact_path = MODELS_DIR / filename
            if not artifact_path.exists():
                missing.append(f"  [{league}] {key} → {filename} — FILE MISSING on disk")
            else:
                found.append(f"  [{league}] {key} → {filename} OK")

    typer.echo(f"\n=== Model Coverage Check ===")
    typer.echo(f"Manifest : {MANIFEST_PATH}")
    typer.echo(f"Models dir: {MODELS_DIR}")
    typer.echo(f"Leagues  : {', '.join(LEAGUES)}")
    typer.echo(f"Checks   : {len(LEAGUES) * len(EXPECTED_SUFFIXES)} expected slots\n")

    if found:
        typer.echo(f"[OK] {len(found)} artifacts present on disk")

    calibrator_dir = MODELS_DIR / "calibrators"
    calibrator_artifacts = []
    if calibrator_dir.exists():
        calibrator_artifacts = [
            path
            for pattern in ("*.pkl", "*.joblib")
            for path in calibrator_dir.glob(pattern)
            if path.is_file()
        ]
    typer.echo(f"[INFO] Calibrators: {len(calibrator_artifacts)} artifacts present in calibrators/")

    missing_calibrators: list[str] = []
    for manifest_key, manifest_value in manifest.items():
        if "calibrat" not in str(manifest_key).lower():
            continue

        filename = None
        if isinstance(manifest_value, dict):
            filename = (
                manifest_value.get("filename")
                or manifest_value.get("path")
                or manifest_value.get("calibrator_filename")
            )
        elif isinstance(manifest_value, str):
            filename = manifest_value

        if not filename:
            missing_calibrators.append(f"  {manifest_key} -- no filename in entry")
            continue

        calibrator_path = Path(filename)
        if not calibrator_path.is_absolute():
            calibrator_path = MODELS_DIR / calibrator_path
        if not calibrator_path.exists():
            missing_calibrators.append(f"  {manifest_key} -> {filename} -- FILE MISSING on disk")

    if not_in_active:
        typer.echo(f"\n[WARN] {len(not_in_active)} keys not registered in active_models:")
        for line in not_in_active:
            typer.echo(line)

    if missing_calibrators:
        typer.echo(f"\n[FAIL] {len(missing_calibrators)} missing calibrator artifacts:")
        for line in missing_calibrators:
            typer.echo(line)
        typer.echo("\nRun calibration training to regenerate missing calibrators.")
        raise typer.Exit(1)

    if missing:
        typer.echo(f"\n[FAIL] {len(missing)} missing artifacts:")
        for line in missing:
            typer.echo(line)
        typer.echo("\nRun training to regenerate missing models.")
        raise typer.Exit(1)
    else:
        typer.echo("\n[PASS] All registered artifacts present on disk.")


@app.command("run-daily-pipeline")
def run_daily_pipeline(
    league: str = typer.Option("PL", "--league", "-l", help="League code for fetch, drift, and predictions."),
    season: Optional[str] = typer.Option(None, "--season", help="Season token YYZZ (default: inferred)."),
    lookback: int = typer.Option(120, "--lookback", min=7, max=120, help="Days for drift check."),
    prediction_date: str = typer.Option("today", "--prediction-date", help="Date filter for predictions."),
    show_all_predictions: bool = typer.Option(False, "--all", help="Show all upcoming predictions."),
    tz: str = typer.Option("LOCAL", "--tz", help="Timezone for prediction filtering."),
    simulate: bool = typer.Option(
        True,
        "--simulate/--no-simulate",
        help="Use Monte Carlo simulation in predictions.",
    ),
    reconcile_date: Optional[str] = typer.Option(
        None,
        "--reconcile-date",
        help="Date to reconcile (YYYY-MM-DD). Default: yesterday.",
    ),
    alert_on_failure: bool = typer.Option(
        True,
        "--alert-on-failure/--no-alert-on-failure",
        help="Send CRITICAL alert if any pipeline step fails.",
    ),
    force_alert: bool = typer.Option(
        False,
        "--force-alert",
        help="Bypass alert de-duplication cooldown.",
    ),
) -> None:
    """
    Run the daily ops sequence:
    Data fetch -> Drift check -> Predictions -> Reconciliation.
    """
    from rich.console import Console
    from src.cli.commands.data import fetch_latest_season, fetch_upcoming
    from src.cli.commands.evaluation import reconcile as reconcile_predictions
    from src.cli.commands.prediction import show_predictions
    from src.cli.utils import resolve_league_code

    console = Console()
    resolved_league = resolve_league_code(league)
    if resolved_league is None:
        console.print(f"[red]Unknown league code: {league}[/red]")
        raise typer.Exit(code=1)

    season_token = season or _default_season_token()
    normalized_league = resolved_league.value
    steps = [
        ("Data fetch", lambda: (fetch_latest_season(season=season_token), fetch_upcoming(league=resolved_league))),
        ("Drift check", lambda: check_drift(league=normalized_league, lookback=lookback)),
        (
            "Predictions",
            lambda: show_predictions(
                date=prediction_date,
                league=normalized_league,
                all=show_all_predictions,
                tz=tz,
                simulate=simulate,
            ),
        ),
        ("Reconciliation", lambda: reconcile_predictions(date_str=reconcile_date)),
    ]

    console.print("[bold cyan]Daily Ops Pipeline[/bold cyan]")
    console.print(
        "Data fetch -> Drift check -> Predictions -> Reconciliation"
    )

    current_step = "not_started"
    try:
        for idx, (step_name, run_step) in enumerate(steps, start=1):
            current_step = step_name
            console.print(f"[cyan]{idx}/{len(steps)} {step_name}[/cyan]")
            run_step()
        console.print("[bold green]Pipeline completed successfully.[/bold green]")
    except Exception as exc:
        logger.error(
            "Daily ops pipeline failed",
            exc_info=True,
            extra={"step": current_step, "league": normalized_league, "season": season_token},
        )
        console.print(f"[bold red]Pipeline failed at step: {current_step}[/bold red]")
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        if alert_on_failure:
            alert_sent = _send_pipeline_failure_alert(
                failed_step=current_step,
                league=normalized_league,
                season=season_token,
                lookback=lookback,
                prediction_date=prediction_date,
                reconcile_date=reconcile_date,
                error=exc,
                force=force_alert,
            )
            if alert_sent:
                console.print("[yellow]Failure alert dispatched.[/yellow]")
            else:
                console.print("[yellow]Failure alert dispatch failed.[/yellow]")
        raise typer.Exit(code=1) from exc


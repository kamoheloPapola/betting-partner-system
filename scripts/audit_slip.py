"""
Deep Diagnostic: Audit Suggested Slip Selections.

Reconstructs the feature vectors, model outputs, and adjustments
for each selection in the Forbidden Fruit slip.

Run: python scripts/audit_slip.py
"""
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.features.pipeline import FeaturePipeline
from src.ml.registry import ModelRegistry
from src.ml.models.corners.team_offsets import TeamOffsetManager

console = Console()

def audit_match(home_team: str, away_team: str, league: str, market: str):
    """Audit a specific match prediction."""
    console.rule(f"[bold cyan]{home_team} vs {away_team} ({league})[/bold cyan]")
    
    # 1. Load Features
    pipeline = FeaturePipeline()
    df = pipeline.run(league=league)
    
    # Find the match
    mask = (
        (df['home_team'].str.upper() == home_team.upper()) &
        (df['away_team'].str.upper() == away_team.upper())
    )
    match_row = df[mask]
    
    if match_row.empty:
        console.print(f"[red]ERROR: Match not found in feature matrix![/red]")
        return
    
    match_row = match_row.iloc[0]
    
    # 2. Show Key Features
    feature_table = Table(title="Key Rolling Features", show_header=True)
    feature_table.add_column("Feature", style="cyan")
    feature_table.add_column("Value", style="green")
    
    key_features = [
        'home_rolling_goals_scored_5', 'home_rolling_goals_conceded_5',
        'away_rolling_goals_scored_5', 'away_rolling_goals_conceded_5',
        'home_rolling_corners_scored_5', 'away_rolling_corners_scored_5',
        'home_form_rating', 'away_form_rating',
        'home_days_rest', 'away_days_rest'
    ]
    
    for feat in key_features:
        if feat in match_row:
            val = match_row[feat]
            feature_table.add_row(feat, f"{val:.2f}" if pd.notna(val) else "N/A")
    
    console.print(feature_table)
    
    # 3. Check Offsets
    mgr = TeamOffsetManager()
    home_offset = mgr.get_offset(home_team, league)
    away_offset = mgr.get_offset(away_team, league)
    
    console.print()
    console.print("[bold]Bayesian Offsets Applied:[/bold]")
    if home_offset:
        console.print(f"  {home_team}: corner_bias={home_offset.home_corner_bias:.3f}, matches={home_offset.match_count}")
    else:
        console.print(f"  {home_team}: [yellow]No offset (using global prior)[/yellow]")
        
    if away_offset:
        console.print(f"  {away_team}: corner_bias={away_offset.away_corner_bias:.3f}, matches={away_offset.match_count}")
    else:
        console.print(f"  {away_team}: [yellow]No offset (using global prior)[/yellow]")
    
    # 4. Model Version
    registry = ModelRegistry()
    model_name = "poisson_home_base" if "HOME" in market else "poisson_away_base"
    model_meta = registry.get_production_model_for_league(league, model_name)
    
    console.print()
    console.print("[bold]Model Used:[/bold]")
    if model_meta:
        console.print(f"  Name: {model_meta.get('name')}")
        console.print(f"  Version: {model_meta.get('version')}")
        console.print(f"  Calibration: {model_meta.get('metrics', {}).get('calibration_score', 'N/A'):.4f}")
        console.print(f"  Trained: {model_meta.get('trained_at', 'N/A')}")
    else:
        console.print("  [red]ERROR: No production model found![/red]")
    
    console.print()


if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold green]SUGGESTED SLIP DEEP AUDIT[/bold green]\n"
        "Verifying feature engineering, offsets, and model state for each selection.",
        border_style="green"
    ))
    console.print()
    
    # Today's Suggested Slip (from the Forbidden Fruit output)
    selections = [
        ("Bayern Munich", "VfL Wolfsburg", "BL1", "AWAY_TG_U1.5"),
        ("Levante", "Espanyol", "PD", "HOME_TG_U1.5"),
        ("Rayo Vallecano", "Mallorca", "PD", "CORNERS_U11.5"),
        ("Fiorentina", "AC Milan", "SA", "HOME_TG_U1.5"),
        ("Verona", "Lazio", "SA", "HOME_TG_U1.5"),
    ]
    
    for home, away, league, market in selections:
        audit_match(home, away, league, market)
    
    console.print(Panel.fit(
        "[bold green]AUDIT COMPLETE[/bold green]",
        border_style="green"
    ))

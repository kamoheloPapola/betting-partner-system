"""
Test Calibrated Selection Pipeline.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from rich.console import Console
from src.strategies.forbidden_fruit import ForbiddenFruitEngine

console = Console()

def test_calibrated_selection():
    """Test the calibrated selection pipeline."""
    console.print("=" * 60, style="bold")
    console.print("CALIBRATED SELECTION PIPELINE TEST", style="bold green")
    console.print("=" * 60, style="bold")
    
    engine = ForbiddenFruitEngine()
    
    # Sample match
    match = {
        'home_team': 'LIVERPOOL',
        'away_team': 'CHELSEA',
        'league': 'PL',
        'match_id': 'test123',
        'date': '2026-01-14'
    }
    
    # Sample raw predictions
    raw_predictions = {
        'u25': 0.58,      # Goals under 2.5
        'o25': 0.42,      # Goals over 2.5
        'btts': 0.55,     # BTTS yes
        'btts_no': 0.45,  # BTTS no
    }
    
    # Sample odds
    odds = {
        'goals_u25': 1.75,
        'goals_o25': 2.10,
        'btts_yes': 1.85,
        'btts_no': 1.95,
    }
    
    console.print("\n[cyan]Match:[/cyan]", match)
    console.print("[cyan]Raw Predictions:[/cyan]", raw_predictions)
    console.print("[cyan]Odds:[/cyan]", odds)
    
    console.print("\n[bold]Running calibrated selection...[/bold]\n")
    
    selections = engine.calibrated_selection(
        match=match,
        raw_predictions=raw_predictions,
        odds=odds,
        bankroll=1000.0
    )
    
    console.print(f"\n[bold]Results: {len(selections)} selections[/bold]")
    
    for s in selections:
        console.print(f"  {s['market']:12} | tier={s['tier']:6} | p_raw={s['p_raw']:.3f} | p_cal={s['p_cal']:.3f} | EV={s['ev']:.3f} | stake={s['stake']:.2f}")
    
    if not selections:
        console.print("[yellow]No selections passed all filters (expected for low confidence)[/yellow]")
    
    console.print("\n[green]✅ Pipeline executed without errors[/green]")
    return True

if __name__ == "__main__":
    test_calibrated_selection()

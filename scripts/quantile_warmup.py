"""
Quantile Warm-Up Script.

Run predictions-only mode for 7-14 days to populate rolling windows.
No betting, no overrides - just data collection.

Once windows >= 500 samples, tiers unlock naturally.

Usage:
    python scripts/quantile_warmup.py
    
    # Run daily via cron/task scheduler:
    # 0 6 * * * cd /path/to/project && python scripts/quantile_warmup.py
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.features.pipeline import FeaturePipeline
from src.cli.commands.prediction import _load_prediction_models, _calculate_probabilities
from src.ml.calibration import get_calibrator, get_sharpness_gate
from src.monitoring.confidence_drift import get_stratifier
from src.config import DATA_DIR

console = Console()

LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']
WARMUP_LOG = DATA_DIR / "calibration" / "warmup_log.json"

# Markets to populate
MARKET_MAP = {
    'u25': 'goals_u25',
    'o25': 'goals_o25',
    'btts': 'btts_yes',
    'btts_no': 'btts_no',
}


def run_warmup():
    """Run predictions-only warm-up to populate quantile rolling windows."""
    console.print("=" * 60, style="bold")
    console.print("QUANTILE WARM-UP MODE", style="bold magenta")
    console.print("=" * 60, style="bold")
    console.print("[yellow]NO BETTING - DATA COLLECTION ONLY[/yellow]\n")
    
    calibrator = get_calibrator()
    sharpness = get_sharpness_gate()
    stratifier = get_stratifier()
    
    stats = {
        'timestamp': datetime.now().isoformat(),
        'markets': {}
    }
    
    for league in LEAGUES:
        console.print(f"\n[cyan][{league}] Processing...[/cyan]")
        
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            suite = _load_prediction_models(league)
            
            # Only process recent matches (last 30 days)
            df['date'] = pd.to_datetime(df['date'])
            cutoff = datetime.now() - pd.Timedelta(days=30)
            df = df[df['date'] >= cutoff]
            
            console.print(f"  {len(df)} matches in last 30 days")
            
            for _, match in track(df.iterrows(), total=len(df), description=f"[{league}]"):
                try:
                    probs, _ = _calculate_probabilities(match, suite, league)
                    
                    # Collect calibrated probs for each market
                    for pred_key, market in MARKET_MAP.items():
                        if pred_key not in probs:
                            continue
                        
                        p_raw = probs[pred_key]
                        
                        # Check sharpness first
                        if sharpness.get_status(market) == "DISABLED":
                            continue
                        
                        # Calibrate
                        p_cal = calibrator.calibrate(market, p_raw)
                        if p_cal is None:
                            continue
                        
                        # Add to rolling window
                        stratifier.add_prediction(market, p_cal)
                        
                        # Track stats
                        if market not in stats['markets']:
                            stats['markets'][market] = {'count': 0}
                        stats['markets'][market]['count'] += 1
                        
                except Exception:
                    continue
                    
        except Exception as e:
            console.print(f"[{league}] [red]Error: {e}[/red]")
    
    # Save stratifier state
    stratifier.save()
    
    # Log warm-up run
    WARMUP_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WARMUP_LOG, 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    console.print("\n" + "=" * 60, style="bold")
    console.print("WARM-UP SUMMARY", style="bold")
    console.print("=" * 60, style="bold")
    
    table = Table(title="Quantile Window Status")
    table.add_column("Market")
    table.add_column("Samples", justify="right")
    table.add_column("Status")
    table.add_column("Tiers")
    
    for market in ['goals_u25', 'goals_o25', 'btts_yes', 'btts_no']:
        pct = stratifier.get_percentiles(market)
        samples = len(stratifier.rolling_predictions.get(market, []))
        
        if samples >= 500:
            status = "[green]UNLOCKED[/green]"
            tiers = "A/B/C"
        elif samples >= 50:
            status = "[yellow]PARTIAL[/yellow]"
            tiers = "A only"
        else:
            status = "[red]LOCKED[/red]"
            tiers = "None"
        
        table.add_row(market.upper(), str(samples), status, tiers)
    
    console.print(table)
    
    console.print(f"\n[bold]Run this script daily until all markets show UNLOCKED.[/bold]")
    console.print(f"Target: 500+ samples per market (typically 7-14 days)\n")


def check_status():
    """Check current warm-up status without running predictions."""
    console.print("=" * 60, style="bold")
    console.print("QUANTILE WARM-UP STATUS", style="bold cyan")
    console.print("=" * 60, style="bold")
    
    stratifier = get_stratifier()
    
    table = Table(title="Current Window Status")
    table.add_column("Market")
    table.add_column("Samples", justify="right")
    table.add_column("Status")
    table.add_column("Q80", justify="right")
    table.add_column("Q90", justify="right")
    table.add_column("Q95", justify="right")
    
    for market in ['goals_u25', 'goals_o25', 'btts_yes', 'btts_no']:
        samples = len(stratifier.rolling_predictions.get(market, []))
        pct = stratifier.get_percentiles(market) or {}
        
        if samples >= 500:
            status = "[green]READY[/green]"
        elif samples >= 50:
            status = "[yellow]PARTIAL[/yellow]"
        else:
            status = "[red]WAITING[/red]"
        
        table.add_row(
            market.upper(),
            str(samples),
            status,
            f"{pct.get('q80', 0):.3f}" if pct else "-",
            f"{pct.get('q90', 0):.3f}" if pct else "-",
            f"{pct.get('q95', 0):.3f}" if pct else "-"
        )
    
    console.print(table)
    
    # Check if ready
    ready_count = sum(
        1 for m in ['goals_u25', 'goals_o25']
        if len(stratifier.rolling_predictions.get(m, [])) >= 500
    )
    
    if ready_count >= 2:
        console.print("\n[green bold]✅ WARM-UP COMPLETE - Ready for live betting[/green bold]")
    else:
        console.print(f"\n[yellow]⏳ {ready_count}/2 markets ready. Continue daily warm-up.[/yellow]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantile Warm-Up")
    parser.add_argument("--status", action="store_true", help="Check status only")
    args = parser.parse_args()
    
    if args.status:
        check_status()
    else:
        run_warmup()

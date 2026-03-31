"""
Deep Investigation: High Confidence Predictions (FIXED)

Check corn_o75 and card_o25 calibration.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.features.pipeline import FeaturePipeline
from src.cli.commands.prediction import _load_prediction_models, _calculate_probabilities
from src.config.leagues import FINISHED_STATUSES

console = Console()
LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']


def investigate_market(market_name: str, pred_key: str, outcome_fn):
    """Investigate a market's calibration."""
    console.print(f"\n[bold cyan]= {market_name.upper()} =[/bold cyan]")
    
    data = []
    
    for league in LEAGUES:
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            suite = _load_prediction_models(league)
            
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['status'].str.upper().isin(FINISHED_STATUSES)]
            
            # Last year
            cutoff = df['date'].max() - pd.Timedelta(days=365)
            df = df[df['date'] >= cutoff]
            
            for _, match in track(df.iterrows(), total=len(df), description=f"[{league}]"):
                try:
                    probs, _ = _calculate_probabilities(match, suite, league)
                    
                    if pred_key not in probs:
                        continue
                    
                    conf = probs[pred_key]
                    outcome = outcome_fn(match)
                    
                    if pd.isna(outcome):
                        continue
                    
                    data.append({
                        'league': league,
                        'conf': conf,
                        'outcome': 1 if outcome else 0
                    })
                    
                except Exception:
                    continue
                    
        except Exception as e:
            console.print(f"[{league}] [red]Error: {e}[/red]")
    
    if not data:
        console.print("[red]No data collected[/red]")
        return {}
    
    df_results = pd.DataFrame(data)
    
    # Bucket analysis
    buckets = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]
    
    table = Table(title=f"{market_name.upper()} Calibration")
    table.add_column("Bucket")
    table.add_column("N", justify="right")
    table.add_column("Conf", justify="right")
    table.add_column("Hit%", justify="right")
    table.add_column("Gap", justify="right")
    table.add_column("Status")
    
    for low, high in buckets:
        bucket = df_results[(df_results['conf'] >= low) & (df_results['conf'] < high)]
        if len(bucket) < 10:
            continue
        
        mean_conf = bucket['conf'].mean()
        hit_rate = bucket['outcome'].mean()
        gap = mean_conf - hit_rate
        
        status = "[red]BAD[/red]" if gap > 0.10 else "[yellow]WARN[/yellow]" if gap > 0.05 else "[green]OK[/green]"
        
        table.add_row(f"{low:.0%}-{high:.0%}", str(len(bucket)), f"{mean_conf:.1%}", f"{hit_rate:.1%}", f"{gap:+.1%}", status)
    
    console.print(table)
    
    # High confidence (>80%)
    high = df_results[df_results['conf'] >= 0.80]
    if len(high) > 0:
        console.print(f"\n[bold]HIGH CONF (>80%): n={len(high)} | conf={high['conf'].mean():.1%} | hit={high['outcome'].mean():.1%} | gap={high['conf'].mean() - high['outcome'].mean():+.1%}[/bold]")
    
    return {
        'n': len(df_results),
        'mean_conf': df_results['conf'].mean(),
        'hit_rate': df_results['outcome'].mean(),
        'std': df_results['conf'].std()
    }


def main():
    console.print("=" * 60, style="bold")
    console.print("DEEP CONFIDENCE INVESTIGATION", style="bold magenta")
    console.print("=" * 60, style="bold")
    
    # CORN O7.5
    def corn_o75_outcome(row):
        hc = row.get('home_corners')
        ac = row.get('away_corners')
        if pd.isna(hc) or pd.isna(ac):
            return None
        return (hc + ac) > 7.5
    
    corn_stats = investigate_market("Corners O7.5", "corn_o75", corn_o75_outcome)
    
    # CARD O2.5
    def card_o25_outcome(row):
        hc = row.get('home_cards')
        ac = row.get('away_cards')
        if pd.isna(hc) or pd.isna(ac):
            return None
        return (hc + ac) > 2.5
    
    card_stats = investigate_market("Cards O2.5", "card_o25", card_o25_outcome)
    
    # Summary
    console.print("\n" + "=" * 60, style="bold")
    console.print("SUMMARY", style="bold")
    console.print("=" * 60, style="bold")
    
    if corn_stats:
        console.print(f"CORN O7.5: std={corn_stats.get('std', 0):.4f}")
        if corn_stats.get('std', 0) < 0.06:
            console.print("  [red]⚠️ LOW DISCRIMINATION - likely model issue[/red]")
    
    if card_stats:
        console.print(f"CARD O2.5: std={card_stats.get('std', 0):.4f}")
        if card_stats.get('std', 0) < 0.06:
            console.print("  [red]⚠️ LOW DISCRIMINATION - likely model issue[/red]")


if __name__ == "__main__":
    main()

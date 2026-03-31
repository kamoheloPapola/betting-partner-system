"""
ECE Bucket Population Script.

Computes real Expected Calibration Error (ECE) per:
- League (PL, PD, SA, BL1, FL1)
- Market (goals_u25, goals_o25, btts_yes, corn_o75, etc.)
- Probability Bucket (0.50-0.60, 0.60-0.70, etc.)

Outputs Python code to update calibration_penalty.py ECE_BUCKETS.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.features.pipeline import FeaturePipeline
from src.cli.commands.prediction import _load_prediction_models, _calculate_probabilities
from src.config.leagues import FINISHED_STATUSES
from src.config import DATA_DIR

console = Console()

LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']
LOOKBACK_DAYS = 365  # 1 year of data

# Markets to analyze
MARKET_OUTCOMES = {
    'u25': ('goals_u25', lambda r: (r['home_score'] + r['away_score']) < 2.5),
    'o25': ('goals_o25', lambda r: (r['home_score'] + r['away_score']) > 2.5),
    'btts': ('btts_yes', lambda r: r['home_score'] > 0 and r['away_score'] > 0),
    'btts_no': ('btts_no', lambda r: r['home_score'] == 0 or r['away_score'] == 0),
    'corn_o75': ('corn_o75', lambda r: (r.get('home_corners', 0) + r.get('away_corners', 0)) > 7.5),
    'corn_u11': ('corn_u11', lambda r: (r.get('home_corners', 0) + r.get('away_corners', 0)) < 11.5),
    'card_o25': ('card_o25', lambda r: r.get('match_total_cards', 0) > 2.5),
    'card_u45': ('card_u45', lambda r: r.get('match_total_cards', 0) < 4.5),
    'card_u55': ('card_u55', lambda r: r.get('match_total_cards', 0) < 5.5),
    'home_under_1_5': ('home_under_1_5', lambda r: r['home_score'] < 1.5),
    'away_under_1_5': ('away_under_1_5', lambda r: r['away_score'] < 1.5),
}

BUCKETS = [
    ('0.50-0.60', 0.50, 0.60),
    ('0.60-0.70', 0.60, 0.70),
    ('0.70-0.80', 0.70, 0.80),
    ('0.80-0.90', 0.80, 0.90),
    ('0.90-1.00', 0.90, 1.00),
]

OUTPUT_PATH = DATA_DIR / "calibration" / "ece_buckets.json"


def get_bucket(prob: float) -> str:
    for name, low, high in BUCKETS:
        if low <= prob < high:
            return name
    if prob >= 1.0:
        return '0.90-1.00'
    return '0.00-0.50'


def compute_ece_buckets():
    """Compute ECE for each league/market/bucket combination."""
    console.print("=" * 70)
    console.print("[bold magenta]ECE BUCKET POPULATION[/bold magenta]")
    console.print("=" * 70)
    console.print(f"Lookback: {LOOKBACK_DAYS} days | Markets: {len(MARKET_OUTCOMES)}")
    
    # Collect predictions by bucket
    # {(league, market, bucket): [(prob, outcome), ...]}
    bucket_data = defaultdict(list)
    
    for league in LEAGUES:
        console.print(f"\n[cyan][{league}] Processing...[/cyan]")
        
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            suite = _load_prediction_models(league)
            
            df['date'] = pd.to_datetime(df['date'])
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS)
            
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            df = df[
                (df['date'] >= cutoff) &
                (df['status'].str.upper().isin(FINISHED_STATUSES)) &
                (df['home_score'].notna()) &
                (df['away_score'].notna())
            ]
            
            console.print(f"  {len(df)} matches in window")
            
            for _, match in track(df.iterrows(), total=len(df), description=f"[{league}]"):
                try:
                    probs, _ = _calculate_probabilities(match, suite, league)
                    
                    for pred_key, (market, outcome_fn) in MARKET_OUTCOMES.items():
                        if pred_key not in probs:
                            continue
                        
                        prob = probs[pred_key]
                        bucket = get_bucket(prob)
                        
                        if bucket == '0.00-0.50':
                            continue  # Skip low confidence
                        
                        try:
                            outcome = 1 if outcome_fn(match) else 0
                        except Exception:
                            continue
                        
                        bucket_data[(league, market, bucket)].append((prob, outcome))
                        
                except Exception:
                    continue
                    
        except Exception as e:
            console.print(f"[{league}] [red]Error: {e}[/red]")
    
    return bucket_data


def calculate_ece(data: list) -> tuple:
    """Calculate ECE for a list of (prob, outcome) tuples."""
    if len(data) < 10:
        return None, len(data)
    
    probs = np.array([p for p, _ in data])
    outcomes = np.array([o for _, o in data])
    
    mean_prob = np.mean(probs)
    hit_rate = np.mean(outcomes)
    ece = abs(mean_prob - hit_rate)
    
    return ece, len(data)


def generate_output(bucket_data: dict):
    """Generate ECE_BUCKETS dict and save."""
    console.print("\n" + "=" * 70)
    console.print("[bold]ECE RESULTS[/bold]")
    console.print("=" * 70)
    
    ece_buckets = {}
    
    table = Table(title="ECE by Bucket")
    table.add_column("Key")
    table.add_column("N", justify="right")
    table.add_column("Mean Conf", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Status")
    
    for key, data in sorted(bucket_data.items()):
        league, market, bucket = key
        ece, n = calculate_ece(data)
        
        if ece is None:
            continue
        
        probs = [p for p, _ in data]
        outcomes = [o for _, o in data]
        mean_conf = np.mean(probs)
        hit_rate = np.mean(outcomes)
        
        ece_buckets[key] = round(ece, 4)
        
        if ece > 0.10:
            status = "[red]BAD[/red]"
        elif ece > 0.05:
            status = "[yellow]WARN[/yellow]"
        else:
            status = "[green]GOOD[/green]"
        
        table.add_row(
            f"({league}, {market}, {bucket})",
            str(n),
            f"{mean_conf:.1%}",
            f"{hit_rate:.1%}",
            f"{ece:.4f}",
            status
        )
    
    console.print(table)
    
    # Save to JSON
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tuple keys to strings for JSON
    json_data = {
        str(k): v for k, v in ece_buckets.items()
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    console.print(f"\n[bold]Saved to: {OUTPUT_PATH}[/bold]")
    
    # Generate Python code
    console.print("\n[bold cyan]Python code to add to calibration_penalty.py:[/bold cyan]")
    console.print("```python")
    console.print("ECE_BUCKETS = {")
    for key, ece in sorted(ece_buckets.items()):
        console.print(f"    {key}: {ece},")
    console.print("}")
    console.print("```")
    
    return ece_buckets


def main():
    bucket_data = compute_ece_buckets()
    ece_buckets = generate_output(bucket_data)
    
    console.print("\n" + "=" * 70)
    console.print("[bold green]ECE BUCKET POPULATION COMPLETE[/bold green]")
    console.print("=" * 70)
    
    # Summary
    good = sum(1 for e in ece_buckets.values() if e <= 0.05)
    warn = sum(1 for e in ece_buckets.values() if 0.05 < e <= 0.10)
    bad = sum(1 for e in ece_buckets.values() if e > 0.10)
    
    console.print(f"  [green]GOOD (ECE ≤ 5%): {good}[/green]")
    console.print(f"  [yellow]WARN (5% < ECE ≤ 10%): {warn}[/yellow]")
    console.print(f"  [red]BAD (ECE > 10%): {bad}[/red]")


if __name__ == "__main__":
    main()

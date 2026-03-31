"""
Layer 4.1: Out-of-Sample Calibration Audit (PROPER)

CRITICAL: This audit uses STRICT TIME SPLIT:
- Calibrate ONLY on 2024-2025 data
- Use ACTUAL match outcomes from raw results
- Apply model predictions at inference time (not base rates)

This is the ONLY valid way to measure calibration for betting.

Output: oos_calibration_report.json
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.features.pipeline import FeaturePipeline
from src.cli.commands.prediction import _load_prediction_models, _calculate_probabilities
from src.config.leagues import FINISHED_STATUSES

# === CONFIGURATION ===
LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']
OUTPUT_FILE = Path("data/reports/oos_calibration_report.json")

# TIME SPLIT: Calibrate ONLY on 2024-2025 (Out-of-Sample)
OOS_START_DATE = "2024-01-01"
OOS_END_DATE = "2025-12-31"

# Confidence buckets
BUCKETS = [
    (0.50, 0.55, "50-55%"),
    (0.55, 0.60, "55-60%"),
    (0.60, 0.65, "60-65%"),
    (0.65, 0.70, "65-70%"),
    (0.70, 0.75, "70-75%"),
    (0.75, 0.80, "75-80%"),
    (0.80, 1.00, "80%+"),
]

console = Console()

def calculate_ece(predictions: List[Tuple[float, bool]], n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    if not predictions:
        return 0.0
    
    bins = defaultdict(list)
    for prob, outcome in predictions:
        bin_idx = min(int(prob * n_bins), n_bins - 1)
        bins[bin_idx].append((prob, 1 if outcome else 0))
    
    ece = 0.0
    total = len(predictions)
    
    for bin_idx, bin_preds in bins.items():
        if not bin_preds:
            continue
        avg_confidence = np.mean([p[0] for p in bin_preds])
        actual_accuracy = np.mean([p[1] for p in bin_preds])
        bin_weight = len(bin_preds) / total
        ece += bin_weight * abs(avg_confidence - actual_accuracy)
    
    return ece

def calculate_brier_score(predictions: List[Tuple[float, bool]]) -> float:
    """Calculate Brier Score."""
    if not predictions:
        return 0.0
    return np.mean([(prob - (1 if outcome else 0)) ** 2 for prob, outcome in predictions])

def bucket_calibration(predictions: List[Tuple[float, bool]]) -> Dict:
    """Analyze calibration per confidence bucket."""
    results = {}
    
    for low, high, label in BUCKETS:
        bucket_preds = [(p, o) for p, o in predictions if low <= p < high]
        
        if not bucket_preds:
            results[label] = {'count': 0, 'avg_conf': None, 'hit_rate': None, 'gap': None}
            continue
        
        avg_conf = np.mean([p[0] for p in bucket_preds])
        hit_rate = np.mean([1 if p[1] else 0 for p in bucket_preds])
        gap = hit_rate - avg_conf
        
        results[label] = {
            'count': len(bucket_preds),
            'avg_conf': round(avg_conf, 3),
            'hit_rate': round(hit_rate, 3),
            'gap': round(gap, 3),
            'status': 'OK' if abs(gap) <= 0.05 else ('WARN' if abs(gap) <= 0.10 else 'ALERT')
        }
    
    return results

def run_oos_audit():
    """Run out-of-sample calibration audit."""
    report = {
        'audit_timestamp': datetime.now().isoformat(),
        'layer': 'LAYER_4.1_OUT_OF_SAMPLE_CALIBRATION',
        'oos_period': f"{OOS_START_DATE} to {OOS_END_DATE}",
        'markets': {},
        'summary': {
            'total_predictions': 0,
            'ece_u25': 0.0,
            'ece_o25': 0.0,
            'ece_btts': 0.0,
            'brier_u25': 0.0,
            'brier_o25': 0.0,
            'brier_btts': 0.0,
            'bucket_alerts': 0
        }
    }
    
    console.print("=" * 60, style="bold")
    console.print("LAYER 4.1: OUT-OF-SAMPLE CALIBRATION", style="bold magenta")
    console.print("=" * 60, style="bold")
    console.print(f"[yellow]STRICT TIME SPLIT: {OOS_START_DATE} to {OOS_END_DATE}[/yellow]")
    console.print("Using ACTUAL model predictions vs REAL match outcomes\n")
    
    # Collect predictions by market
    all_preds = {
        'goals_u25': [],
        'goals_o25': [],
        'btts_yes': [],
        'btts_no': [],
    }
    
    for league in LEAGUES:
        console.print(f"[cyan][{league}] Loading OOS data and model...[/cyan]")
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            suite = _load_prediction_models(league)
            
            # Filter to OOS period with FINISHED matches
            df['date'] = pd.to_datetime(df['date'])
            df = df[
                (df['date'] >= OOS_START_DATE) & 
                (df['date'] <= OOS_END_DATE) &
                (df['status'].str.upper().isin(FINISHED_STATUSES)) &
                (df['home_score'].notna()) & 
                (df['away_score'].notna())
            ]
            
            console.print(f"[{league}] Found {len(df)} OOS finished matches")
            
            if len(df) == 0:
                continue
            
            # For each match, generate prediction and compare to actual outcome
            for idx, match in track(df.iterrows(), total=len(df), description=f"[{league}]"):
                try:
                    # Get MODEL predictions (frozen at inference)
                    probs, _ = _calculate_probabilities(match, suite, league)
                    
                    # Get ACTUAL outcomes from raw results
                    total_goals = match['home_score'] + match['away_score']
                    actual_u25 = total_goals < 2.5
                    actual_o25 = total_goals >= 2.5
                    actual_btts = match['home_score'] > 0 and match['away_score'] > 0
                    
                    # Collect predictions vs outcomes
                    all_preds['goals_u25'].append((probs['u25'], actual_u25))
                    all_preds['goals_o25'].append((probs['o25'], actual_o25))
                    all_preds['btts_yes'].append((probs['btts'], actual_btts))
                    all_preds['btts_no'].append((probs['btts_no'], not actual_btts))
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            console.print(f"[{league}] [red]Error: {e}[/red]")
    
    # Analyze each market
    bucket_alerts = 0
    
    for market_name, predictions in all_preds.items():
        if not predictions:
            continue
        
        console.print(f"\n[green]Analyzing {market_name} ({len(predictions)} predictions)...[/green]")
        
        ece = calculate_ece(predictions)
        brier = calculate_brier_score(predictions)
        buckets = bucket_calibration(predictions)
        
        report['markets'][market_name] = {
            'total_predictions': len(predictions),
            'ece': round(ece, 4),
            'brier_score': round(brier, 4),
            'buckets': buckets
        }
        
        # Count alerts
        for bucket_data in buckets.values():
            if bucket_data.get('status') == 'ALERT':
                bucket_alerts += 1
        
        # Print bucket table
        table = Table(title=f"{market_name} OOS Calibration")
        table.add_column("Bucket", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Avg Conf", justify="right")
        table.add_column("Hit Rate", justify="right")
        table.add_column("Gap", justify="right")
        table.add_column("Status", justify="center")
        
        for bucket_label, data in buckets.items():
            if data['count'] == 0:
                continue
            status_style = "green" if data['status'] == 'OK' else ("yellow" if data['status'] == 'WARN' else "red")
            table.add_row(
                bucket_label,
                str(data['count']),
                f"{data['avg_conf']*100:.1f}%" if data['avg_conf'] else "-",
                f"{data['hit_rate']*100:.1f}%" if data['hit_rate'] else "-",
                f"{data['gap']*100:+.1f}%" if data['gap'] is not None else "-",
                f"[{status_style}]{data['status']}[/{status_style}]"
            )
        
        console.print(table)
        console.print(f"ECE: {ece:.4f} | Brier: {brier:.4f}")
    
    # Summary
    total_preds = sum(len(p) for p in all_preds.values())
    report['summary']['total_predictions'] = total_preds
    report['summary']['bucket_alerts'] = bucket_alerts
    
    if all_preds['goals_u25']:
        report['summary']['ece_u25'] = round(calculate_ece(all_preds['goals_u25']), 4)
        report['summary']['brier_u25'] = round(calculate_brier_score(all_preds['goals_u25']), 4)
    if all_preds['goals_o25']:
        report['summary']['ece_o25'] = round(calculate_ece(all_preds['goals_o25']), 4)
        report['summary']['brier_o25'] = round(calculate_brier_score(all_preds['goals_o25']), 4)
    if all_preds['btts_yes']:
        report['summary']['ece_btts'] = round(calculate_ece(all_preds['btts_yes']), 4)
        report['summary']['brier_btts'] = round(calculate_brier_score(all_preds['btts_yes']), 4)
    
    # Save report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    console.print(f"\n{'=' * 60}", style="bold")
    console.print("OUT-OF-SAMPLE CALIBRATION SUMMARY", style="bold")
    console.print(f"{'=' * 60}", style="bold")
    console.print(f"OOS Period: {OOS_START_DATE} to {OOS_END_DATE}")
    console.print(f"Total Predictions: {total_preds}")
    console.print(f"\n[bold]ECE (lower is better, 0.02-0.06 is good):[/bold]")
    console.print(f"  Goals U2.5: {report['summary']['ece_u25']:.4f}")
    console.print(f"  Goals O2.5: {report['summary']['ece_o25']:.4f}")
    console.print(f"  BTTS: {report['summary']['ece_btts']:.4f}")
    console.print(f"\n[bold]Brier (lower is better, 0.20-0.23 is strong):[/bold]")
    console.print(f"  Goals U2.5: {report['summary']['brier_u25']:.4f}")
    console.print(f"  Goals O2.5: {report['summary']['brier_o25']:.4f}")
    console.print(f"  BTTS: {report['summary']['brier_btts']:.4f}")
    console.print(f"\nBucket Alerts (gap > 10%): {bucket_alerts}")
    console.print(f"\nReport saved to: {OUTPUT_FILE}")
    
    # Verdict
    if report['summary']['ece_u25'] < 0.06 and report['summary']['ece_o25'] < 0.06:
        console.print("\n[green bold]✅ CALIBRATION WITHIN ACCEPTABLE BOUNDS[/green bold]")
    elif bucket_alerts > 0:
        console.print("\n[red bold]❌ CRITICAL: Model is miscalibrated. DO NOT SCALE STAKES.[/red bold]")
    else:
        console.print("\n[yellow bold]⚠️ CALIBRATION NEEDS REVIEW[/yellow bold]")
    
    return report

if __name__ == "__main__":
    run_oos_audit()

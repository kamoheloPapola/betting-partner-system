"""
Train Calibrators Script.

Trains isotonic calibrators for all markets using OOS backtest data (2022-2023).
Validates with sharpness gates and saves to disk.
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
from src.ml.calibration import MarketCalibrator, SharpnessGate, get_calibrator, get_sharpness_gate
from src.config.leagues import FINISHED_STATUSES

console = Console()

# Training window: 2022-2023 (OOS from model training perspective)
TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31"

LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']


def collect_training_data():
    """Collect raw predictions and outcomes for training."""
    console.print("=" * 60, style="bold")
    console.print("CALIBRATOR TRAINING", style="bold blue")
    console.print("=" * 60, style="bold")
    console.print(f"Training window: {TRAIN_START} to {TRAIN_END}\n")
    
    # Collect per market
    data = {
        'goals_u25': {'p_raw': [], 'y': []},
        'goals_o25': {'p_raw': [], 'y': []},
        'btts_yes': {'p_raw': [], 'y': []},
        'btts_no': {'p_raw': [], 'y': []},
    }
    
    for league in LEAGUES:
        console.print(f"[cyan][{league}] Loading data...[/cyan]")
        
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            suite = _load_prediction_models(league)
            
            # Filter to training window
            df['date'] = pd.to_datetime(df['date'])
            df = df[
                (df['date'] >= TRAIN_START) &
                (df['date'] <= TRAIN_END) &
                (df['status'].str.upper().isin(FINISHED_STATUSES)) &
                (df['home_score'].notna()) &
                (df['away_score'].notna())
            ]
            
            console.print(f"  {len(df)} matches in training window")
            
            for _, match in track(df.iterrows(), total=len(df), description=f"[{league}]"):
                try:
                    probs, _ = _calculate_probabilities(match, suite, league)
                    
                    # Compute outcomes
                    total_goals = match['home_score'] + match['away_score']
                    btts = match['home_score'] > 0 and match['away_score'] > 0
                    
                    # Collect
                    data['goals_u25']['p_raw'].append(probs['u25'])
                    data['goals_u25']['y'].append(1 if total_goals < 2.5 else 0)
                    
                    data['goals_o25']['p_raw'].append(probs['o25'])
                    data['goals_o25']['y'].append(1 if total_goals >= 2.5 else 0)
                    
                    data['btts_yes']['p_raw'].append(probs['btts'])
                    data['btts_yes']['y'].append(1 if btts else 0)
                    
                    data['btts_no']['p_raw'].append(probs['btts_no'])
                    data['btts_no']['y'].append(1 if not btts else 0)
                    
                except Exception:
                    continue
                    
        except Exception as e:
            console.print(f"[{league}] [red]Error: {e}[/red]")
    
    return data


def train_and_validate(data):
    """Train calibrators and run sharpness gates."""
    calibrator = MarketCalibrator()
    sharpness = SharpnessGate()
    
    results = []
    
    for market, d in data.items():
        p_raw = np.array(d['p_raw'])
        y = np.array(d['y'])
        
        console.print(f"\n[green]Training {market}...[/green]")
        console.print(f"  Samples: {len(p_raw)} | Unique probs: {np.unique(p_raw).size}")
        
        # Fit calibrator
        success = calibrator.fit(market, p_raw, y)
        
        if not success:
            results.append({
                'market': market,
                'status': 'DISABLED',
                'reason': 'Calibration failed',
                'ece': None
            })
            continue
        
        # Get calibrated probs for sharpness check
        p_cal = np.array([calibrator.calibrate(market, p) for p in p_raw])
        
        # Sharpness gate
        gate_status = sharpness.check_market(market, p_cal, y)
        
        metrics = calibrator.get_validation(market)
        
        results.append({
            'market': market,
            'status': gate_status,
            'ece': metrics['ece'] if metrics else None,
            'auc': sharpness.get_metrics(market)['auc'] if sharpness.get_metrics(market) else None,
            'std': sharpness.get_metrics(market)['std'] if sharpness.get_metrics(market) else None,
        })
    
    # Save calibrators
    calibrator.save(version="v1")
    console.print("\n[bold]Calibrators saved to models/calibrators/[/bold]")
    
    return results


def print_results(results):
    """Print training results."""
    table = Table(title="Calibrator Training Results")
    table.add_column("Market")
    table.add_column("Status")
    table.add_column("ECE")
    table.add_column("AUC")
    table.add_column("STD")
    
    for r in results:
        status_style = "green" if r['status'] == 'ACTIVE' else "red"
        table.add_row(
            r['market'].upper(),
            f"[{status_style}]{r['status']}[/{status_style}]",
            f"{r['ece']:.4f}" if r['ece'] else "-",
            f"{r['auc']:.3f}" if r['auc'] else "-",
            f"{r['std']:.4f}" if r['std'] else "-"
        )
    
    console.print(table)
    
    # Summary
    active = sum(1 for r in results if r['status'] == 'ACTIVE')
    disabled = len(results) - active
    
    console.print(f"\n[bold]Summary: {active} ACTIVE, {disabled} DISABLED[/bold]")


if __name__ == "__main__":
    data = collect_training_data()
    results = train_and_validate(data)
    print_results(results)

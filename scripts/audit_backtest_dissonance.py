"""
Backtest Dissonance Analysis

For every failed high-confidence bet, compare:
1. Backtest hit rate for that market
2. Live confidence distribution
3. Calibration curve

Red flag patterns:
- Backtest OK, live bad → data shift
- Backtest inflated → leakage
- Live confidence spiky → overconfidence

Output: backtest_dissonance_report.json
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

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
OUTPUT_FILE = Path("data/reports/backtest_dissonance_report.json")

# Backtest period: Training data (2017-2023)
BACKTEST_END = "2023-12-31"

# Live period: Out-of-sample (2024+)
LIVE_START = "2024-01-01"

# High confidence threshold
HIGH_CONF_THRESHOLD = 0.65

console = Console()

def calculate_market_hit_rates(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Calculate actual hit rates for each market from historical data."""
    hit_rates = {}
    
    # Compute total goals and BTTS
    df = df.copy()
    df['total_goals'] = df['home_score'] + df['away_score']
    df['btts'] = (df['home_score'] > 0) & (df['away_score'] > 0)
    
    # Goals markets
    if 'total_goals' in df.columns:
        valid = df[df['total_goals'].notna()]
        if len(valid) > 0:
            hit_rates['goals_u25'] = {
                'hit_rate': (valid['total_goals'] < 2.5).mean(),
                'sample_size': len(valid)
            }
            hit_rates['goals_o25'] = {
                'hit_rate': (valid['total_goals'] >= 2.5).mean(),
                'sample_size': len(valid)
            }
            hit_rates['goals_u35'] = {
                'hit_rate': (valid['total_goals'] < 3.5).mean(),
                'sample_size': len(valid)
            }
    
    # BTTS markets
    if 'btts' in df.columns:
        valid = df[df['btts'].notna()]
        if len(valid) > 0:
            hit_rates['btts_yes'] = {
                'hit_rate': valid['btts'].mean(),
                'sample_size': len(valid)
            }
            hit_rates['btts_no'] = {
                'hit_rate': (~valid['btts']).mean(),
                'sample_size': len(valid)
            }
    
    return hit_rates

def analyze_confidence_distribution(predictions: List[float]) -> Dict[str, Any]:
    """Analyze the distribution of confidence values."""
    if not predictions:
        return {}
    
    arr = np.array(predictions)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p50': float(np.percentile(arr, 50)),
        'p75': float(np.percentile(arr, 75)),
        'count': len(arr),
        'spiky': float(np.std(arr)) < 0.05,  # Low variance = spiky distribution
        'high_conf_ratio': float((arr >= HIGH_CONF_THRESHOLD).mean())
    }

def detect_dissonance_pattern(
    backtest_hit: float,
    live_hit: float,
    live_conf_mean: float,
    live_conf_std: float
) -> List[str]:
    """Detect red flag patterns."""
    flags = []
    
    # Data shift: Backtest OK but live bad
    if backtest_hit >= 0.60 and live_hit < 0.50:
        flags.append("DATA_SHIFT: Backtest OK ({:.1%}) but live failing ({:.1%})".format(
            backtest_hit, live_hit
        ))
    
    # Leakage: Backtest too good to be true
    if backtest_hit >= 0.75:
        flags.append("POSSIBLE_LEAKAGE: Backtest hit rate suspiciously high ({:.1%})".format(
            backtest_hit
        ))
    
    # Overconfidence: High mean confidence but low hit rate
    if live_conf_mean >= 0.65 and live_hit < live_conf_mean - 0.10:
        flags.append("OVERCONFIDENCE: Mean conf {:.1%} but hit rate only {:.1%}".format(
            live_conf_mean, live_hit
        ))
    
    # Spiky confidence: Low variance in predictions
    if live_conf_std < 0.05:
        flags.append("SPIKY_CONFIDENCE: Std={:.3f} - model not discriminating well".format(
            live_conf_std
        ))
    
    return flags

def run_dissonance_audit():
    """Run complete backtest dissonance analysis."""
    report = {
        'audit_timestamp': datetime.now().isoformat(),
        'backtest_period': f"<= {BACKTEST_END}",
        'live_period': f">= {LIVE_START}",
        'high_conf_threshold': HIGH_CONF_THRESHOLD,
        'markets': {},
        'summary': {
            'total_flags': 0,
            'data_shift_count': 0,
            'leakage_count': 0,
            'overconfidence_count': 0,
            'spiky_count': 0
        }
    }
    
    console.print("=" * 60, style="bold")
    console.print("BACKTEST DISSONANCE ANALYSIS", style="bold magenta")
    console.print("=" * 60, style="bold")
    console.print(f"Backtest period: <= {BACKTEST_END}")
    console.print(f"Live period: >= {LIVE_START}")
    console.print(f"High confidence threshold: {HIGH_CONF_THRESHOLD:.0%}\n")
    
    # Aggregate data across leagues
    all_backtest = []
    all_live = []
    live_predictions = defaultdict(list)  # market -> list of (conf, outcome)
    
    for league in LEAGUES:
        console.print(f"[cyan][{league}] Loading data...[/cyan]")
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            
            # Split into backtest and live
            df['date'] = pd.to_datetime(df['date'])
            df = df[
                (df['status'].str.upper().isin(FINISHED_STATUSES)) &
                (df['home_score'].notna()) &
                (df['away_score'].notna())
            ]
            
            df_backtest = df[df['date'] <= BACKTEST_END]
            df_live = df[df['date'] >= LIVE_START]
            
            console.print(f"  Backtest: {len(df_backtest)} matches | Live: {len(df_live)} matches")
            
            all_backtest.append(df_backtest)
            all_live.append(df_live)
            
            # Generate live predictions
            suite = _load_prediction_models(league)
            
            for _, match in track(df_live.iterrows(), total=len(df_live), description=f"[{league}] Predicting"):
                try:
                    probs, _ = _calculate_probabilities(match, suite, league)
                    
                    # Compute actual outcomes
                    total_goals = match['home_score'] + match['away_score']
                    actual_u25 = total_goals < 2.5
                    actual_o25 = total_goals >= 2.5
                    actual_btts = match['home_score'] > 0 and match['away_score'] > 0
                    
                    # Collect high-confidence predictions
                    if probs['u25'] >= HIGH_CONF_THRESHOLD:
                        live_predictions['goals_u25'].append((probs['u25'], actual_u25))
                    if probs['o25'] >= HIGH_CONF_THRESHOLD:
                        live_predictions['goals_o25'].append((probs['o25'], actual_o25))
                    if probs['btts'] >= HIGH_CONF_THRESHOLD:
                        live_predictions['btts_yes'].append((probs['btts'], actual_btts))
                    if probs['btts_no'] >= HIGH_CONF_THRESHOLD:
                        live_predictions['btts_no'].append((probs['btts_no'], not actual_btts))
                        
                except Exception:
                    continue
                    
        except Exception as e:
            console.print(f"[{league}] [red]Error: {e}[/red]")
    
    # Aggregate backtest and live data
    df_bt_all = pd.concat(all_backtest, ignore_index=True) if all_backtest else pd.DataFrame()
    df_live_all = pd.concat(all_live, ignore_index=True) if all_live else pd.DataFrame()
    
    # Calculate backtest hit rates
    backtest_rates = calculate_market_hit_rates(df_bt_all)
    live_rates = calculate_market_hit_rates(df_live_all)
    
    console.print(f"\n[bold]Backtest: {len(df_bt_all)} matches | Live: {len(df_live_all)} matches[/bold]\n")
    
    # Analyze each market
    table = Table(title="Market Dissonance Analysis")
    table.add_column("Market", style="cyan")
    table.add_column("BT Hit", justify="right")
    table.add_column("Live Hit", justify="right")
    table.add_column("HC Conf", justify="right")
    table.add_column("HC Hit", justify="right")
    table.add_column("Flags", style="yellow")
    
    for market in ['goals_u25', 'goals_o25', 'btts_yes', 'btts_no']:
        bt_data = backtest_rates.get(market, {})
        live_data = live_rates.get(market, {})
        hc_preds = live_predictions.get(market, [])
        
        bt_hit = bt_data.get('hit_rate', 0)
        live_hit = live_data.get('hit_rate', 0)
        
        # High-confidence prediction analysis
        hc_confs = [p[0] for p in hc_preds]
        hc_outcomes = [p[1] for p in hc_preds]
        hc_hit_rate = np.mean(hc_outcomes) if hc_outcomes else 0
        conf_dist = analyze_confidence_distribution(hc_confs)
        
        # Detect dissonance
        flags = detect_dissonance_pattern(
            bt_hit, live_hit,
            conf_dist.get('mean', 0),
            conf_dist.get('std', 0.1)
        )
        
        # Store in report
        report['markets'][market] = {
            'backtest_hit_rate': round(bt_hit, 4),
            'backtest_sample': bt_data.get('sample_size', 0),
            'live_hit_rate': round(live_hit, 4),
            'live_sample': live_data.get('sample_size', 0),
            'high_conf_count': len(hc_preds),
            'high_conf_hit_rate': round(hc_hit_rate, 4) if hc_preds else None,
            'confidence_distribution': conf_dist,
            'flags': flags
        }
        
        # Count flags
        report['summary']['total_flags'] += len(flags)
        for f in flags:
            if 'DATA_SHIFT' in f:
                report['summary']['data_shift_count'] += 1
            elif 'LEAKAGE' in f:
                report['summary']['leakage_count'] += 1
            elif 'OVERCONFIDENCE' in f:
                report['summary']['overconfidence_count'] += 1
            elif 'SPIKY' in f:
                report['summary']['spiky_count'] += 1
        
        # Add to table
        flag_str = " | ".join(flags[:1]) if flags else "[green]OK[/green]"
        table.add_row(
            market.upper(),
            f"{bt_hit:.1%}",
            f"{live_hit:.1%}",
            f"{conf_dist.get('mean', 0):.1%}" if hc_preds else "-",
            f"{hc_hit_rate:.1%}" if hc_preds else "-",
            flag_str
        )
    
    console.print(table)
    
    # Save report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    console.print(f"\n{'=' * 60}", style="bold")
    console.print("DISSONANCE SUMMARY", style="bold")
    console.print(f"{'=' * 60}", style="bold")
    console.print(f"Total Flags: {report['summary']['total_flags']}")
    console.print(f"  - Data Shift: {report['summary']['data_shift_count']}")
    console.print(f"  - Leakage: {report['summary']['leakage_count']}")
    console.print(f"  - Overconfidence: {report['summary']['overconfidence_count']}")
    console.print(f"  - Spiky Confidence: {report['summary']['spiky_count']}")
    console.print(f"\nReport saved to: {OUTPUT_FILE}")
    
    if report['summary']['total_flags'] > 0:
        console.print("\n[yellow bold]⚠️ DISSONANCE DETECTED - Review flags above[/yellow bold]")
    else:
        console.print("\n[green bold]✅ NO DISSONANCE DETECTED[/green bold]")
    
    return report

if __name__ == "__main__":
    run_dissonance_audit()

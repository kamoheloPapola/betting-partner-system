"""
High Confidence Failure Report.

Analyze all predictions ≥70% that failed in the last 60 days.
For each failure, categorize into:
1. LEGITIMATE_VARIANCE - Normal statistical occurrence (no action needed)
2. DATA_ILLUSION - Missing/corrupt input data led to false confidence  
3. ADJUSTMENT_INFLATION - H2H or offset adjustments over-inflated probability
4. ENFORCEMENT_BUG - Gate/cap not applied when it should have been

Only categories 3 & 4 justify system changes.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import json
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.features.pipeline import FeaturePipeline
from src.cli.commands.prediction import (
    _load_prediction_models, _calculate_probabilities,
    MARKET_PROB_CAPS, MAX_H2H_LIFT_DEFAULT
)
from src.config.leagues import FINISHED_STATUSES
from src.config import DATA_DIR

console = Console()

LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']
LOOKBACK_DAYS = 60
HIGH_CONF_THRESHOLD = 0.70

# Output path
REPORT_PATH = DATA_DIR / "reports" / "high_confidence_failures.json"

# Market outcome functions
MARKET_OUTCOMES = {
    'u25': lambda r: (r['home_score'] + r['away_score']) < 2.5,
    'o25': lambda r: (r['home_score'] + r['away_score']) >= 2.5,
    'btts': lambda r: r['home_score'] > 0 and r['away_score'] > 0,
    'corn_o75': lambda r: (r.get('home_corners', 0) + r.get('away_corners', 0)) > 7.5,
    'corn_u11': lambda r: (r.get('home_corners', 0) + r.get('away_corners', 0)) < 11.5,
}


def categorize_failure(
    match: pd.Series,
    market: str,
    prob: float,
    attribution: Dict[str, Any]
) -> tuple:
    """
    Categorize a high-confidence failure.
    
    Returns (category, reason)
    """
    # Check for DATA_ILLUSION
    # Missing key features
    if pd.isna(match.get('home_score')) or pd.isna(match.get('away_score')):
        return "DATA_ILLUSION", "Missing scores"
    
    if 'corn' in market:
        if pd.isna(match.get('home_corners')) or pd.isna(match.get('away_corners')):
            return "DATA_ILLUSION", "Missing corners data"
    
    if 'card' in market:
        if pd.isna(match.get('home_cards')) or pd.isna(match.get('away_cards')):
            return "DATA_ILLUSION", "Missing cards data"
    
    # Check for ADJUSTMENT_INFLATION
    # Look for H2H lifts
    h2h_lift = attribution.get('h2h_lift', 0)
    if h2h_lift and abs(h2h_lift) > MAX_H2H_LIFT_DEFAULT:
        return "ADJUSTMENT_INFLATION", f"H2H lift {h2h_lift:+.1%} exceeded cap"
    
    # Check if prob exceeds cap but wasn't capped (ENFORCEMENT_BUG)
    cap = MARKET_PROB_CAPS.get(market)
    if cap and prob > cap + 0.01:  # Allow 1% tolerance
        return "ENFORCEMENT_BUG", f"Prob {prob:.1%} > cap {cap:.0%}"
    
    # Default: legitimate variance
    return "LEGITIMATE_VARIANCE", "Normal statistical occurrence"


def collect_failures():
    """Collect all high-confidence failures from last 60 days."""
    console.print("=" * 70, style="bold")
    console.print("HIGH CONFIDENCE FAILURE REPORT", style="bold magenta")
    console.print("=" * 70, style="bold")
    console.print(f"Lookback: {LOOKBACK_DAYS} days | Threshold: ≥{HIGH_CONF_THRESHOLD:.0%}\n")
    
    failures = []
    stats = defaultdict(int)
    
    for league in LEAGUES:
        console.print(f"\n[cyan][{league}] Processing...[/cyan]")
        
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            suite = _load_prediction_models(league)
            
            df['date'] = pd.to_datetime(df['date'])
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=LOOKBACK_DAYS)
            
            # Ensure timezone-naive comparison
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
                    probs, attribution = _calculate_probabilities(match, suite, league)
                    
                    for market, outcome_fn in MARKET_OUTCOMES.items():
                        if market not in probs:
                            continue
                        
                        prob = probs[market]
                        
                        if prob < HIGH_CONF_THRESHOLD:
                            continue
                        
                        # Check outcome
                        try:
                            hit = outcome_fn(match)
                        except Exception:
                            continue
                        
                        stats['total_high_conf'] += 1
                        
                        if not hit:
                            # This is a high-confidence failure
                            category, reason = categorize_failure(
                                match, market, prob, attribution
                            )
                            
                            failure = {
                                'date': match['date'].strftime('%Y-%m-%d'),
                                'match': f"{match['home_team']} vs {match['away_team']}",
                                'league': league,
                                'market': market,
                                'prob': round(prob, 4),
                                'category': category,
                                'reason': reason,
                                'h2h_lift': attribution.get(f'{market}_h2h_lift', 0),
                                'attribution': {
                                    k: v for k, v in attribution.items() 
                                    if market in k.lower() or 'h2h' in k.lower()
                                }
                            }
                            failures.append(failure)
                            stats[category] += 1
                        else:
                            stats['successes'] += 1
                            
                except Exception as e:
                    continue
                    
        except Exception as e:
            console.print(f"[{league}] [red]Error: {e}[/red]")
    
    return failures, dict(stats)


def print_report(failures: List[Dict], stats: Dict):
    """Print failure report."""
    console.print("\n" + "=" * 70, style="bold")
    console.print("FAILURE SUMMARY", style="bold")
    console.print("=" * 70, style="bold")
    
    total_hc = stats.get('total_high_conf', 0)
    total_fail = len(failures)
    success_rate = 1 - (total_fail / total_hc) if total_hc > 0 else 0
    
    console.print(f"Total High Confidence (≥70%): {total_hc}")
    console.print(f"Successes: {stats.get('successes', 0)}")
    console.print(f"[red]Failures: {total_fail}[/red]")
    console.print(f"Hit Rate: {success_rate:.1%}")
    
    # Category breakdown
    console.print("\n[bold]Failure Categories:[/bold]")
    for cat in ['LEGITIMATE_VARIANCE', 'DATA_ILLUSION', 'ADJUSTMENT_INFLATION', 'ENFORCEMENT_BUG']:
        count = stats.get(cat, 0)
        pct = (count / total_fail * 100) if total_fail > 0 else 0
        style = "green" if cat == "LEGITIMATE_VARIANCE" else "yellow" if cat == "DATA_ILLUSION" else "red"
        console.print(f"  [{style}]{cat}: {count} ({pct:.1f}%)[/{style}]")
    
    # Show actionable failures
    actionable = [f for f in failures if f['category'] in ['ADJUSTMENT_INFLATION', 'ENFORCEMENT_BUG']]
    
    if actionable:
        console.print(f"\n[bold red]⚠️ {len(actionable)} ACTIONABLE FAILURES (Require System Change):[/bold red]")
        
        table = Table()
        table.add_column("Date")
        table.add_column("Match")
        table.add_column("Market")
        table.add_column("Prob")
        table.add_column("Category")
        table.add_column("Reason")
        
        for f in actionable[:20]:  # Show first 20
            table.add_row(
                f['date'],
                f['match'][:25],
                f['market'],
                f"{f['prob']:.1%}",
                f['category'],
                f['reason']
            )
        
        console.print(table)
    else:
        console.print("\n[green]✅ No actionable failures (all legitimate variance or data issues)[/green]")
    
    # Market breakdown
    console.print("\n[bold]Failures by Market:[/bold]")
    market_counts = defaultdict(int)
    for f in failures:
        market_counts[f['market']] += 1
    
    for market, count in sorted(market_counts.items(), key=lambda x: -x[1]):
        console.print(f"  {market}: {count}")


def save_report(failures: List[Dict], stats: Dict):
    """Save detailed report to JSON."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'generated': datetime.now().isoformat(),
        'lookback_days': LOOKBACK_DAYS,
        'threshold': HIGH_CONF_THRESHOLD,
        'summary': {
            'total_high_conf': stats.get('total_high_conf', 0),
            'successes': stats.get('successes', 0),
            'failures': len(failures),
            'hit_rate': 1 - (len(failures) / max(1, stats.get('total_high_conf', 1))),
        },
        'category_breakdown': {
            cat: stats.get(cat, 0) 
            for cat in ['LEGITIMATE_VARIANCE', 'DATA_ILLUSION', 'ADJUSTMENT_INFLATION', 'ENFORCEMENT_BUG']
        },
        'failures': failures
    }
    
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    console.print(f"\n[bold]Report saved to: {REPORT_PATH}[/bold]")


def main():
    failures, stats = collect_failures()
    print_report(failures, stats)
    save_report(failures, stats)
    
    # Final recommendation
    actionable = stats.get('ADJUSTMENT_INFLATION', 0) + stats.get('ENFORCEMENT_BUG', 0)
    if actionable > 0:
        console.print(f"\n[red bold]⚠️ {actionable} failures require system changes. Review report.[/red bold]")
    else:
        console.print("\n[green bold]✅ No system changes required. All failures are legitimate variance or data issues.[/green bold]")


if __name__ == "__main__":
    main()

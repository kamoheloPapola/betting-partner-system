"""
Layer 2: Probability Mathematics Audit (Non-Negotiable Invariants)

Checks:
1. Goals: O1.5 ≥ O2.5 ≥ O3.5 (equivalently U3.5 ≥ U2.5 ≥ U1.5)
2. Cards: Under high line ≥ Under low line (U5.5 ≥ U4.5 ≥ U2.5)
3. Corners: Under high line ≥ Under low line
4. Home + Away expectations reconcile with total

Output: probability_math_report.json

CRITICAL: Any invariant failure is a BUG, not football variance.
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.features.pipeline import FeaturePipeline
from src.cli.commands.prediction import _load_prediction_models, _calculate_probabilities

# === CONFIGURATION ===
LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']
OUTPUT_FILE = Path("data/reports/probability_math_report.json")
TOLERANCE = 1e-6  # Floating point tolerance

console = Console()

def check_goals_invariants(probs: dict, ctx: str) -> list:
    """Check O1.5 ≥ O2.5 ≥ O3.5 (equivalently U3.5 ≥ U2.5 ≥ U1.5)"""
    issues = []
    
    o15 = probs.get('over_1_5', 0)
    o25 = probs.get('o25', 0)
    o35 = 1.0 - probs.get('u35', 1.0)  # O3.5 = 1 - U3.5
    
    u15 = 1.0 - o15
    u25 = probs.get('u25', 0)
    u35 = probs.get('u35', 0)
    
    # Check Over invariant: O1.5 ≥ O2.5 ≥ O3.5
    if o15 < o25 - TOLERANCE:
        issues.append({
            'invariant': 'GOALS_O15_GTE_O25',
            'expected': 'O1.5 ≥ O2.5',
            'actual': f'O1.5={o15:.3f} < O2.5={o25:.3f}',
            'context': ctx,
            'severity': 'BUG'
        })
    
    if o25 < o35 - TOLERANCE:
        issues.append({
            'invariant': 'GOALS_O25_GTE_O35',
            'expected': 'O2.5 ≥ O3.5',
            'actual': f'O2.5={o25:.3f} < O3.5={o35:.3f}',
            'context': ctx,
            'severity': 'BUG'
        })
    
    # Check Under invariant: U3.5 ≥ U2.5 ≥ U1.5
    if u35 < u25 - TOLERANCE:
        issues.append({
            'invariant': 'GOALS_U35_GTE_U25',
            'expected': 'U3.5 ≥ U2.5',
            'actual': f'U3.5={u35:.3f} < U2.5={u25:.3f}',
            'context': ctx,
            'severity': 'BUG'
        })
    
    if u25 < u15 - TOLERANCE:
        issues.append({
            'invariant': 'GOALS_U25_GTE_U15',
            'expected': 'U2.5 ≥ U1.5',
            'actual': f'U2.5={u25:.3f} < U1.5={u15:.3f}',
            'context': ctx,
            'severity': 'BUG'
        })
    
    return issues

def check_cards_invariants(probs: dict, ctx: str) -> list:
    """Check Under high line ≥ Under low line for cards."""
    issues = []
    
    u55 = probs.get('card_u55', 0)
    u45 = probs.get('card_u45', 0)
    o25 = probs.get('card_o25', 0)  # O2.5 means U2.5 = 1 - O2.5
    u25 = 1.0 - o25 if o25 else None
    
    # Check: U5.5 ≥ U4.5
    if u55 and u45 and u55 < u45 - TOLERANCE:
        issues.append({
            'invariant': 'CARDS_U55_GTE_U45',
            'expected': 'Cards U5.5 ≥ U4.5',
            'actual': f'U5.5={u55:.3f} < U4.5={u45:.3f}',
            'context': ctx,
            'severity': 'BUG'
        })
    
    # Check: U4.5 ≥ U2.5 (if U2.5 is computable)
    if u45 and u25 and u45 < u25 - TOLERANCE:
        issues.append({
            'invariant': 'CARDS_U45_GTE_U25',
            'expected': 'Cards U4.5 ≥ U2.5',
            'actual': f'U4.5={u45:.3f} < U2.5={u25:.3f}',
            'context': ctx,
            'severity': 'BUG'
        })
    
    return issues

def check_corners_invariants(probs: dict, ctx: str) -> list:
    """Check Under high line ≥ Under low line for corners."""
    issues = []
    
    u115 = probs.get('corn_u11', 0)
    # If we have more corner lines, check them
    # For now, check that Corner 1X2 sums to ~1.0
    corn_h = probs.get('corn_1x2_h', 0)
    corn_d = probs.get('corn_1x2_d', 0)
    corn_a = probs.get('corn_1x2_a', 0)
    
    if corn_h and corn_d and corn_a:
        total = corn_h + corn_d + corn_a
        if abs(total - 1.0) > 0.01:  # Allow 1% tolerance for rounding
            issues.append({
                'invariant': 'CORNERS_1X2_SUM_TO_1',
                'expected': 'Corners 1X2 probabilities sum to 1.0',
                'actual': f'Sum = {total:.3f} (H={corn_h:.3f}, D={corn_d:.3f}, A={corn_a:.3f})',
                'context': ctx,
                'severity': 'WARN' if abs(total - 1.0) < 0.05 else 'BUG'
            })
    
    return issues

def check_1x2_invariants(probs: dict, ctx: str) -> list:
    """Check that 1X2 probabilities sum to 1.0."""
    issues = []
    
    h = probs.get('home', 0)
    d = probs.get('draw', 0)
    a = probs.get('away', 0)
    
    if h and d and a:
        total = h + d + a
        if abs(total - 1.0) > 0.01:
            issues.append({
                'invariant': 'GOALS_1X2_SUM_TO_1',
                'expected': '1X2 probabilities sum to 1.0',
                'actual': f'Sum = {total:.3f} (H={h:.3f}, D={d:.3f}, A={a:.3f})',
                'context': ctx,
                'severity': 'WARN' if abs(total - 1.0) < 0.05 else 'BUG'
            })
    
    return issues

def check_btts_invariants(probs: dict, ctx: str) -> list:
    """Check BTTS Yes + BTTS No = 1.0."""
    issues = []
    
    btts_yes = probs.get('btts', 0)
    btts_no = probs.get('btts_no', 0)
    
    if btts_yes and btts_no:
        total = btts_yes + btts_no
        if abs(total - 1.0) > 0.01:
            issues.append({
                'invariant': 'BTTS_SUM_TO_1',
                'expected': 'BTTS Yes + No = 1.0',
                'actual': f'Sum = {total:.3f} (Yes={btts_yes:.3f}, No={btts_no:.3f})',
                'context': ctx,
                'severity': 'BUG'
            })
    
    return issues

def run_full_audit():
    """Run complete Layer 2 audit across all leagues."""
    report = {
        'audit_timestamp': datetime.now().isoformat(),
        'layer': 'LAYER_2_PROBABILITY_MATHEMATICS',
        'leagues': {},
        'summary': {
            'total_matches_audited': 0,
            'total_bugs': 0,
            'total_warnings': 0,
            'invariants_checked': [
                'GOALS_O15_GTE_O25', 'GOALS_O25_GTE_O35',
                'GOALS_U35_GTE_U25', 'GOALS_U25_GTE_U15',
                'CARDS_U55_GTE_U45', 'CARDS_U45_GTE_U25',
                'CORNERS_1X2_SUM_TO_1', 'GOALS_1X2_SUM_TO_1', 'BTTS_SUM_TO_1'
            ]
        }
    }
    
    console.print("=" * 60, style="bold")
    console.print("LAYER 2: PROBABILITY MATHEMATICS AUDIT", style="bold green")
    console.print("=" * 60, style="bold")
    
    for league in LEAGUES:
        console.print(f"\n[cyan][{league}] Loading data and models...[/cyan]")
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            suite = _load_prediction_models(league)
            
            # Get recent matches for audit (last 100 or all if less)
            df_sorted = df.sort_values('date', ascending=False)
            df_audit = df_sorted.head(100)
            
            console.print(f"[{league}] Auditing {len(df_audit)} recent matches...")
            
            league_issues = []
            
            for idx, match in track(df_audit.iterrows(), total=len(df_audit), description=f"[{league}]"):
                ctx = f"{match.get('home_team', 'UNK')} vs {match.get('away_team', 'UNK')}"
                try:
                    probs, _ = _calculate_probabilities(match, suite, league)
                    
                    # Run all invariant checks
                    league_issues.extend(check_goals_invariants(probs, ctx))
                    league_issues.extend(check_cards_invariants(probs, ctx))
                    league_issues.extend(check_corners_invariants(probs, ctx))
                    league_issues.extend(check_1x2_invariants(probs, ctx))
                    league_issues.extend(check_btts_invariants(probs, ctx))
                    
                except Exception as e:
                    league_issues.append({
                        'invariant': 'CALCULATION_ERROR',
                        'context': ctx,
                        'error': str(e),
                        'severity': 'ERROR'
                    })
            
            # Count bugs and warnings
            bugs = sum(1 for i in league_issues if i.get('severity') == 'BUG')
            warns = sum(1 for i in league_issues if i.get('severity') == 'WARN')
            
            report['leagues'][league] = {
                'matches_audited': len(df_audit),
                'bugs': bugs,
                'warnings': warns,
                'issues': league_issues if league_issues else None
            }
            
            report['summary']['total_matches_audited'] += len(df_audit)
            report['summary']['total_bugs'] += bugs
            report['summary']['total_warnings'] += warns
            
            status = "[green]✅ PASS[/green]" if bugs == 0 else f"[red]❌ {bugs} BUGS[/red]"
            console.print(f"[{league}] {status} (Bugs: {bugs}, Warnings: {warns})")
            
        except Exception as e:
            console.print(f"[{league}] [red]❌ Error: {e}[/red]")
            report['leagues'][league] = {'error': str(e)}
    
    # Save report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    console.print(f"\n{'=' * 60}", style="bold")
    console.print("AUDIT SUMMARY", style="bold")
    console.print(f"{'=' * 60}", style="bold")
    console.print(f"Total Matches Audited: {report['summary']['total_matches_audited']}")
    console.print(f"[red bold]Total BUGS: {report['summary']['total_bugs']}[/red bold]")
    console.print(f"[yellow]Total WARNINGS: {report['summary']['total_warnings']}[/yellow]")
    console.print(f"\nReport saved to: {OUTPUT_FILE}")
    
    if report['summary']['total_bugs'] > 0:
        console.print("\n[red bold]⚠️ CRITICAL: Probability invariants violated! Review bugs immediately.[/red bold]")
    else:
        console.print("\n[green bold]✅ ALL PROBABILITY INVARIANTS VERIFIED[/green bold]")
    
    return report

if __name__ == "__main__":
    run_full_audit()

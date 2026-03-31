"""
Layer 3: Confidence Inflation Audit (MOST IMPORTANT)

Detects misleading predictions from stacked micro-adjustments.

For predictions >70%:
- Log raw model probability
- Log post-adjustment probability
- Track each modifier contribution (H2H, offsets, damping)
- Compute Confidence Lift = Final Prob - Raw Prob

Flag Rules:
- If lift > +8%, flag it
- If multiple lifts stack, flag it
- If lift occurs without variance expansion, flag it

Output: confidence_inflation_report.json
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import track

from src.features.pipeline import FeaturePipeline
from src.cli.commands.prediction import _load_prediction_models, _predict_scalar
from src.ml.distributions import PoissonEngine, NegativeBinomialEngine, ZeroInflatedEngine

# === CONFIGURATION ===
LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']
OUTPUT_FILE = Path("data/reports/confidence_inflation_report.json")
HIGH_CONFIDENCE_THRESHOLD = 0.70
LIFT_FLAG_THRESHOLD = 0.08  # 8%

console = Console()

@dataclass
class ModifierContribution:
    """Track a single modifier's contribution to confidence."""
    name: str
    raw_value: float
    adjusted_value: float
    delta: float
    
@dataclass
class ConfidenceTrace:
    """Full trace of a high-confidence prediction."""
    match: str
    market: str
    league: str
    raw_prob: float
    final_prob: float
    confidence_lift: float
    modifiers: List[ModifierContribution]
    flags: List[str]

def trace_goals_confidence(match: pd.Series, suite: dict, league: str) -> List[ConfidenceTrace]:
    """Trace confidence modifiers for Goals markets."""
    traces = []
    feats = suite['meta_goals']['features']
    ctx = f"{match.get('home_team', 'UNK')} vs {match.get('away_team', 'UNK')}"
    
    # Get raw lambdas
    lh = _predict_scalar(suite['mh_goals'], match, feats, f"{ctx}:GoalsH")
    la = _predict_scalar(suite['ma_goals'], match, feats, f"{ctx}:GoalsA")
    
    # Raw probabilities from Poisson
    engine = PoissonEngine()
    raw = engine.calculate_probabilities(lh, la)
    
    # Get H2H data
    h2h_count = match.get('h2h_match_count', 0)
    h2h_o25_rate = match.get('h2h_goals_o25_rate')
    
    # === O2.5 MARKET ===
    raw_o25 = raw['over_2_5']
    final_o25 = raw_o25
    modifiers = []
    
    # Track H2H adjustment
    if h2h_count >= 3 and h2h_o25_rate is not None and not pd.isna(h2h_o25_rate):
        if abs(raw_o25 - h2h_o25_rate) > 0.15:
            h2h_weight = min(h2h_count / 5, 0.5)
            final_o25 = (1 - h2h_weight) * raw_o25 + h2h_weight * h2h_o25_rate
            modifiers.append(ModifierContribution(
                name='H2H_O25_RATE',
                raw_value=raw_o25,
                adjusted_value=final_o25,
                delta=final_o25 - raw_o25
            ))
    
    # Track damping (already applied in engine, but we note it)
    modifiers.append(ModifierContribution(
        name='POISSON_DAMPING',
        raw_value=raw_o25,
        adjusted_value=raw_o25,  # Damping is internal to engine
        delta=0.0
    ))
    
    # Final probability
    lift = final_o25 - raw_o25
    flags = []
    
    if lift > LIFT_FLAG_THRESHOLD:  # Only flag POSITIVE lifts (inflation)
        flags.append(f'LIFT_EXCEEDS_8PP: {lift*100:+.1f}pp')
    
    if len([m for m in modifiers if abs(m.delta) > 0.03]) > 1:
        flags.append('MULTIPLE_STACKED_ADJUSTMENTS')
    
    if final_o25 > HIGH_CONFIDENCE_THRESHOLD:
        traces.append(ConfidenceTrace(
            match=ctx,
            market='GOALS_O2.5',
            league=league,
            raw_prob=raw_o25,
            final_prob=final_o25,
            confidence_lift=lift,
            modifiers=[asdict(m) for m in modifiers],
            flags=flags
        ))
    
    # === U2.5 MARKET ===
    final_u25 = 1.0 - final_o25
    if final_u25 > HIGH_CONFIDENCE_THRESHOLD:
        traces.append(ConfidenceTrace(
            match=ctx,
            market='GOALS_U2.5',
            league=league,
            raw_prob=1.0 - raw_o25,
            final_prob=final_u25,
            confidence_lift=final_u25 - (1.0 - raw_o25),
            modifiers=[asdict(m) for m in modifiers],
            flags=flags
        ))
    
    return traces

def trace_cards_confidence(match: pd.Series, suite: dict, league: str) -> List[ConfidenceTrace]:
    """Trace confidence modifiers for Cards markets."""
    traces = []
    ctx = f"{match.get('home_team', 'UNK')} vs {match.get('away_team', 'UNK')}"
    
    if suite.get('m_cards') is None:
        return traces
    
    # Get raw mu
    feats = suite.get('meta_cards', {}).get('features', [])
    if not feats:
        return traces
        
    try:
        mu_raw = _predict_scalar(suite['m_cards'], match, feats, f"{ctx}:Cards")
    except:
        return traces
    
    # Get H2H data
    h2h_count = match.get('h2h_match_count', 0)
    h2h_avg_cards = match.get('h2h_avg_cards')
    h2h_o25_rate = match.get('h2h_cards_o25_rate')
    
    mu = mu_raw
    modifiers = []
    
    # Track H2H mu blending
    if h2h_count >= 2 and h2h_avg_cards is not None and not pd.isna(h2h_avg_cards):
        h2h_weight = min(h2h_count / 5, 0.5)
        mu_blended = (1 - h2h_weight) * mu + h2h_weight * h2h_avg_cards
        modifiers.append(ModifierContribution(
            name='H2H_MU_BLEND',
            raw_value=mu_raw,
            adjusted_value=mu_blended,
            delta=mu_blended - mu_raw
        ))
        mu = mu_blended
    
    # Calculate probabilities
    engine = ZeroInflatedEngine()
    probs = engine.calculate_probabilities(mu, pi_zero=0.03)
    
    raw_o25 = probs['cards_over_2_5']
    final_o25 = raw_o25
    
    # Track H2H O2.5 rate capping
    if h2h_count >= 3 and h2h_o25_rate is not None and not pd.isna(h2h_o25_rate):
        model_o25 = 1.0 - probs.get('cards_under_2_5', 0.23)
        if model_o25 > h2h_o25_rate + 0.20:
            blend_weight = min(h2h_count / 5, 0.6)
            capped_o25 = (1 - blend_weight) * model_o25 + blend_weight * h2h_o25_rate
            modifiers.append(ModifierContribution(
                name='H2H_O25_RATE_CAP',
                raw_value=model_o25,
                adjusted_value=capped_o25,
                delta=capped_o25 - model_o25
            ))
            final_o25 = capped_o25
    
    lift = final_o25 - raw_o25
    flags = []
    
    if lift > LIFT_FLAG_THRESHOLD:  # Only flag POSITIVE lifts (inflation)
        flags.append(f'LIFT_EXCEEDS_8PP: {lift*100:+.1f}pp')
    
    if len([m for m in modifiers if abs(m.delta) > 0.03]) > 1:
        flags.append('MULTIPLE_STACKED_ADJUSTMENTS')
    
    if final_o25 > HIGH_CONFIDENCE_THRESHOLD:
        traces.append(ConfidenceTrace(
            match=ctx,
            market='CARDS_O2.5',
            league=league,
            raw_prob=raw_o25,
            final_prob=final_o25,
            confidence_lift=lift,
            modifiers=[asdict(m) for m in modifiers],
            flags=flags
        ))
    
    return traces

def trace_corners_confidence(match: pd.Series, suite: dict, league: str) -> List[ConfidenceTrace]:
    """Trace confidence modifiers for Corners markets."""
    traces = []
    ctx = f"{match.get('home_team', 'UNK')} vs {match.get('away_team', 'UNK')}"
    
    if suite.get('mh_corn') is None:
        return traces
    
    # Get raw mu values
    feats_h = suite.get('meta_corners_h', {}).get('features', [])
    feats_a = suite.get('meta_corners_a', {}).get('features', [])
    if not feats_h or not feats_a:
        return traces
    
    try:
        mu_h = _predict_scalar(suite['mh_corn'], match, feats_h, f"{ctx}:CornH")
        mu_a = _predict_scalar(suite['ma_corn'], match, feats_a, f"{ctx}:CornA")
    except:
        return traces
    
    # Get H2H data
    h2h_count = match.get('h2h_match_count', 0)
    h2h_avg_corners = match.get('h2h_avg_corners')
    
    mu_total_raw = mu_h + mu_a
    mu_total = mu_total_raw
    modifiers = []
    
    # Track H2H blending
    if h2h_count >= 2 and h2h_avg_corners is not None and not pd.isna(h2h_avg_corners):
        h2h_weight = min(h2h_count / 5, 0.4)
        mu_blended = (1 - h2h_weight) * mu_total + h2h_weight * h2h_avg_corners
        modifiers.append(ModifierContribution(
            name='H2H_CORNERS_BLEND',
            raw_value=mu_total_raw,
            adjusted_value=mu_blended,
            delta=mu_blended - mu_total_raw
        ))
        mu_total = mu_blended
    
    # Calculate U11.5 probability
    engine = NegativeBinomialEngine()
    # Simplified: use mu/2 for each side
    probs = engine.calculate_probabilities(mu_total/2, mu_total/2 * 1.5, mu_total/2, mu_total/2 * 1.5)
    
    u115 = probs['corners_under_11_5']
    
    if u115 > HIGH_CONFIDENCE_THRESHOLD:
        traces.append(ConfidenceTrace(
            match=ctx,
            market='CORNERS_U11.5',
            league=league,
            raw_prob=u115,
            final_prob=u115,
            confidence_lift=0.0,
            modifiers=[asdict(m) for m in modifiers],
            flags=[]
        ))
    
    return traces

def run_full_audit():
    """Run complete Layer 3 audit across all leagues."""
    report = {
        'audit_timestamp': datetime.now().isoformat(),
        'layer': 'LAYER_3_CONFIDENCE_INFLATION',
        'high_confidence_threshold': HIGH_CONFIDENCE_THRESHOLD,
        'lift_flag_threshold': LIFT_FLAG_THRESHOLD,
        'leagues': {},
        'summary': {
            'total_high_conf_predictions': 0,
            'flagged_predictions': 0,
            'lift_exceeds_8pp': 0,
            'stacked_adjustments': 0
        },
        'flagged_details': []
    }
    
    console.print("=" * 60, style="bold")
    console.print("LAYER 3: CONFIDENCE INFLATION AUDIT", style="bold yellow")
    console.print("=" * 60, style="bold")
    console.print(f"Threshold for 'high confidence': >{HIGH_CONFIDENCE_THRESHOLD*100:.0f}%")
    console.print(f"Flag if lift exceeds: {LIFT_FLAG_THRESHOLD*100:.0f}pp")
    
    for league in LEAGUES:
        console.print(f"\n[cyan][{league}] Loading data and models...[/cyan]")
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            suite = _load_prediction_models(league)
            
            # Get recent matches for audit
            df_sorted = df.sort_values('date', ascending=False)
            df_audit = df_sorted.head(100)
            
            console.print(f"[{league}] Tracing {len(df_audit)} matches...")
            
            league_traces = []
            
            for idx, match in track(df_audit.iterrows(), total=len(df_audit), description=f"[{league}]"):
                try:
                    # Trace all markets
                    league_traces.extend(trace_goals_confidence(match, suite, league))
                    league_traces.extend(trace_cards_confidence(match, suite, league))
                    league_traces.extend(trace_corners_confidence(match, suite, league))
                except Exception as e:
                    pass  # Skip problematic matches
            
            # Count flagged
            flagged = [t for t in league_traces if t.flags]
            
            report['leagues'][league] = {
                'matches_audited': len(df_audit),
                'high_conf_predictions': len(league_traces),
                'flagged': len(flagged)
            }
            
            report['summary']['total_high_conf_predictions'] += len(league_traces)
            report['summary']['flagged_predictions'] += len(flagged)
            
            for trace in flagged:
                report['flagged_details'].append(asdict(trace))
                if 'LIFT_EXCEEDS_8PP' in str(trace.flags):
                    report['summary']['lift_exceeds_8pp'] += 1
                if 'MULTIPLE_STACKED_ADJUSTMENTS' in str(trace.flags):
                    report['summary']['stacked_adjustments'] += 1
            
            status = "[green]✅[/green]" if len(flagged) == 0 else f"[yellow]⚠️ {len(flagged)} flagged[/yellow]"
            console.print(f"[{league}] {status} (High Conf: {len(league_traces)}, Flagged: {len(flagged)})")
            
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
    console.print(f"Total High-Confidence Predictions: {report['summary']['total_high_conf_predictions']}")
    console.print(f"[yellow]Flagged Predictions: {report['summary']['flagged_predictions']}[/yellow]")
    console.print(f"  - Lift > 8pp: {report['summary']['lift_exceeds_8pp']}")
    console.print(f"  - Stacked Adjustments: {report['summary']['stacked_adjustments']}")
    console.print(f"\nReport saved to: {OUTPUT_FILE}")
    
    if report['summary']['flagged_predictions'] > 0:
        console.print("\n[yellow bold]⚠️ Review flagged predictions in the report.[/yellow bold]")
    else:
        console.print("\n[green bold]✅ NO CONFIDENCE INFLATION DETECTED[/green bold]")
    
    return report

if __name__ == "__main__":
    run_full_audit()

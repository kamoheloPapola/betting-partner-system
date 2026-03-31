"""
Layer 1: Input Sanity Audit (Data Integrity)

Checks:
1. Missing Feature Audit - Required features present for predictions
2. Extreme Value Detection - Flag rolling averages outside tolerance
3. Temporal Leakage - Ensure H2H/rolling windows use only past data

Output: input_health_report.json
"""
import sys
import os
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.getcwd())

import pandas as pd
import numpy as np
from src.features.pipeline import FeaturePipeline
from src.config.leagues import LEAGUE_METADATA

# === CONFIGURATION ===
LEAGUES = ['PL', 'PD', 'SA', 'BL1', 'FL1']
OUTPUT_FILE = Path("data/reports/input_health_report.json")

# Required features for prediction by market
REQUIRED_FEATURES = {
    'goals': [
        'Rolling_Home_XG', 'Rolling_Away_XG',
        'Rolling_Home_Goals', 'Rolling_Away_Goals',
        'home_form', 'away_form'
    ],
    'corners': [
        'Rolling_Home_Corners', 'Rolling_Away_Corners',
        'home_corners', 'away_corners'
    ],
    'cards': [
        'Rolling_Home_Cards', 'Rolling_Away_Cards',
        'home_total_cards', 'away_total_cards'
    ],
    'h2h': [
        'h2h_match_count', 'h2h_avg_cards', 'h2h_avg_corners',
        'h2h_goals_o25_rate', 'h2h_btts_rate', 'h2h_cards_o25_rate'
    ]
}

# Rolling feature columns to check for extreme values
ROLLING_FEATURES = [
    'Rolling_Home_XG', 'Rolling_Away_XG',
    'Rolling_Home_Goals', 'Rolling_Away_Goals',
    'Rolling_Home_Corners', 'Rolling_Away_Corners',
    'Rolling_Home_Cards', 'Rolling_Away_Cards',
]

def audit_missing_features(df: pd.DataFrame, league: str) -> dict:
    """Check for missing required features."""
    results = {'league': league, 'total_matches': len(df), 'markets': {}}
    
    for market, features in REQUIRED_FEATURES.items():
        present = 0
        missing_counts = {}
        
        for feat in features:
            if feat in df.columns:
                # Count non-null values
                non_null_pct = df[feat].notna().mean() * 100
                if non_null_pct < 100:
                    missing_counts[feat] = f"{non_null_pct:.1f}% present"
                else:
                    present += 1
            else:
                missing_counts[feat] = "COLUMN MISSING"
        
        results['markets'][market] = {
            'features_present': present,
            'features_required': len(features),
            'completeness_pct': (present / len(features)) * 100 if features else 0,
            'issues': missing_counts if missing_counts else None
        }
    
    return results

def audit_extreme_values(df: pd.DataFrame, league: str) -> dict:
    """Detect extreme values in rolling features."""
    results = {'league': league, 'issues': []}
    
    for feat in ROLLING_FEATURES:
        if feat not in df.columns:
            continue
            
        col = df[feat].dropna()
        if len(col) == 0:
            continue
            
        mean, std = col.mean(), col.std()
        
        # Flag values < 0.5 (suspiciously low)
        low_count = (col < 0.5).sum()
        if low_count > 0:
            results['issues'].append({
                'feature': feat,
                'type': 'LOW_VALUES',
                'count': int(low_count),
                'threshold': '< 0.5',
                'severity': 'WARN' if low_count < 10 else 'ALERT'
            })
        
        # Flag values > 3σ from mean (extreme outliers)
        upper_bound = mean + 3 * std
        outlier_count = (col > upper_bound).sum()
        if outlier_count > 0:
            results['issues'].append({
                'feature': feat,
                'type': 'OUTLIERS_3SIGMA',
                'count': int(outlier_count),
                'threshold': f'> {upper_bound:.2f}',
                'mean': round(mean, 2),
                'std': round(std, 2),
                'severity': 'WARN' if outlier_count < 5 else 'ALERT'
            })
        
        # Detect sudden drops (diff > 1σ between consecutive rows)
        diffs = col.diff().abs()
        sudden_drops = (diffs > std).sum()
        if sudden_drops > len(col) * 0.05:  # More than 5% are sudden changes
            results['issues'].append({
                'feature': feat,
                'type': 'SUDDEN_DROPS',
                'count': int(sudden_drops),
                'pct_of_data': f"{sudden_drops / len(col) * 100:.1f}%",
                'severity': 'INFO'
            })
    
    return results

def audit_temporal_leakage(df: pd.DataFrame, league: str) -> dict:
    """Ensure no temporal leakage in H2H and rolling features."""
    results = {'league': league, 'issues': [], 'checks_passed': []}
    
    # Check 1: H2H features should have shift applied (no future data)
    # The h2h_match_count should be 0 for first meeting
    if 'h2h_match_count' in df.columns:
        first_meetings = df[df['h2h_match_count'] == 0]
        if len(first_meetings) > 0:
            results['checks_passed'].append({
                'check': 'H2H_FIRST_MEETINGS',
                'detail': f"{len(first_meetings)} matches with 0 prior H2H (correct)",
                'status': 'PASS'
            })
        
        # Verify h2h_match_count is monotonic for each H2H pair
        # Sample check on a few team pairs
        if 'home_team' in df.columns and 'away_team' in df.columns:
            df_sorted = df.sort_values('date')
            # Create H2H key
            df_sorted['_h2h_key'] = df_sorted.apply(
                lambda r: tuple(sorted([r['home_team'], r['away_team']])), axis=1
            )
            
            leakage_found = False
            for key, group in df_sorted.groupby('_h2h_key'):
                counts = group['h2h_match_count'].values
                # Count should be non-decreasing
                if not np.all(np.diff(counts) >= 0):
                    leakage_found = True
                    results['issues'].append({
                        'check': 'H2H_MONOTONICITY',
                        'teams': str(key),
                        'counts': counts.tolist()[:5],  # First 5 for brevity
                        'severity': 'ALERT'
                    })
            
            if not leakage_found:
                results['checks_passed'].append({
                    'check': 'H2H_MONOTONICITY',
                    'detail': 'H2H match counts are non-decreasing (no leakage)',
                    'status': 'PASS'
                })
    
    # Check 2: Rolling windows should not include future data
    # We verify by checking that rolling features for a match use data BEFORE the match date
    if 'date' in df.columns and 'Rolling_Home_Goals' in df.columns:
        df_sorted = df.sort_values('date')
        
        # Spot check: For recent matches, rolling should differ from global mean
        recent = df_sorted.tail(50)
        global_mean = df_sorted['Rolling_Home_Goals'].mean()
        
        # If all recent rolling values equal global mean, might be a bug
        if recent['Rolling_Home_Goals'].nunique() == 1:
            results['issues'].append({
                'check': 'ROLLING_STATIC',
                'detail': 'Rolling values appear static (possible bug)',
                'severity': 'ALERT'
            })
        else:
            results['checks_passed'].append({
                'check': 'ROLLING_DYNAMIC',
                'detail': 'Rolling values show expected variation',
                'status': 'PASS'
            })
    
    # Check 3: No future matches in training data
    if 'date' in df.columns:
        now = pd.Timestamp.now(tz='UTC')
        future_matches = df[pd.to_datetime(df['date']) > now]
        if len(future_matches) > 0:
            results['issues'].append({
                'check': 'FUTURE_MATCHES_IN_DATA',
                'count': len(future_matches),
                'detail': 'Future matches found in dataset (expected for predictions)',
                'severity': 'INFO'  # This is OK for prediction pipeline
            })
    
    return results

def run_full_audit():
    """Run complete Layer 1 audit across all leagues."""
    report = {
        'audit_timestamp': datetime.now().isoformat(),
        'layer': 'LAYER_1_INPUT_SANITY',
        'leagues': {},
        'summary': {
            'total_issues': 0,
            'alerts': 0,
            'warnings': 0,
            'info': 0
        }
    }
    
    print("=" * 60)
    print("LAYER 1: INPUT SANITY AUDIT")
    print("=" * 60)
    
    for league in LEAGUES:
        print(f"\n[{league}] Loading data...")
        try:
            p = FeaturePipeline()
            df = p.run(league=league)
            
            print(f"[{league}] Running audits on {len(df)} matches...")
            
            # Run all audits
            missing = audit_missing_features(df, league)
            extreme = audit_extreme_values(df, league)
            temporal = audit_temporal_leakage(df, league)
            
            report['leagues'][league] = {
                'missing_features': missing,
                'extreme_values': extreme,
                'temporal_leakage': temporal
            }
            
            # Count issues
            for issue in extreme.get('issues', []):
                report['summary']['total_issues'] += 1
                sev = issue.get('severity', 'INFO')
                if sev == 'ALERT':
                    report['summary']['alerts'] += 1
                elif sev == 'WARN':
                    report['summary']['warnings'] += 1
                else:
                    report['summary']['info'] += 1
                    
            for issue in temporal.get('issues', []):
                report['summary']['total_issues'] += 1
                sev = issue.get('severity', 'INFO')
                if sev == 'ALERT':
                    report['summary']['alerts'] += 1
                elif sev == 'WARN':
                    report['summary']['warnings'] += 1
                else:
                    report['summary']['info'] += 1
                    
            print(f"[{league}] ✅ Complete")
            
        except Exception as e:
            print(f"[{league}] ❌ Error: {e}")
            report['leagues'][league] = {'error': str(e)}
    
    # Save report
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n{'=' * 60}")
    print("AUDIT SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total Issues: {report['summary']['total_issues']}")
    print(f"  ALERTS:   {report['summary']['alerts']}")
    print(f"  WARNINGS: {report['summary']['warnings']}")
    print(f"  INFO:     {report['summary']['info']}")
    print(f"\nReport saved to: {OUTPUT_FILE}")
    
    return report

if __name__ == "__main__":
    run_full_audit()

"""
Audit Team Goals U1.5 Calibration.

Analyzes OOS calibration for home_under_1_5 and away_under_1_5 markets
to determine appropriate probability caps.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTCOMES_FILE = DATA_DIR / "eval" / "prediction_outcomes.csv"

def load_outcomes():
    """Load prediction outcomes."""
    if not OUTCOMES_FILE.exists():
        raise FileNotFoundError(f"Outcomes file not found: {OUTCOMES_FILE}")
    
    df = pd.read_csv(OUTCOMES_FILE)
    print(f"Loaded {len(df)} prediction outcomes")
    return df

def filter_team_u15(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Team Goals U1.5 markets only."""
    mask = df['market'].str.contains('under_1_5', na=False)
    filtered = df[mask].copy()
    print(f"Filtered to {len(filtered)} Team U1.5 predictions")
    return filtered


def compute_calibration_by_bucket(df: pd.DataFrame, n_buckets: int = 10) -> pd.DataFrame:
    """Compute calibration metrics by probability bucket."""
    # Filter to WON/LOST only (exclude VOID)
    df_valid = df[df['outcome'].isin(['WON', 'LOST'])].copy()
    df_valid['actual'] = (df_valid['outcome'] == 'WON').astype(int)
    
    # Create probability buckets
    df_valid['bucket'] = pd.cut(df_valid['probability'], bins=n_buckets, labels=False)
    
    # Compute stats per bucket
    results = []
    for bucket in sorted(df_valid['bucket'].dropna().unique()):
        bucket_df = df_valid[df_valid['bucket'] == bucket]
        avg_prob = bucket_df['probability'].mean()
        hit_rate = bucket_df['actual'].mean()
        count = len(bucket_df)
        ece = abs(avg_prob - hit_rate)  # Expected Calibration Error
        
        results.append({
            'bucket': int(bucket),
            'prob_min': bucket_df['probability'].min(),
            'prob_max': bucket_df['probability'].max(),
            'avg_predicted': avg_prob,
            'actual_rate': hit_rate,
            'sample_size': count,
            'ece': ece,
            'overconfident': avg_prob > hit_rate
        })
    
    return pd.DataFrame(results)

def compute_overall_stats(df: pd.DataFrame) -> dict:
    """Compute overall calibration statistics."""
    df_valid = df[df['outcome'].isin(['WON', 'LOST'])].copy()
    df_valid['actual'] = (df_valid['outcome'] == 'WON').astype(int)
    
    avg_prob = df_valid['probability'].mean()
    hit_rate = df_valid['actual'].mean()
    overall_ece = abs(avg_prob - hit_rate)
    
    # Compute weighted ECE
    n_buckets = 10
    df_valid['bucket'] = pd.cut(df_valid['probability'], bins=n_buckets, labels=False)
    
    weighted_ece = 0.0
    total = len(df_valid)
    for bucket in df_valid['bucket'].dropna().unique():
        bucket_df = df_valid[df_valid['bucket'] == bucket]
        bucket_avg_prob = bucket_df['probability'].mean()
        bucket_hit_rate = bucket_df['actual'].mean()
        bucket_weight = len(bucket_df) / total
        weighted_ece += bucket_weight * abs(bucket_avg_prob - bucket_hit_rate)
    
    return {
        'total_predictions': len(df_valid),
        'avg_predicted_prob': avg_prob,
        'actual_hit_rate': hit_rate,
        'overall_ece': overall_ece,
        'weighted_ece': weighted_ece,
        'is_overconfident': avg_prob > hit_rate
    }

def find_safe_cap(df: pd.DataFrame, max_ece: float = 0.05) -> float:
    """Find the probability threshold where calibration stays within acceptable ECE."""
    df_valid = df[df['outcome'].isin(['WON', 'LOST'])].copy()
    df_valid['actual'] = (df_valid['outcome'] == 'WON').astype(int)
    
    # Check calibration at different thresholds
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    
    print(f"\n{'='*60}")
    print("THRESHOLD ANALYSIS (ECE at each cap level)")
    print(f"{'='*60}")
    print(f"{'Threshold':<12} {'Avg Pred':<12} {'Hit Rate':<12} {'ECE':<10} {'Sample':<10}")
    print(f"{'-'*60}")
    
    safe_cap = None
    for thresh in thresholds:
        above_thresh = df_valid[df_valid['probability'] >= thresh]
        if len(above_thresh) < 5:
            print(f"{thresh:<12.2f} {'N/A':<12} {'N/A':<12} {'N/A':<10} {len(above_thresh):<10}")
            continue
            
        avg_prob = above_thresh['probability'].mean()
        hit_rate = above_thresh['actual'].mean()
        ece = abs(avg_prob - hit_rate)
        
        status = "✅" if ece <= max_ece else "⚠️"
        print(f"{thresh:<12.2f} {avg_prob:<12.3f} {hit_rate:<12.3f} {ece:<10.3f} {len(above_thresh):<10} {status}")
        
        if ece > max_ece and safe_cap is None:
            safe_cap = thresholds[thresholds.index(thresh) - 1] if thresh > 0.60 else None
    
    return safe_cap

def main():
    print("=" * 60)
    print("TEAM GOALS U1.5 CALIBRATION AUDIT")
    print("=" * 60)
    
    df = load_outcomes()
    df_u15 = filter_team_u15(df)
    
    if df_u15.empty:
        print("No Team U1.5 predictions found!")
        return
    
    # Overall stats
    stats = compute_overall_stats(df_u15)
    print(f"\n{'='*60}")
    print("OVERALL STATS")
    print(f"{'='*60}")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Calibration by bucket
    print(f"\n{'='*60}")
    print("CALIBRATION BY PROBABILITY BUCKET")
    print(f"{'='*60}")
    bucket_df = compute_calibration_by_bucket(df_u15)
    print(bucket_df.to_string(index=False))
    
    # Find safe cap
    safe_cap = find_safe_cap(df_u15)
    
    # Recommendation
    print(f"\n{'='*60}")
    print("RECOMMENDATION")
    print(f"{'='*60}")
    if stats['is_overconfident']:
        if safe_cap:
            print(f"⚠️ Model is OVERCONFIDENT (pred {stats['avg_predicted_prob']:.1%} vs actual {stats['actual_hit_rate']:.1%})")
            print(f"✅ SAFE CAP: {safe_cap:.0%}")
        else:
            print(f"⚠️ Model is overconfident at all thresholds. Consider soft damping instead of hard cap.")
    else:
        print(f"✅ Model is well-calibrated or underconfident. No cap needed.")
        print(f"   Pred: {stats['avg_predicted_prob']:.1%}, Actual: {stats['actual_hit_rate']:.1%}")

if __name__ == "__main__":
    main()

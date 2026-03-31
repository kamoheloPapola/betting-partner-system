"""
Investigation Script: Over 2.5 Cards Market Calibration Analysis

This script analyzes the actual vs predicted probabilities for the Cards Over 2.5 market
to identify calibration issues and potential over/under-confidence in the model.

Run: python scripts/investigate_cards_o25.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from collections import defaultdict

# === CONFIG ===
HISTORICAL_DIR = Path("data/historical")
RESULTS_RAW_DIR = Path("data/results/raw")
SLIPS_DIR = Path("data/slips")

# Cards O2.5 threshold: Total cards > 2.5
CARD_THRESHOLD = 2.5


def load_all_league_data():
    """Load historical match data from all leagues."""
    all_matches = []
    
    leagues = {
        "PL": ["PL_season_2024_2025.csv", "PL_season_2023_2024.csv", "PL_season_2022_2023.csv"],
        "BL1": ["D1_season_2024_2025.csv", "D1_season_2023_2024.csv", "D1_season_2022_2023.csv"],
        "SA": ["I1_season_2024_2025.csv", "I1_season_2023_2024.csv", "I1_season_2022_2023.csv"],
        "PD": ["SP1_season_2024_2025.csv", "SP1_season_2023_2024.csv", "SP1_season_2022_2023.csv"],
        "FL1": ["F1_season_2024_2025.csv", "F1_season_2023_2024.csv", "F1_season_2022_2023.csv"]
    }
    
    # Try alternative file naming conventions
    for league, filenames in leagues.items():
        league_path = HISTORICAL_DIR / league
        if not league_path.exists():
            print(f"  [WARN] League directory not found: {league}")
            continue
            
        for file in league_path.iterdir():
            if file.suffix == '.csv':
                try:
                    df = pd.read_csv(file)
                    # Look for card columns (may vary by source)
                    card_cols = [c for c in df.columns if 'card' in c.lower() or c in ['HY', 'AY', 'HR', 'AR']]
                    if card_cols:
                        df['league'] = league
                        df['source_file'] = file.name
                        all_matches.append(df)
                        # print(f"  [OK] Loaded {len(df)} matches from {file.name}")
                except Exception as e:
                    print(f"  [FAIL] {file.name}: {e}")
    
    # Also try raw results directory
    if RESULTS_RAW_DIR.exists():
        for file in RESULTS_RAW_DIR.iterdir():
            if file.suffix == '.csv':
                try:
                    df = pd.read_csv(file)
                    card_cols = [c for c in df.columns if 'card' in c.lower() or c in ['HY', 'AY', 'HR', 'AR']]
                    if card_cols:
                        df['source_file'] = file.name
                        all_matches.append(df)
                        # print(f"  [OK] Loaded {len(df)} matches from raw/{file.name}")
                except Exception as e:
                    print(f"  [FAIL] raw/{file.name}: {e}")
                    
    if not all_matches:
        return pd.DataFrame()
        
    return pd.concat(all_matches, ignore_index=True)


def calculate_total_cards(df: pd.DataFrame) -> pd.Series:
    """Calculate total cards from various column naming conventions."""
    total = None
    
    # Method 1: Separate yellow and red cards
    if all(c in df.columns for c in ['HY', 'AY', 'HR', 'AR']):
        total = df['HY'].fillna(0) + df['AY'].fillna(0) + df['HR'].fillna(0) + df['AR'].fillna(0)
        
    # Method 2: Combined cards columns
    elif all(c in df.columns for c in ['HC', 'AC']):
        total = df['HC'].fillna(0) + df['AC'].fillna(0)
        
    # Method 3: Generic names
    elif 'home_cards' in df.columns and 'away_cards' in df.columns:
        total = df['home_cards'].fillna(0) + df['away_cards'].fillna(0)
        
    return total


def analyze_base_rates(df: pd.DataFrame):
    """Analyze the base rate (empirical frequency) of cards Over 2.5."""
    print("\n" + "=" * 60)
    print("CARDS OVER 2.5 BASE RATE ANALYSIS")
    print("=" * 60)
    
    total_cards = calculate_total_cards(df)
    if total_cards is None:
        print("[ERROR] Could not find card columns in data")
        return {}
        
    # Filter out rows with missing card data
    valid_mask = total_cards.notna()
    total_cards = total_cards[valid_mask]
    df_valid = df[valid_mask].copy()
    df_valid['total_cards'] = total_cards
    
    print(f"\nTotal matches with card data: {len(total_cards)}")
    
    # Calculate outcomes
    over_25 = (total_cards > CARD_THRESHOLD).sum()
    base_rate = over_25 / len(total_cards) if len(total_cards) > 0 else 0
    
    print(f"\nGlobal Cards O2.5 (>2.5 cards): {over_25} matches")
    print(f">>> GLOBAL BASE RATE FOR CARDS O2.5: {base_rate:.1%} <<<")
    
    # League breakdown
    print("\n--- League-Level Base Rates ---")
    if 'league' in df_valid.columns:
        league_stats = df_valid.groupby('league')['total_cards'].agg([
            ('matches', 'count'),
            ('o25_rate', lambda x: (x > CARD_THRESHOLD).mean()),
            ('mean', 'mean')
        ]).sort_values('o25_rate', ascending=False)
        
        print(f"{'League':<10} {'Matches':<10} {'O2.5 Rate':<12} {'Mean Cards'}")
        print("-" * 45)
        for league, row in league_stats.iterrows():
            print(f"{league:<10} {int(row['matches']):<10} {row['o25_rate']:<12.1%} {row['mean']:.2f}")
    
    # Distribution analysis (Global)
    print("\n--- Global Cards Distribution ---")
    card_dist = total_cards.value_counts().sort_index()
    cumulative = 0
    for cards, count in card_dist.items():
        cumulative += count
        pct = count / len(total_cards) * 100
        cum_pct = cumulative / len(total_cards) * 100
        marker = " <-- O2.5 CUTOFF" if cards == 2 else "" 
        print(f"  {int(cards):2d} cards: {count:4d} matches ({pct:5.1f}%) | Cumulative (U): {cum_pct:5.1f}%{marker}")
    
    return {
        'base_rate': base_rate,
        'total_matches': len(total_cards)
    }


def analyze_predicted_vs_actual(df: pd.DataFrame):
    """Compare model predictions with actual outcomes."""
    print("\n" + "=" * 60)
    print("PREDICTED VS ACTUAL ANALYSIS")
    print("=" * 60)
    
    # Try to load prediction history
    pred_files = [
        SLIPS_DIR / "ff_reconstructed_history.jsonl",
        SLIPS_DIR / "slip_history.jsonl"
    ]
    
    predictions = []
    for pf in pred_files:
        if pf.exists():
            with open(pf) as f:
                for line in f:
                    try:
                        sel = json.loads(line)
                        if 'CARDS_O2.5' in sel.get('market', '') or 'cards_over_2_5' in sel.get('market', '').lower():
                            predictions.append(sel)
                    except:
                        pass
    
    if not predictions:
        print("[WARN] No Cards O2.5 predictions found in history files")
        return {}
    
    print(f"\nFound {len(predictions)} Cards O2.5 predictions in history")
    
    # Analyze by outcome
    wins = [p for p in predictions if p.get('outcome') == 'WIN']
    losses = [p for p in predictions if p.get('outcome') == 'LOSS']
    
    print(f"  Wins:   {len(wins)}")
    print(f"  Losses: {len(losses)}")
    
    if (len(wins) + len(losses)) > 0:
        win_rate = len(wins) / (len(wins) + len(losses))
        print(f"\n>>> ACTUAL WIN RATE FOR CARDS O2.5: {win_rate:.1%} <<<")
    else:
        win_rate = 0
        print("\n>>> ACTUAL WIN RATE: N/A (No results) <<<")
    
    # Average Predicted Probability
    probs = [p.get('probability', p.get('conf', 0)) for p in predictions]
    avg_conf = sum(probs) / len(probs) if probs else 0
    print(f"Average Predicted Confidence: {avg_conf:.1%}")

    # Analyze by confidence bucket
    if probs:
        print("\n--- Calibration by Confidence Bucket ---")
        buckets = defaultdict(list)
        for p in predictions:
            prob = p.get('probability', p.get('conf', 0))
            if p.get('outcome') in ['WIN', 'LOSS']:
                bucket = int(prob * 10) / 10  # Round to nearest 0.1
                buckets[bucket].append(1 if p.get('outcome') == 'WIN' else 0)
        
        print(f"{'Predicted':<12} {'Actual':<10} {'Count':<8} {'Gap':<10}")
        print("-" * 40)
        for bucket in sorted(buckets.keys()):
            outcomes = buckets[bucket]
            actual = sum(outcomes) / len(outcomes)
            gap = bucket - actual
            print(f"{bucket:.0%}          {actual:.0%}        {len(outcomes):<8} {gap:+.0%}")
    
    return {
        'actual_win_rate': win_rate,
        'avg_conf': avg_conf
    }


def analyze_distribution_engine():
    """Analyze the ZeroInflatedEngine probability calculation for O2.5."""
    print("\n" + "=" * 60)
    print("ZEROINFLATED ENGINE ANALYSIS (O2.5)")
    print("=" * 60)
    
    try:
        import sys
        sys.path.insert(0, '.')
        from src.ml.distributions import ZeroInflatedEngine
        from scipy.stats import poisson
        
        engine = ZeroInflatedEngine()
        
        # Test with typical expected card values
        test_mus = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
        
        print("\n--- Probability vs Expected Cards (mu) ---")
        print(f"{'Mu':<6} {'P(O2.5) Raw':<15} {'P(O2.5) Damped':<18} {'P(O2.5) Clamped':<18} {'Target Damping'}")
        print("-" * 80)
        
        for mu in test_mus:
            pi_zero = 0.1
            x = np.arange(16)
            probs = (1 - pi_zero) * poisson.pmf(x, mu)
            probs[0] += pi_zero
            
            raw_prob = np.sum(probs[3:]) # Sum of index 3 and above (0,1,2 are under)
            
            # Manual dampening calculation to show effect (Updated to match codebase fix)
            # Codebase now uses ALPHA_HIGH_FREQ for O2.5
            alpha = engine.ALPHA_HIGH_FREQ
            damped = 0.5 + alpha * (raw_prob - 0.5)
            note = f"New Fix ({alpha:.2f})"
                
            clamped = max(0.05, min(0.92, damped))
            
            print(f"{mu:<6.1f} {raw_prob:<15.3f} {damped:<18.3f} {clamped:<18.3f} {note}")
        
    except Exception as e:
        print(f"[ERROR] Could not analyze engine: {e}")


def main():
    print("=" * 60)
    print("CARDS OVER 2.5 MARKET CALIBRATION CHECK")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)
    
    # Step 1: Load historical data
    df = load_all_league_data()
    if df.empty:
        print("[FATAL] No data loaded.")
        return
    
    # Step 2: Analyze base rates
    base_stats = analyze_base_rates(df)
    
    # Step 3: Analyze predictions vs actual
    analyze_predicted_vs_actual(df)
    
    # Step 4: Engine analysis
    analyze_distribution_engine()
    
    print("\n[DONE] Check complete.")


if __name__ == "__main__":
    main()

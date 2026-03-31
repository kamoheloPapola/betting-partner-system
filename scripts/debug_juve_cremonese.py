"""
Deep Investigation: JUVENTUS vs CREMONESE Cards Prediction Failure

The model predicted 72% for Over 2.5 Cards but only 1 card occurred.
This script investigates the root cause.
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.getcwd())

from pathlib import Path

def load_historical_data():
    """Load all historical data with card information."""
    data_dirs = [
        Path("data/processed/matches"),
        Path("data/historical")
    ]
    
    dfs = []
    for d in data_dirs:
        if d.exists():
            for f in d.rglob("*.csv"):
                try:
                    df = pd.read_csv(f)
                    df['source_file'] = f.name
                    dfs.append(df)
                except:
                    pass
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def calculate_total_cards(df):
    """Calculate total cards from various column schemas."""
    # Try different column patterns
    if 'total_cards' in df.columns:
        return df['total_cards']
    elif 'home_total_cards' in df.columns and 'away_total_cards' in df.columns:
        return df['home_total_cards'].fillna(0) + df['away_total_cards'].fillna(0)
    elif 'HY' in df.columns:
        return df['HY'].fillna(0) + df['AY'].fillna(0) + df['HR'].fillna(0) + df['AR'].fillna(0)
    else:
        return None

def main():
    print("=" * 70)
    print("DEEP INVESTIGATION: JUVENTUS vs CREMONESE Cards O2.5 Prediction Failure")
    print("=" * 70)
    
    df = load_historical_data()
    if df.empty:
        print("[FATAL] No data loaded")
        return
    
    print(f"Loaded {len(df)} total matches")
    
    # Calculate total cards
    total_cards = calculate_total_cards(df)
    if total_cards is None:
        print("[ERROR] Could not find card columns")
        return
    df['calc_total_cards'] = total_cards
    
    # Normalize team names
    for col in ['home_team', 'HomeTeam', 'Home']:
        if col in df.columns:
            df['home'] = df[col].astype(str).str.upper()
            break
    for col in ['away_team', 'AwayTeam', 'Away']:
        if col in df.columns:
            df['away'] = df[col].astype(str).str.upper()
            break
    
    # ============ JUVENTUS ANALYSIS ============
    print("\n" + "=" * 50)
    print("JUVENTUS CARD STATISTICS")
    print("=" * 50)
    
    juve_mask = (df['home'].str.contains('JUVENTUS', na=False)) | \
                (df['away'].str.contains('JUVENTUS', na=False))
    juve_df = df[juve_mask]
    
    print(f"Total Juventus matches: {len(juve_df)}")
    if len(juve_df) > 0:
        juve_cards = juve_df['calc_total_cards'].dropna()
        print(f"Avg total cards: {juve_cards.mean():.2f}")
        print(f"Std total cards: {juve_cards.std():.2f}")
        print(f"Median: {juve_cards.median():.0f}")
        print(f"Min/Max: {juve_cards.min():.0f} / {juve_cards.max():.0f}")
        
        o25 = (juve_cards > 2.5).mean() * 100
        print(f"\nCards O2.5 rate: {o25:.1f}%")
        
        # Low card games (<= 2 cards)
        low_card_games = juve_df[juve_df['calc_total_cards'] <= 2]
        print(f"\nLow card games (<= 2 cards): {len(low_card_games)} ({len(low_card_games)/len(juve_df)*100:.1f}%)")
        
        # Distribution
        print("\nCard Distribution:")
        dist = juve_cards.value_counts().sort_index()
        for cards, count in dist.items():
            pct = count / len(juve_cards) * 100
            bar = "#" * int(pct / 2)
            print(f"  {int(cards):2d} cards: {count:4d} ({pct:5.1f}%) {bar}")
    
    # ============ CREMONESE ANALYSIS ============
    print("\n" + "=" * 50)
    print("CREMONESE CARD STATISTICS")
    print("=" * 50)
    
    crem_mask = (df['home'].str.contains('CREMONESE', na=False)) | \
                (df['away'].str.contains('CREMONESE', na=False))
    crem_df = df[crem_mask]
    
    print(f"Total Cremonese matches: {len(crem_df)}")
    if len(crem_df) > 0:
        crem_cards = crem_df['calc_total_cards'].dropna()
        print(f"Avg total cards: {crem_cards.mean():.2f}")
        print(f"Std total cards: {crem_cards.std():.2f}")
        
        o25 = (crem_cards > 2.5).mean() * 100
        print(f"\nCards O2.5 rate: {o25:.1f}%")
        
        # Low card games
        low_card_games = crem_df[crem_df['calc_total_cards'] <= 2]
        print(f"Low card games (<= 2 cards): {len(low_card_games)} ({len(low_card_games)/len(crem_df)*100:.1f}%)")
        
        # Show recent matches
        print("\nRecent Cremonese matches with cards:")
        date_col = None
        for col in ['date', 'Date', 'DATE']:
            if col in crem_df.columns:
                date_col = col
                break
        
        if date_col:
            crem_recent = crem_df.sort_values(date_col, ascending=False).head(10)
            for _, row in crem_recent.iterrows():
                print(f"  {row.get(date_col, 'N/A')}: {row['home']} vs {row['away']} - {int(row['calc_total_cards'])} cards")
    
    # ============ H2H ANALYSIS ============
    print("\n" + "=" * 50)
    print("JUVENTUS vs CREMONESE HEAD TO HEAD")
    print("=" * 50)
    
    h2h = df[(df['home'].str.contains('JUVENTUS', na=False) & df['away'].str.contains('CREMONESE', na=False)) |
             (df['home'].str.contains('CREMONESE', na=False) & df['away'].str.contains('JUVENTUS', na=False))]
    
    print(f"Total H2H matches: {len(h2h)}")
    if len(h2h) > 0:
        h2h_cards = h2h['calc_total_cards'].dropna()
        print(f"Avg cards in H2H: {h2h_cards.mean():.2f}")
        print(f"Cards O2.5 in H2H: {(h2h_cards > 2.5).mean()*100:.1f}%")
        
        for _, row in h2h.iterrows():
            print(f"  {row.get('date', 'N/A')}: {row['home']} vs {row['away']} - {int(row['calc_total_cards'])} cards")
    
    # ============ PROBLEM: GLOBAL ANALYSIS ============
    print("\n" + "=" * 50)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 50)
    
    # Overall base rate
    all_cards = df['calc_total_cards'].dropna()
    base_rate = (all_cards > 2.5).mean()
    print(f"Global Cards O2.5 Base Rate: {base_rate*100:.1f}%")
    print(f"Avg cards per match: {all_cards.mean():.2f}")
    
    # Now simulate what the ZeroInflatedEngine would predict
    print("\n--- Engine Simulation ---")
    try:
        from src.ml.distributions import ZeroInflatedEngine
        
        engine = ZeroInflatedEngine()
        
        # Test with different mu values
        print(f"\n{'Mu (Expected Cards)':<22} {'Raw P(O2.5)':<15} {'Damped':<15} {'After Clamp'}")
        print("-" * 70)
        
        # What mu would the model have predicted?
        # Juve avg + Crem avg
        juve_avg = juve_df['calc_total_cards'].mean() if len(juve_df) > 0 else 4.0
        crem_avg = crem_df['calc_total_cards'].mean() if len(crem_df) > 0 else 4.0
        
        test_mus = [2.0, 3.0, 4.0, 4.5, 5.0, juve_avg, crem_avg, (juve_avg + crem_avg)/2]
        
        for mu in test_mus:
            res = engine.calculate_probabilities(mu, pi_zero=0.03)
            # Calculate raw
            x = np.arange(16)
            probs = engine.pmf(x, mu, 0.03)
            raw_o25 = np.sum(probs[3:])
            
            damped = 0.5 + 0.90 * (raw_o25 - 0.5)  # Using ALPHA_HIGH_FREQ
            clamped = res['cards_over_2_5']
            
            print(f"{mu:22.2f} {raw_o25:15.3f} {damped:15.3f} {clamped:.3f}")
        
        # CRITICAL CHECK: What happens with mu derived from league average?
        print("\n--- CRITICAL: What mu gives 72% probability? ---")
        for mu in np.arange(3.0, 7.0, 0.1):
            res = engine.calculate_probabilities(mu, pi_zero=0.03)
            if abs(res['cards_over_2_5'] - 0.72) < 0.01:
                print(f"Mu = {mu:.2f} gives P(O2.5) = {res['cards_over_2_5']:.3f}")
        
    except Exception as e:
        print(f"Engine simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ============ HYPOTHESIS ============
    print("\n" + "=" * 50)
    print("HYPOTHESIS")
    print("=" * 50)
    
    print("""
Potential causes for the erroneous 72% prediction:

1. MODEL PREDICTION TOO HIGH:
   - The underlying ml model (nb_cards) is predicting mu too high
   - This could be due to:
     a) Training data imbalance
     b) Stale features (old rolling averages)
     c) Missing context (new season, different referee, etc.)

2. TEAM OFFSETS:
   - The TeamOffsetManager might be adding positive bias
   - Check off_h (home_card_bias) and off_a (away_card_bias)

3. ZERO-INFLATION PARAMETER:
   - pi_zero=0.1 is used in _calc_cards but base analysis uses 0.03
   - This inconsistency could affect probabilities

4. VARIANCE MULTIPLIER:
   - intensity_boost might be incorrectly triggered
   - This would widen the distribution

5. FUNDAMENTAL ISSUE:
   - The model predicts AVERAGE cards, not the range
   - A 1-card game is a tail event, but the model has no way to
     reduce confidence based on uncertainty.
""")

if __name__ == "__main__":
    main()

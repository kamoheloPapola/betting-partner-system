
import pandas as pd
import numpy as np
import glob
from pathlib import Path

def analyze_all_markets():
    print("Starting Comprehensive Market Analysis...")
    
    # 1. Load All Predictions
    # Include all prediction files
    all_files = glob.glob("data/predictions/*.csv")
    print(f"Found {len(all_files)} prediction files.")
    
    all_preds = []
    for f in all_files:
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            
            # Normalize Columns
            if 'probability' in df.columns and 'predicted_probability' not in df.columns:
                df.rename(columns={'probability': 'predicted_probability'}, inplace=True)
            
            # Ensure essential columns exist
            if 'predicted_probability' not in df.columns:
                print(f"Skipping {f}: No probability column")
                continue
                
            all_preds.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not all_preds:
        print("No predictions loaded.")
        return

    preds_df = pd.concat(all_preds, ignore_index=True)
    print(f"Total Predictions Loaded: {len(preds_df)}")
    
    # 2. Load Results
    result_files = glob.glob("data/processed/matches/*.csv")
    all_results = []
    
    for f in result_files:
        if "upcoming" in f: continue
        try:
            df = pd.read_csv(f)
            # Map Result Columns to standard Analysis expected names
            # Standard: HC, AC, HY, AY
            # File: home_corners, away_corners, home_total_cards, away_total_cards
            
            rename_map = {}
            if 'home_corners' in df.columns: rename_map['home_corners'] = 'HC'
            if 'away_corners' in df.columns: rename_map['away_corners'] = 'AC'
            if 'home_total_cards' in df.columns: rename_map['home_total_cards'] = 'HY' # Using Total Cards as HY for generic count
            if 'away_total_cards' in df.columns: rename_map['away_total_cards'] = 'AY'
            
            df.rename(columns=rename_map, inplace=True)
            
            # Ensure needed columns
            cols = ['match_id', 'home_score', 'away_score', 'status', 'home_team', 'away_team', 'date',
                    'HC', 'AC', 'HY', 'AY']
            df = df[[c for c in cols if c in df.columns]]
            all_results.append(df)
        except Exception as e:
            pass 
            
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"Total Matches in DB: {len(results_df)}")
    
    # 3. Merge
    if 'match_id' in preds_df.columns: preds_df['match_id'] = preds_df['match_id'].astype(str)
    if 'match_id' in results_df.columns: results_df['match_id'] = results_df['match_id'].astype(str)
    
    merged = pd.merge(preds_df, results_df, on='match_id', how='inner')
    print(f"Matched Predictions (by ID): {len(merged)}")
    
    # 4. Evaluate
    stats = {} # {market_group: {correct, total, brier}}
    
    print("\nEvaluating...")
    for idx, row in merged.iterrows():
        status = row.get('status')
        if status not in ['FT', 'AET', 'PEN', 'FINISHED', 'Match Finished']: 
            continue
            
        market = str(row.get('market')).lower()
        prob = row.get('predicted_probability')
        
        try:
            prob = float(prob) if pd.notna(prob) else np.nan
        except:
            prob = np.nan
            
        if pd.isna(prob): continue
        
        # Ground Truth
        h = float(row.get('home_score', 0))
        a = float(row.get('away_score', 0))
        hc = float(row.get('HC', 0)) if pd.notna(row.get('HC')) else 0
        ac = float(row.get('AC', 0)) if pd.notna(row.get('AC')) else 0
        hy = float(row.get('HY', 0)) if pd.notna(row.get('HY')) else 0
        ay = float(row.get('AY', 0)) if pd.notna(row.get('AY')) else 0
        
        outcome = None
        
        # --- OUTCOMES ---
        if market == 'home_win': outcome = 1.0 if h > a else 0.0
        elif market == 'away_win': outcome = 1.0 if a > h else 0.0
        elif market == 'draw': outcome = 1.0 if h == a else 0.0
        elif 'double_chance' in market:
            if '1x' in market: outcome = 1.0 if h >= a else 0.0
            elif 'x2' in market: outcome = 1.0 if a >= h else 0.0
            elif '12' in market: outcome = 1.0 if h != a else 0.0
            
        # --- GOALS ---
        elif market == 'under_2_5': outcome = 1.0 if (h + a) < 2.5 else 0.0
        elif market == 'over_2_5': outcome = 1.0 if (h + a) > 2.5 else 0.0
        elif market == 'over_1_5': outcome = 1.0 if (h + a) > 1.5 else 0.0
        elif market == 'home_under_1_5' or 'home_tg_u1.5' in market: outcome = 1.0 if h < 1.5 else 0.0
        elif market == 'away_under_1_5' or 'away_tg_u1.5' in market: outcome = 1.0 if a < 1.5 else 0.0
        elif market == 'btts_yes': outcome = 1.0 if (h > 0 and a > 0) else 0.0
        
        # --- CORNERS ---
        elif 'corners' in market or 'corn_' in market:
            # corn_home_o4.5, corn_u11, etc.
            if 'u11' in market: outcome = 1.0 if (hc + ac) < 11.5 else 0.0 # Standard u11.5 line usually
            elif 'o7.5' in market: outcome = 1.0 if (hc+ac) > 7.5 else 0.0
            elif 'home_win' in market: outcome = 1.0 if hc > ac else 0.0
            elif 'away_win' in market: outcome = 1.0 if ac > hc else 0.0
            else:
                # Try simple thresholds like corn_home_o4.5
                try:
                    parts = market.split('_')
                    # e.g. ['corn', 'home', 'o4.5']
                    team_target = hc if 'home' in market else ac
                    if 'total' in market or ('home' not in market and 'away' not in market): team_target = hc + ac
                    
                    val_str = parts[-1].replace('o','').replace('u','')
                    val = float(val_str) if '.' in val_str else float(val_str)/10.0 # Fix: 25 -> 2.5
                    
                    if 'o' in parts[-1]: outcome = 1.0 if team_target > val else 0.0
                    elif 'u' in parts[-1]: outcome = 1.0 if team_target < val else 0.0
                except: pass

        # --- CARDS ---
        elif 'card' in market:
            # cards_over_2_5, etc.
            total_cards = hy + ay # Approximating
            
            try:
                parts = market.split('_')
                # e.g. ['cards', 'over', '2', '5'] -> 2.5
                # or ['card', 'u45'] -> 4.5
                
                # Check for simple over/under
                val = None
                is_over = 'over' in market or 'o' in parts[-1]
                
                if 'u4.5' in market or 'u45' in market or 'under_4_5' in market: val = 4.5; is_over = False
                elif 'o2.5' in market or 'over_2_5' in market: val = 2.5; is_over = True
                elif 'o3.5' in market or 'over_3_5' in market: val = 3.5; is_over = True
                
                if val is not None:
                     outcome = 1.0 if total_cards > val else 0.0
                     if not is_over: outcome = 1.0 - outcome # Under
            except: pass
            
        if outcome is not None:
            is_correct = (round(prob) == outcome)
            
            if market not in stats:
                stats[market] = {'correct': 0, 'total': 0, 'brier': 0.0}
            
            stats[market]['total'] += 1
            if is_correct: stats[market]['correct'] += 1
            stats[market]['brier'] += (prob - outcome) ** 2

    # 5. Print Report
    print("\n" + "="*80)
    print("COMPREHENSIVE MARKET PERFORMANCE REPORT")
    print("="*80)
    print(f"{'Market':<25} | {'Acc':<8} | {'Brier':<8} | {'N':<6}")
    print("-" * 80)
    
    sorted_markets = sorted(stats.items(), key=lambda x: x[1]['correct']/x[1]['total'] if x[1]['total']>0 else 0, reverse=True)
    
    for mkt, s in sorted_markets:
        if s['total'] < 5: continue # Lower threshold to see rarer markets
        acc = s['correct'] / s['total']
        brier = s['brier'] / s['total']
        print(f"{mkt:<25} | {acc:.2%} | {brier:.4f}   | {s['total']}")
    
    print("-" * 80)

if __name__ == "__main__":
    analyze_all_markets()

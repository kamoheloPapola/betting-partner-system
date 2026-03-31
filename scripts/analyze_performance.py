import pandas as pd
import numpy as np
from pathlib import Path
import glob

def analyze_performance():
    print("Starting Robust Performance Analysis...")
    
    # 1. Load Predictions
    # Exclude partial 'predictions_corners' files which have mismatched IDs
    all_files = glob.glob("data/predictions/predictions_*.csv")
    pred_files = [f for f in all_files if "predictions_corners" not in f and "predictions_cards" not in f]
    
    print(f"Found {len(pred_files)} prediction files (excluding legacy partials).")
    print(f"Loading: {[str(Path(f).name) for f in pred_files[:5]]}...")
    
    all_preds = []
    
    for f in pred_files:
        try:
            # Skip bad lines to avoid ParserError
            df = pd.read_csv(f, on_bad_lines='skip')
            all_preds.append(df)
            print(f"Loaded {f}: {len(df)} rows")
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not all_preds:
        print("No predictions loaded.")
        return

    preds_df = pd.concat(all_preds, ignore_index=True)
    print(f"Total Predictions: {len(preds_df)}")
    
    # 2. Load Results
    result_files = glob.glob("data/processed/matches/*.csv")
    all_results = []
    
    for f in result_files:
        if "upcoming" in f: continue
        try:
            df = pd.read_csv(f)
            # Ensure needed columns
            cols = ['match_id', 'home_score', 'away_score', 'status', 'home_team', 'away_team', 'date',
                    'HC', 'AC', 'HY', 'AY'] # Corners/Cards columns
            df = df[[c for c in cols if c in df.columns]]
            all_results.append(df)
        except Exception as e:
            pass # benign
            
    results_df = pd.concat(all_results, ignore_index=True)
    print(f"Total Matches in DB: {len(results_df)}")
    
    # Debug Columns
    print(f"Preds Columns: {list(preds_df.columns)}")
    print(f"Results Columns: {list(results_df.columns)}")
    if 'date' in results_df.columns:
        print(f"Results Date Range: {results_df['date'].min()} to {results_df['date'].max()}")
    
    # 3. Merge
    # Ensure match_id is string
    if 'match_id' in preds_df.columns: preds_df['match_id'] = preds_df['match_id'].astype(str)
    if 'match_id' in results_df.columns: results_df['match_id'] = results_df['match_id'].astype(str)
    
    merged = pd.DataFrame()
    if 'match_id' in preds_df.columns and 'match_id' in results_df.columns:
        merged = pd.merge(
            preds_df, 
            results_df, 
            on='match_id', 
            how='inner'
        )
    
    print(f"Matched Predictions to Results (by ID): {len(merged)}")
    
    # Fallback: Merge on Date + Teams (Fuzzy)
    if len(merged) == 0:
        print("Attempting duplicate fuzzy merge on (date, home_team, away_team)...")
        # Normalize date to YYYY-MM-DD
        if 'date' in preds_df.columns:
            preds_df['date_short'] = preds_df['date'].astype(str).str[:10]
        elif 'kickoff_date' in preds_df.columns:
             preds_df['date_short'] = preds_df['kickoff_date'].astype(str).str[:10]
             
        if 'date' in results_df.columns:
            results_df['date_short'] = results_df['date'].astype(str).str[:10]
            
        if 'date_short' not in preds_df.columns or 'date_short' not in results_df.columns:
             print("Cannot fuzzy merge: missing date column.")
             return

        # Normalize Team Names (Simple)
        def clean(x): return str(x).lower().strip()
        
        if 'home_team' in preds_df.columns: preds_df['home_clean'] = preds_df['home_team'].apply(clean)
        if 'away_team' in preds_df.columns: preds_df['away_clean'] = preds_df['away_team'].apply(clean)
        
        if 'home_team' in results_df.columns: results_df['home_clean'] = results_df['home_team'].apply(clean)
        if 'away_team' in results_df.columns: results_df['away_clean'] = results_df['away_team'].apply(clean)
        
        merged = pd.merge(
            preds_df,
            results_df,
            left_on=['date_short', 'home_clean', 'away_clean'],
            right_on=['date_short', 'home_clean', 'away_clean'],
            how='inner',
            suffixes=('', '_res')
        )
        print(f"Matched Predictions to Results (by Teams): {len(merged)}")
    
    # Debug Markets
    print(f"Unique Markets in Merged Data: {merged['market'].unique()}")

    # Debug SA Alignment
    print("\n--- DEBUG SA ALIGNMENT ---")
    sa_merged = merged[merged['league'] == 'SA']
    if not sa_merged.empty:
        # Check suffixes
        h_col_p = 'home_team_x' if 'home_team_x' in sa_merged.columns else 'home_team'
        h_col_r = 'home_team_y' if 'home_team_y' in sa_merged.columns else 'home_team'
        
        # Filter for non-NaN prob to see active predictions
        valid_sa = sa_merged.dropna(subset=['predicted_probability'])
        print(f"Valid SA Predictions: {len(valid_sa)} / {len(sa_merged)}")
        
        cols = ['match_id', h_col_p, h_col_r, 'market', 'predicted_probability', 'home_score', 'away_score']
        cols = [c for c in cols if c in sa_merged.columns]
        
        if not valid_sa.empty:
            print(valid_sa[cols].head(20).to_string())
        else:
             print("All SA predictions are NaN??")
    else:
        print("No SA matches in merged data??")
    print("-" * 30)

    # 4. Evaluate
    correct_count = 0
    total_eval = 0
    
    # Breakdown
    # Structure: {league: {market: stats}}
    stats = {}
    
    for idx, row in merged.iterrows():
        status = row.get('status')
        if status not in ['FT', 'AET', 'PEN', 'FINISHED', 'Match Finished']: 
            continue
            
        market = row.get('market')
        league = row.get('league', 'Unknown')
        prob = row.get('predicted_probability')
        try:
            prob = float(prob) if pd.notna(prob) else np.nan
        except:
            prob = np.nan
        
        # Ground Truth
        h = float(row.get('home_score', 0))
        a = float(row.get('away_score', 0))
        hc = float(row.get('HC', 0)) if pd.notna(row.get('HC')) else 0
        ac = float(row.get('AC', 0)) if pd.notna(row.get('AC')) else 0
        hy = float(row.get('HY', 0)) if pd.notna(row.get('HY')) else 0
        ay = float(row.get('AY', 0)) if pd.notna(row.get('AY')) else 0
        
        m_lower = str(market).lower()
        outcome = None
        
        # --- OUTCOMES ---
        if m_lower == 'home_win':
            outcome = 1.0 if h > a else 0.0
        elif m_lower == 'away_win':
            outcome = 1.0 if a > h else 0.0
        elif m_lower == 'draw':
            outcome = 1.0 if h == a else 0.0
        elif 'double_chance_1x' in m_lower:
            outcome = 1.0 if (h > a or h == a) else 0.0
        elif 'double_chance_x2' in m_lower:
            outcome = 1.0 if (a > h or h == a) else 0.0
        elif 'double_chance_12' in m_lower:
            outcome = 1.0 if (h > a or a > h) else 0.0
        
        # --- GOALS ---
        elif m_lower == 'under_2_5':
            outcome = 1.0 if (h + a) < 2.5 else 0.0
        elif m_lower == 'over_2_5':
            outcome = 1.0 if (h + a) > 2.5 else 0.0
        elif m_lower == 'over_1_5':
            outcome = 1.0 if (h + a) > 1.5 else 0.0
        elif m_lower == 'btts_yes':
            outcome = 1.0 if (h > 0 and a > 0) else 0.0
        elif 'home_tg_u1.5' in m_lower or 'home_under_1_5' in m_lower:
            outcome = 1.0 if h < 1.5 else 0.0
        elif 'away_tg_u1.5' in m_lower or 'away_under_1_5' in m_lower:
            outcome = 1.0 if a < 1.5 else 0.0

        # --- CORNERS ---
        elif 'corners_u11.5' in m_lower or 'corn_u11' in m_lower:
            outcome = 1.0 if (hc + ac) < 11.5 else 0.0
        elif 'corners_home_win' in m_lower:
            outcome = 1.0 if hc > ac else 0.0
        elif 'corners_away_win' in m_lower:
            outcome = 1.0 if ac > hc else 0.0
        elif 'corn_home_o' in m_lower: # e.g. corn_home_o45 -> Home Corners > 4.5
             # extract number
             try:
                 val_str = m_lower.split('o')[-1]
                 val = float(val_str) if '.' in val_str else float(val_str)/10.0 # 45 -> 4.5
                 outcome = 1.0 if hc > val else 0.0
             except: pass
        elif 'corn_away_o' in m_lower:
             try:
                 val_str = m_lower.split('o')[-1]
                 val = float(val_str) if '.' in val_str else float(val_str)/10.0
                 outcome = 1.0 if ac > val else 0.0
             except: pass

        # --- CARDS ---
        elif 'cards_u4.5' in m_lower or 'card_u45' in m_lower:
            outcome = 1.0 if (hy + ay) < 4.5 else 0.0
        elif 'cards_over_2_5' in m_lower:
            outcome = 1.0 if (hy + ay) > 2.5 else 0.0
            
        if outcome is not None and pd.notna(prob):
            is_correct = (round(prob) == outcome)
            
            if league not in stats:
                stats[league] = {}
            if market not in stats[league]:
                stats[league][market] = {'correct': 0, 'total': 0, 'brier': 0.0}
            
            stats[league][market]['total'] += 1
            if is_correct:
                stats[league][market]['correct'] += 1
            
            stats[league][market]['brier'] += (prob - outcome) ** 2
            
            total_eval += 1
            if is_correct:
                correct_count += 1

    # 5. Report
    print("\n" + "="*60)
    print("GLOBAL LIVE PERFORMANCE REPORT")
    print("="*60)
    
    if total_eval == 0:
        print("No finished matches evaluated.")
        return

    acc = correct_count / total_eval
    print(f"Overall Accuracy: {acc:.2%} ({correct_count}/{total_eval})")
    
    for league in sorted(stats.keys()):
        print(f"\n[ {league} ]")
        l_correct = 0
        l_total = 0
        
        for mkt, s in stats[league].items():
            if s['total'] > 0:
                m_acc = s['correct'] / s['total']
                m_brier = s['brier'] / s['total']
                print(f"  {mkt:<15} | Acc: {m_acc:.2%} | Brier: {m_brier:.4f} | N: {s['total']}")
                l_correct += s['correct']
                l_total += s['total']
        
        if l_total > 0:
            print(f"  >> League Avg   | Acc: {l_correct/l_total:.2%} | N: {l_total}")

if __name__ == "__main__":
    analyze_performance()

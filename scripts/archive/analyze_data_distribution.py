import pandas as pd
from pathlib import Path
from src.config import PROCESSED_DATA_DIR

def analyze_distributions():
    matches_dir = PROCESSED_DATA_DIR / "matches"
    files = list(matches_dir.glob("*.csv"))
    
    data = []
    for f in files:
        if "upcoming" in f.name or "BACKFILL" in f.name: continue
        try:
            df = pd.read_csv(f)
            # Infer league and season from filename or columns
            parts = f.stem.split('_')
            league = parts[0]
            season = parts[1] if len(parts) > 1 else "Unknown"
            
            total = len(df)
            if total == 0: continue
            
            home_wins = len(df[df['home_score'] > df['away_score']])
            draws = len(df[df['home_score'] == df['away_score']])
            away_wins = len(df[df['home_score'] < df['away_score']])
            
            data.append({
                "league": league,
                "season": season,
                "matches": total,
                "home_win_pct": home_wins / total,
                "draw_pct": draws / total,
                "away_win_pct": away_wins / total
            })
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            
    df_stats = pd.DataFrame(data)
    
    print("\n--- Home Win % by League & Season (Recent) ---")
    # Filter for recent seasons
    recent = df_stats[df_stats['season'].astype(str).str.startswith('202')].sort_values(['league', 'season'])
    print(recent.to_string(formatters={
        'home_win_pct': '{:.2%}'.format,
        'draw_pct': '{:.2%}'.format,
        'away_win_pct': '{:.2%}'.format
    }))
    
    print("\n--- Aggregate Stats by League (All Time) ---")
    agg = df_stats.groupby('league').apply(lambda x: pd.Series({
        'weighted_hw': (x['home_win_pct'] * x['matches']).sum() / x['matches'].sum(),
        'total_matches': x['matches'].sum()
    }))
    print(agg.to_string(formatters={'weighted_hw': '{:.2%}'.format}))

if __name__ == "__main__":
    analyze_distributions()

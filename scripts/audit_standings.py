from src.utils.standings import StandingsManager
from datetime import datetime
import pandas as pd
import json

def audit():
    sm = StandingsManager()
    as_of = datetime(2026, 1, 6)
    
    leagues = ['PL', 'PD', 'SA', 'BL1', 'FL1']
    
    for lg in leagues:
        print(f"\n--- {lg} STANDINGS (As of {as_of.date()}) ---")
        table = sm.get_table(lg, 2025, as_of)
        
        if not table:
            print(f"No standings found for {lg} 2025/26.")
            continue

        rows = []
        for team, data in table.items():
            rows.append({
                'Rank': data['rank'],
                'Team': team,
                'PPG': data['season_ppg'],
                'RecPPG': data['recent_ppg'],
                'Div': data['divergence'],
                'Mom': data['momentum_band'],
                'Band': data['status_band']
            })
        
        df = pd.DataFrame(rows).sort_values('Rank')
        print(df.to_string(index=False))

if __name__ == "__main__":
    audit()

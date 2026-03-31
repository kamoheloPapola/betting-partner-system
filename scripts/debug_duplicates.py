
import pandas as pd
from src.features.pipeline import FeaturePipeline
from src.core.container import ServiceContainer

def check_duplicates():
    print("Running Pipeline...")
    pipeline = FeaturePipeline()
    # Force fresh load to be sure
    df = pipeline.run(league=None, force_refresh=True)
    
    print(f"Total Rows: {len(df)}")
    
    # Check simple duplicates
    duplicates = df[df.duplicated(subset=['match_id'], keep=False)]
    
    if not duplicates.empty:
        print(f"\n[!] Found {len(duplicates)} duplicate rows based on 'match_id':")
        print(duplicates[['date', 'league', 'home_team', 'away_team', 'match_id']].sort_values('match_id').head(20))
    else:
        print("\n[+] No duplicates found in Pipeline output.")

    # Check specifically for today/upcoming
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    print(f"\nChecking data around {today}...")
    upcoming = df[df['date'] >= today]
    dup_upcoming = upcoming[upcoming.duplicated(subset=['home_team', 'away_team', 'date'], keep=False)]
    
    if not dup_upcoming.empty:
        print(f"[!] Found duplicates in UPCOMING/TODAY data:")
        print(dup_upcoming[['date', 'league', 'home_team', 'away_team']].sort_values('home_team'))

if __name__ == "__main__":
    check_duplicates()

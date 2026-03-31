import pandas as pd
import glob
from pathlib import Path
from src.config import PROCESSED_DATA_DIR

def audit_naming():
    print("=== Naming Impact Audit ===")
    matches_dir = PROCESSED_DATA_DIR / "matches"
    files = list(matches_dir.glob("*.csv"))
    
    all_rows = []
    for f in files:
        df = pd.read_csv(f)
        df['source_file'] = f.name
        all_rows.append(df)
    
    if not all_rows:
        print("No matches found.")
        return
        
    df = pd.concat(all_rows, ignore_index=True)
    
    # 1. Identify Casing Variants
    teams = pd.concat([df['home_team'], df['away_team']]).unique()
    variants = {}
    for team in teams:
        lower = str(team).lower()
        if lower not in variants:
            variants[lower] = set()
        variants[lower].add(team)
    
    multiple_variants = {k: v for k, v in variants.items() if len(v) > 1}
    
    print(f"\nTotal Teams Found: {len(teams)}")
    print(f"Teams with Multiple Casing Variants: {len(multiple_variants)}")
    
    if multiple_variants:
        print("\nTop Variants:")
        for k, v in list(multiple_variants.items())[:10]:
            print(f"- {k}: {v}")
            
    # 2. Impact on Rolling Stats (Simulation)
    # We choose a team with multiple variants
    if multiple_variants:
        target_lower = list(multiple_variants.keys())[0]
        target_variants = multiple_variants[target_lower]
        
        # Count matches per variant
        for v in target_variants:
            v_count = len(df[(df['home_team'] == v) | (df['away_team'] == v)])
            print(f"  Variant '{v}': {v_count} matches")
            
        full_count = len(df[df['home_team'].str.lower() == target_lower]) + \
                     len(df[df['away_team'].str.lower() == target_lower])
        print(f"  Total (Case-Insensitive): {full_count} matches")
        
    # 3. Overall Fragmentation Rate
    affected_matches = df[
        df['home_team'].str.lower().isin(multiple_variants.keys()) | 
        df['away_team'].str.lower().isin(multiple_variants.keys())
    ]
    print(f"\nMatches Affected by Naming Fragmentation: {len(affected_matches)} ({len(affected_matches)/len(df):.1%})")

if __name__ == "__main__":
    audit_naming()

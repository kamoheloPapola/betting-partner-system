import pandas as pd
from pathlib import Path
from src.utils.naming import normalize_team_name
from src.config import DATA_DIR
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("consolidate_offsets")

def consolidate():
    offset_file = DATA_DIR / "models" / "team_offsets.csv"
    if not offset_file.exists():
        logger.error(f"Offset file not found at {offset_file}")
        return

    logger.info(f"Reading {offset_file}...")
    df = pd.read_csv(offset_file)
    
    # Store original row count
    original_count = len(df)
    
    # Track collisions for logging
    # (normalized_name, league) -> list of raw names
    collisions = {}
    
    def get_norm(row):
        raw_name = row['team_name']
        league = row['league']
        norm_name = normalize_team_name(raw_name, league=league)
        
        key = (norm_name, league)
        if key not in collisions:
            collisions[key] = set()
        collisions[key].add(raw_name)
        
        return norm_name

    df['norm_name'] = df.apply(get_norm, axis=1)
    
    # Identify collisions where multiple raw names collapse into one
    for (norm_name, league), raw_names in collisions.items():
        if len(raw_names) > 1:
            logger.info(f"[OFFSETS] Consolidated {len(raw_names)} entries -> '{norm_name}' ({league})")
    
    # Consolidate: Group by normalized name and league, keep entry with max match_count
    # Sort by match_count descending to easily pick the first
    consolidated_df = df.sort_values('match_count', ascending=False).drop_duplicates(['norm_name', 'league'])
    
    # Clean up: remove the temporary norm_name column and restore team_name to normalized value
    consolidated_df['team_name'] = consolidated_df['norm_name']
    consolidated_df = consolidated_df.drop(columns=['norm_name'])
    
    # Sort for predictability
    consolidated_df = consolidated_df.sort_values(['league', 'team_name'])
    
    # Save back
    consolidated_df.to_csv(offset_file, index=False)
    
    new_count = len(consolidated_df)
    logger.info(f"Consolidation complete. Reduced {original_count} entries to {new_count}.")

if __name__ == "__main__":
    consolidate()

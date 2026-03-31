"""Check what matches are being predicted yesterday"""
import sys
import os
sys.path.insert(0, os.getcwd())

import pandas as pd
from datetime import datetime, timedelta
from src.core.container import ServiceContainer
from src.cli.utils import filter_matches_by_date

pipeline = ServiceContainer.get_instance().pipeline
df = pipeline.run(league='SA')

# Filter to yesterday's matches
yesterday = datetime.now() - timedelta(days=1)
df_filtered = df[pd.to_datetime(df['date']).dt.date == yesterday.date()]

print(f'Filtered to {len(df_filtered)} matches for yesterday ({yesterday.date()})')
for _, m in df_filtered.iterrows():
    home = m['home_team']
    away = m['away_team']
    h2h_count = m.get('h2h_match_count', 'N/A')
    h2h_cards = m.get('h2h_avg_cards', 'N/A')
    print(f'  {home} vs {away}')
    print(f'    h2h_match_count: {h2h_count}')
    print(f'    h2h_avg_cards: {h2h_cards}')

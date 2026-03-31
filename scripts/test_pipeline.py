"""Quick test script for show-predictions pipeline"""
import sys
sys.path.insert(0, '.')

# Test imports
from src.core.container import ServiceContainer
from src.cli.utils import filter_matches_by_date

print('Imports successful')

# Test pipeline
container = ServiceContainer.get_instance()
df = container.pipeline.run()
print(f'Pipeline returned {len(df)} rows')

# Test date filter
from datetime import datetime
filtered = filter_matches_by_date(df, 'yesterday', show_all=False, user_timezone='LOCAL')
print(f'Yesterday filter: {len(filtered)} matches')

if len(filtered) > 0:
    print('Sample matches:')
    for idx, row in filtered.head(3).iterrows():
        print(f'  {row["home_team"]} vs {row["away_team"]}')
else:
    print('No matches found for yesterday')
    # Check what dates we have
    print('Available dates in data:')
    print(df['date'].value_counts().head(10))

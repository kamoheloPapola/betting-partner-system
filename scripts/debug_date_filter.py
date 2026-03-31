import pandas as pd
import pytz
from src.core.container import ServiceContainer
from src.cli.utils import filter_matches_by_date

def debug_filter():
    print("--- Date Filter Debug ---")
    
    # 1. Load Data
    container = ServiceContainer.get_instance()
    df = container.pipeline.run()
    
    # 2. Match of interest (Udinese vs Pisa)
    match = df[df['home_team'] == 'Udinese'].iloc[-1:]
    raw_date = match.iloc[0]['date']
    status = match.iloc[0]['status']
    
    print(f"Match: Udinese vs Pisa")
    print(f"Raw Date: {raw_date} (Type: {type(raw_date)})")
    print(f"Status: {status}")
    
    # 3. Simulate filter_matches_by_date
    user_tz = 'UTC'
    tz = pytz.timezone(user_tz)
    
    # Convert match date to local
    match_dt = pd.to_datetime(raw_date, utc=True)
    match_local = match_dt.tz_convert(tz)
    
    # Current time
    now = pd.Timestamp.now(tz=tz)
    today = now.normalize()
    tomorrow = today + pd.Timedelta(days=1)
    
    print(f"\nUser Timezone: {user_tz}")
    print(f"Now: {now}")
    print(f"Match Local Date: {match_local}")
    print(f"Today Range: [{today}, {tomorrow})")
    
    is_today = (match_local >= today) and (match_local < tomorrow)
    print(f"Is Today: {is_today}")
    
    # 4. Run actual filter
    filtered = filter_matches_by_date(match, 'today', user_timezone=user_tz)
    print(f"\nActual Filter Result empty: {filtered.empty}")

if __name__ == "__main__":
    debug_filter()

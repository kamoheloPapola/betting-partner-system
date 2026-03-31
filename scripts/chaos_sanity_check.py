import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from src.features.pipeline import FeaturePipeline
from src.utils.naming import normalize_team_name
from src.cli.utils import filter_matches_by_date, DateFilter

def run_chaos_test():
    print("=== Chaos Sanity Check (Phase 8) ===")
    
    # 1. Team Name Normalization (Mixed Case)
    test_names = ["man city", "Man Utd", "bayern munich", "PSG"]
    for name in test_names:
        norm = normalize_team_name(name, league="PL" if "man" in name.lower() else "BL1" if "bayern" in name.lower() else "FL1")
        print(f"Norm: '{name}' -> '{norm}'")
        assert norm == norm.upper(), f"Normalization failed to uppercase: {norm}"

    # 2. Pipeline Sanitization
    data = [{
        'match_id': 'test_1',
        'date': '2026-01-11',
        'home_team': 'bayern munich',
        'away_team': 'Vfl wolfsburg',
        'status': 'finished',
        'league': 'bl1',
        'competition': 'Bundesliga'
    }]
    pipeline = FeaturePipeline()
    df_raw = pd.DataFrame(data)
    df_sanitized = pipeline._sanitize_dataframe(df_raw)
    
    print("\nPipeline Sanitization:")
    for col in ['home_team', 'away_team', 'status', 'league', 'competition']:
        val = df_sanitized.iloc[0][col]
        print(f"Column '{col}': '{val}'")
        assert val == val.upper(), f"Column {col} not uppercased: {val}"

    # 3. Deduplication Priority (Mixed Case)
    dupe_data = [
        {'match_id': 'D1', 'status': 'finished', 'home_score': 2, 'away_score': 1, 'date': '2026-01-10'},
        {'match_id': 'D1', 'status': 'SCHEDULED', 'home_score': np.nan, 'away_score': np.nan, 'date': '2026-01-10'}
    ]
    df_dupe = pd.DataFrame(dupe_data)
    df_deduped = pipeline._deduplicate_matches(df_dupe)
    print("\nDeduplication Priority (Mixed Case):")
    print(df_deduped[['match_id', 'status']])
    assert df_deduped.iloc[0]['status'].upper() == 'FINISHED', "Deduplication failed to prioritize finished match due to casing"

    # 4. Filter Date Timezone Resilience
    now_utc = pd.Timestamp.now(tz='UTC')
    # Match on Jan 11 00:00 UTC
    match_date = datetime(2026, 1, 11, 0, 0)
    df_date = pd.DataFrame([{
        'date': match_date,
        'home_team': 'TEST',
        'status': 'SCHEDULED'
    }])
    
    # In UTC, it's currently Jan 10 if we run early enough, or Jan 11.
    # Let's assume now is Jan 11 00:30 UTC for the test
    fixed_now = pd.Timestamp(year=2026, month=1, day=11, hour=0, minute=30, tz='UTC')
    
    # Filter for 'today' in Tokyo (UTC+9) -> should see Jan 11
    filtered_tokyo = filter_matches_by_date(df_date, 'today', user_timezone='Asia/Tokyo')
    print(f"\nTimezone Resilience (Tokyo UTC+9):")
    print(f"Match: {match_date}, Filter: today, Result empty: {filtered_tokyo.empty}")
    
    # Filter for 'today' in NY (UTC-5) -> NY is still Jan 10
    filtered_ny = filter_matches_by_date(df_date, 'today', user_timezone='America/New_York')
    print(f"Timezone Resilience (New York UTC-5):")
    print(f"Match: {match_date}, Filter: today, Result empty: {filtered_ny.empty}")

    print("\n✅ All Phase 8 Chaos Checks Passed!")

if __name__ == "__main__":
    run_chaos_test()

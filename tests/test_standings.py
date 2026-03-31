import pytest
import pandas as pd
from datetime import datetime, timedelta
from src.utils.standings import StandingsManager

@pytest.fixture
def mock_results(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    f = d / "results_master.csv"
    
    # Create sample results
    data = [
        # Match 1: Team A vs Team B (3-0)
        ['PL', 2025, '2025-08-16', 'Team A', 'Team B', 3, 0, 'h1'],
        # Match 2: Team C vs Team D (1-1)
        ['PL', 2025, '2025-08-16', 'Team C', 'Team D', 1, 1, 'h2'],
        # Match 3: Team A vs Team C (1-2)
        ['PL', 2025, '2025-08-23', 'Team A', 'Team C', 1, 2, 'h3'],
        # Match 4: Team B vs Team D (1-0)
        ['PL', 2025, '2025-08-23', 'Team B', 'Team D', 1, 0, 'h4'],
    ]
    df = pd.DataFrame(data, columns=['league', 'season', 'match_date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'match_hash'])
    df.to_csv(f, index=False)
    return f

def test_standings_reconstruction(mock_results):
    sm = StandingsManager(master_path=mock_results)
    
    # 1. Check table after first week (2025-08-17)
    as_of = datetime(2025, 8, 17)
    table = sm.get_table('PL', 2025, as_of)
    
    assert table['TEAM A']['rank'] == 1
    assert table['TEAM A']['points'] == 3
    assert table['TEAM A']['status_band'] == "TOP"
    
    assert table['TEAM C']['rank'] == 2 # 1 point, better alphabetical or same GD
    assert table['TEAM D']['rank'] == 3
    assert table['TEAM B']['rank'] == 4 # 0 points
    assert table['TEAM B']['status_band'] == "TOP" # 4-team mock league uses TOP band for all places

def test_safety_assertion(mock_results):
    sm = StandingsManager(master_path=mock_results)
    
    # If we pass a date that is clearly before matches, it should be empty
    table = sm.get_table('PL', 2025, datetime(2025, 8, 10))
    assert table == {}

def test_deterministic_sort(tmp_path):
    # Test tie-breaking with identical stats
    f = tmp_path / "results_master.csv"
    data = [
        ['LB', 2025, '2025-08-16', 'Z_Team', 'Y_Team', 0, 0, 'h1'],
        ['LB', 2025, '2025-08-16', 'A_Team', 'B_Team', 0, 0, 'h2'],
    ]
    df = pd.DataFrame(data, columns=['league', 'season', 'match_date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'match_hash'])
    df.to_csv(f, index=False)
    
    sm = StandingsManager(master_path=f)
    table = sm.get_table('LB', 2025, datetime(2025, 8, 20))
    
    # All have 1 point, 0 GD. Alphabetical should prevail.
    assert table['A_TEAM']['rank'] == 1
    assert table['B_TEAM']['rank'] == 2
    assert table['Y_TEAM']['rank'] == 3
    assert table['Z_TEAM']['rank'] == 4

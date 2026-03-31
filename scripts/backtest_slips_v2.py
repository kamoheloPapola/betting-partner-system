"""
Slip History Backtest using Raw football-data.co.uk CSVs.

Directly reads raw downloaded CSVs from data/results/raw/ to verify
slips from Jan 4-11, 2026.

Run: python scripts/backtest_slips_v2.py
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Output file for slip history
SLIP_HISTORY_FILE = Path("data/slips/slip_history.jsonl")
RAW_DATA_DIR = Path("data/results/raw")


def load_raw_results(start_date: str, end_date: str) -> pd.DataFrame:
    """Load all finished matches directly from raw football-data CSVs."""
    all_matches = []
    
    # League to raw file mapping (2525/26 season)
    league_files = {
        'PL': RAW_DATA_DIR / 'PL' / 'season_2526.csv',
        'PD': RAW_DATA_DIR / 'PD' / 'season_2526.csv',
        'SA': RAW_DATA_DIR / 'SA' / 'season_2526.csv',
        'BL1': RAW_DATA_DIR / 'BL1' / 'season_2526.csv',
        'FL1': RAW_DATA_DIR / 'FL1' / 'season_2526.csv',
    }
    
    for league, csv_path in league_files.items():
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, encoding='latin1')
                df['league'] = league
                # Standardize column names
                df = df.rename(columns={
                    'Date': 'date',
                    'HomeTeam': 'home_team',
                    'AwayTeam': 'away_team',
                    'FTHG': 'home_score',
                    'FTAG': 'away_score',
                    'HC': 'home_corners',
                    'AC': 'away_corners',
                    'HY': 'home_yellow',
                    'AY': 'away_yellow',
                    'HR': 'home_red',
                    'AR': 'away_red',
                })
                all_matches.append(df)
                console.print(f"  Loaded {len(df)} matches from {league}")
            except Exception as e:
                console.print(f"  [red]Error loading {league}: {e}[/red]")
    
    if not all_matches:
        return pd.DataFrame()
    
    combined = pd.concat(all_matches, ignore_index=True)
    
    # Parse dates (football-data uses DD/MM/YYYY format)
    combined['date'] = pd.to_datetime(combined['date'], dayfirst=True, errors='coerce')
    
    # Filter by date range
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    mask = (combined['date'] >= start) & (combined['date'] <= end)
    filtered = combined[mask].copy()
    
    # Calculate total cards
    if 'home_yellow' in filtered.columns:
        filtered['home_cards'] = filtered['home_yellow'].fillna(0) + filtered['home_red'].fillna(0)
        filtered['away_cards'] = filtered['away_yellow'].fillna(0) + filtered['away_red'].fillna(0)
    
    return filtered


def verify_selection(selection: Dict, results_df: pd.DataFrame) -> Tuple[str, Optional[str]]:
    """Verify a single selection against actual results."""
    home = selection.get('home_team', '').upper()
    away = selection.get('away_team', '').upper()
    market = selection.get('market', '')
    
    # Find the match
    mask = (
        (results_df['home_team'].str.upper() == home) &
        (results_df['away_team'].str.upper() == away)
    )
    match = results_df[mask]
    
    if match.empty:
        # Try fuzzy match
        for _, row in results_df.iterrows():
            if home in str(row['home_team']).upper() or str(row['home_team']).upper() in home:
                if away in str(row['away_team']).upper() or str(row['away_team']).upper() in away:
                    match = pd.DataFrame([row])
                    break
    
    if match.empty:
        return 'PENDING', f'Match not found: {home} vs {away}'
    
    match = match.iloc[0]
    
    home_score = int(match.get('home_score', 0) or 0)
    away_score = int(match.get('away_score', 0) or 0)
    home_corners = int(match.get('home_corners', 0) or 0)
    away_corners = int(match.get('away_corners', 0) or 0)
    total_corners = home_corners + away_corners
    home_cards = int(match.get('home_cards', 0) or 0)
    away_cards = int(match.get('away_cards', 0) or 0)
    total_cards = home_cards + away_cards
    
    # Market verification logic
    if 'HOME_TG_U1.5' in market:
        won = home_score <= 1
        return ('WIN' if won else 'LOSS', f"Home scored {home_score}")
    
    elif 'AWAY_TG_U1.5' in market:
        won = away_score <= 1
        return ('WIN' if won else 'LOSS', f"Away scored {away_score}")
    
    elif 'CORNERS_U11.5' in market:
        won = total_corners <= 11
        return ('WIN' if won else 'LOSS', f"Total corners: {total_corners}")
    
    elif 'CORNERS_O9.5' in market:
        won = total_corners >= 10
        return ('WIN' if won else 'LOSS', f"Total corners: {total_corners}")
    
    elif 'CARDS_U5.5' in market:
        won = total_cards <= 5
        return ('WIN' if won else 'LOSS', f"Total cards: {total_cards}")
    
    elif 'BTTS' in market:
        won = home_score > 0 and away_score > 0
        return ('WIN' if won else 'LOSS', f"Score: {home_score}-{away_score}")
    
    elif 'O2.5' in market:
        total = home_score + away_score
        won = total >= 3
        return ('WIN' if won else 'LOSS', f"Total goals: {total}")
    
    elif 'U2.5' in market:
        total = home_score + away_score
        won = total <= 2
        return ('WIN' if won else 'LOSS', f"Total goals: {total}")
    
    return 'PENDING', f"Unknown market: {market}"


def record_slip(date: str, selections: List[Dict], outcomes: List[Tuple[str, str]]) -> Dict:
    """Record a slip to the history file."""
    SLIP_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    record = {
        "date": date,
        "recorded_at": datetime.now().isoformat(),
        "selections": selections,
        "outcomes": [{"result": o[0], "reason": o[1]} for o in outcomes],
        "slip_result": "WIN" if all(o[0] == 'WIN' for o in outcomes) else (
            "LOSS" if any(o[0] == 'LOSS' for o in outcomes) else "PENDING"
        )
    }
    
    with open(SLIP_HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    
    return record


def main():
    console.print(Panel.fit(
        "[bold cyan]SLIP HISTORY BACKTEST V2[/bold cyan]\n"
        "Reading directly from raw football-data.co.uk CSVs\n"
        "Date Range: Jan 4-11, 2026",
        border_style="cyan"
    ))
    
    # Load all finished matches from RAW CSVs
    console.print("\n[bold]Loading raw match results...[/bold]")
    results_df = load_raw_results("2026-01-04", "2026-01-11")
    console.print(f"\n[green]Found {len(results_df)} matches in date range.[/green]\n")
    
    if results_df.empty:
        console.print("[red]No match data found![/red]")
        return 0
    
    # Show all matches found
    console.print("[bold]Matches in range:[/bold]")
    for _, row in results_df.iterrows():
        score = f"{int(row['home_score'])}-{int(row['away_score'])}"
        console.print(f"  {row['date'].strftime('%Y-%m-%d')} | {row['home_team']} {score} {row['away_team']} ({row['league']})")
    
    console.print()
    
    # Simulate typical Forbidden Fruit selections
    # These are the types of markets FF typically selects
    console.print("[bold]Simulating typical FF selections (HOME_TG_U1.5 for underdog/low-scoring profiles):[/bold]\n")
    
    # Summary tracking
    total_selections = 0
    selections_won = 0
    selections_lost = 0
    
    # Verify each match as if it was a HOME_TG_U1.5 selection
    for _, match in results_df.iterrows():
        selection = {
            "home_team": match['home_team'],
            "away_team": match['away_team'],
            "league": match['league'],
            "market": "HOME_TG_U1.5",
            "date": match['date'].strftime('%Y-%m-%d')
        }
        
        outcome, reason = verify_selection(selection, results_df)
        total_selections += 1
        
        if outcome == 'WIN':
            selections_won += 1
            color = "green"
        else:
            selections_lost += 1
            color = "red"
        
        console.print(f"  [{color}]{outcome}[/{color}] {match['home_team']} vs {match['away_team']}: {reason}")
    
    # Summary
    console.print()
    win_rate = (selections_won / total_selections * 100) if total_selections > 0 else 0
    
    console.print(Panel.fit(
        f"[bold green]BACKTEST COMPLETE[/bold green]\n\n"
        f"Total Matches Analyzed: {total_selections}\n"
        f"[green]Home U1.5 Wins: {selections_won}[/green]\n"
        f"[red]Home U1.5 Losses (Boomed): {selections_lost}[/red]\n\n"
        f"Win Rate (if all were HOME_TG_U1.5): {win_rate:.1f}%",
        border_style="cyan"
    ))
    
    return selections_lost


if __name__ == "__main__":
    booms = main()
    print(f"\n=== TOTAL BOOMS: {booms} ===")

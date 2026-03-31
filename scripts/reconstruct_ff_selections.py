"""
Reconstruct Forbidden Fruit Selections with >70% Confidence.

Simulates what the FF engine WOULD have selected for each day from Jan 4-11, 2026,
using the actual model predictions and a 70% confidence threshold.

Run: python scripts/reconstruct_ff_selections.py
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

# Constants
CONFIDENCE_THRESHOLD = 0.70  # 70% probability threshold
RAW_DATA_DIR = Path("data/results/raw")
SLIP_HISTORY_FILE = Path("data/slips/ff_reconstructed_history.jsonl")


def load_raw_results() -> pd.DataFrame:
    """Load all matches from raw football-data CSVs."""
    all_matches = []
    
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
                })
                all_matches.append(df)
            except Exception as e:
                console.print(f"  [red]Error loading {league}: {e}[/red]")
    
    if not all_matches:
        return pd.DataFrame()
    
    combined = pd.concat(all_matches, ignore_index=True)
    combined['date'] = pd.to_datetime(combined['date'], dayfirst=True, errors='coerce')
    
    # Calculate totals
    combined['total_corners'] = combined['home_corners'].fillna(0) + combined['away_corners'].fillna(0)
    combined['total_goals'] = combined['home_score'].fillna(0) + combined['away_score'].fillna(0)
    
    return combined


def calculate_home_u15_probability(row: pd.Series, history_df: pd.DataFrame) -> float:
    """
    Calculate probability of HOME scoring ≤1 goal based on historical form.
    
    Uses rolling average of home team's recent scoring and away team's recent conceding.
    """
    home_team = row['home_team']
    away_team = row['away_team']
    match_date = row['date']
    league = row['league']
    
    # Get home team's recent home matches (before this match)
    home_history = history_df[
        (history_df['home_team'] == home_team) & 
        (history_df['date'] < match_date) &
        (history_df['league'] == league)
    ].tail(5)
    
    # Get away team's recent away matches (before this match)
    away_history = history_df[
        (history_df['away_team'] == away_team) & 
        (history_df['date'] < match_date) &
        (history_df['league'] == league)
    ].tail(5)
    
    # Calculate home team's average goals at home
    if len(home_history) >= 3:
        home_avg_scored = home_history['home_score'].mean()
    else:
        home_avg_scored = 1.3  # League average fallback
    
    # Calculate away team's average goals conceded away
    if len(away_history) >= 3:
        away_avg_conceded = away_history['away_score'].mean()
    else:
        away_avg_conceded = 1.2  # League average fallback
    
    # Estimate expected home goals (Poisson lambda)
    expected_home_goals = (home_avg_scored + away_avg_conceded) / 2
    
    # Poisson probability of 0 or 1 goals
    # P(X <= 1) = P(X=0) + P(X=1) = e^(-λ) + λ*e^(-λ)
    prob_0 = np.exp(-expected_home_goals)
    prob_1 = expected_home_goals * np.exp(-expected_home_goals)
    prob_u15 = prob_0 + prob_1
    
    return min(prob_u15, 0.99)  # Cap at 99%


def calculate_corners_u115_probability(row: pd.Series, history_df: pd.DataFrame) -> float:
    """Calculate probability of total corners ≤11."""
    home_team = row['home_team']
    away_team = row['away_team']
    match_date = row['date']
    league = row['league']
    
    # Get recent matches for both teams
    home_matches = history_df[
        ((history_df['home_team'] == home_team) | (history_df['away_team'] == home_team)) & 
        (history_df['date'] < match_date) &
        (history_df['league'] == league)
    ].tail(5)
    
    away_matches = history_df[
        ((history_df['home_team'] == away_team) | (history_df['away_team'] == away_team)) & 
        (history_df['date'] < match_date) &
        (history_df['league'] == league)
    ].tail(5)
    
    # Calculate average total corners
    if len(home_matches) >= 3 and len(away_matches) >= 3:
        home_avg = home_matches['total_corners'].mean()
        away_avg = away_matches['total_corners'].mean()
        expected_corners = (home_avg + away_avg) / 2
    else:
        expected_corners = 10.5  # League average fallback
    
    # Simple probability estimate
    # If expected < 10, high probability of U11.5
    if expected_corners < 9:
        return 0.80
    elif expected_corners < 10:
        return 0.70
    elif expected_corners < 11:
        return 0.60
    elif expected_corners < 12:
        return 0.50
    else:
        return 0.35


def verify_selection(selection: Dict, actual_result: pd.Series) -> Tuple[str, str]:
    """Verify a selection against actual result."""
    market = selection['market']
    
    home_score = int(actual_result['home_score'] or 0)
    away_score = int(actual_result['away_score'] or 0)
    total_corners = int(actual_result.get('total_corners', 0) or 0)
    
    if 'HOME_TG_U1.5' in market:
        won = home_score <= 1
        return ('WIN' if won else 'LOSS', f"Home scored {home_score}")
    elif 'AWAY_TG_U1.5' in market:
        won = away_score <= 1
        return ('WIN' if won else 'LOSS', f"Away scored {away_score}")
    elif 'CORNERS_U11.5' in market:
        won = total_corners <= 11
        return ('WIN' if won else 'LOSS', f"Total corners: {total_corners}")
    
    return ('PENDING', 'Unknown market')


def main():
    console.print(Panel.fit(
        "[bold cyan]FORBIDDEN FRUIT RECONSTRUCTION[/bold cyan]\n"
        f"Confidence Threshold: >{CONFIDENCE_THRESHOLD*100:.0f}%\n"
        "Date Range: Jan 4-11, 2026",
        border_style="cyan"
    ))
    
    # Load all historical data
    console.print("\n[bold]Loading match data...[/bold]")
    all_matches = load_raw_results()
    console.print(f"Loaded {len(all_matches)} total matches.\n")
    
    # Filter to our analysis period
    start_date = pd.to_datetime("2026-01-04")
    end_date = pd.to_datetime("2026-01-11")
    
    target_matches = all_matches[
        (all_matches['date'] >= start_date) & 
        (all_matches['date'] <= end_date)
    ].copy()
    
    console.print(f"Found {len(target_matches)} matches in analysis period.\n")
    
    # Track results
    all_selections = []
    daily_results = {}
    
    # Process each day
    for day_offset in range(8):
        target_date = start_date + timedelta(days=day_offset)
        date_str = target_date.strftime("%Y-%m-%d")
        
        day_matches = target_matches[target_matches['date'].dt.date == target_date.date()]
        
        if day_matches.empty:
            continue
        
        console.rule(f"[bold]{date_str}[/bold] ({len(day_matches)} matches)")
        
        day_selections = []
        
        for _, match in day_matches.iterrows():
            # Calculate probabilities using historical data (excluding future)
            prob_home_u15 = calculate_home_u15_probability(match, all_matches)
            prob_corners_u115 = calculate_corners_u115_probability(match, all_matches)
            
            # Check if any market meets threshold
            best_market = None
            best_prob = 0
            
            if prob_home_u15 >= CONFIDENCE_THRESHOLD:
                best_market = "HOME_TG_U1.5"
                best_prob = prob_home_u15
            
            if prob_corners_u115 > best_prob and prob_corners_u115 >= CONFIDENCE_THRESHOLD:
                best_market = "CORNERS_U11.5"
                best_prob = prob_corners_u115
            
            if best_market:
                selection = {
                    "date": date_str,
                    "home_team": match['home_team'],
                    "away_team": match['away_team'],
                    "league": match['league'],
                    "market": best_market,
                    "confidence": best_prob
                }
                
                # Verify against actual result
                outcome, reason = verify_selection(selection, match)
                selection['outcome'] = outcome
                selection['reason'] = reason
                
                day_selections.append(selection)
                all_selections.append(selection)
                
                status_color = "green" if outcome == "WIN" else "red"
                console.print(
                    f"  [{status_color}]{outcome}[/{status_color}] "
                    f"{match['home_team']} vs {match['away_team']} | "
                    f"[cyan]{best_market}[/cyan] ({best_prob*100:.1f}%) — {reason}"
                )
        
        if not day_selections:
            console.print("  [dim]No selections met the 70% threshold[/dim]")
        
        # Track daily slip result
        if day_selections:
            all_won = all(s['outcome'] == 'WIN' for s in day_selections)
            any_lost = any(s['outcome'] == 'LOSS' for s in day_selections)
            daily_results[date_str] = {
                "selections": len(day_selections),
                "slip_result": "WIN" if all_won else ("LOSS" if any_lost else "PENDING")
            }
    
    # Save to history file
    SLIP_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SLIP_HISTORY_FILE, "w") as f:
        for sel in all_selections:
            f.write(json.dumps(sel) + "\n")
    
    # Summary
    console.print()
    
    total_selections = len(all_selections)
    wins = sum(1 for s in all_selections if s['outcome'] == 'WIN')
    losses = sum(1 for s in all_selections if s['outcome'] == 'LOSS')
    
    slip_wins = sum(1 for d in daily_results.values() if d['slip_result'] == 'WIN')
    slip_losses = sum(1 for d in daily_results.values() if d['slip_result'] == 'LOSS')
    
    console.print(Panel.fit(
        f"[bold green]RECONSTRUCTION COMPLETE[/bold green]\n\n"
        f"[bold]INDIVIDUAL SELECTIONS (>{CONFIDENCE_THRESHOLD*100:.0f}%):[/bold]\n"
        f"  Total: {total_selections}\n"
        f"  [green]Wins: {wins}[/green] | [red]Losses: {losses}[/red]\n"
        f"  Win Rate: {wins/total_selections*100:.1f}%\n\n"
        f"[bold]DAILY SLIPS:[/bold]\n"
        f"  Days with selections: {len(daily_results)}\n"
        f"  [green]Full slip wins: {slip_wins}[/green] | [red]Boomed: {slip_losses}[/red]\n\n"
        f"[dim]History saved to: {SLIP_HISTORY_FILE}[/dim]",
        border_style="cyan"
    ))
    
    return losses


if __name__ == "__main__":
    booms = main()
    print(f"\n=== SELECTIONS THAT BOOMED: {booms} ===")
